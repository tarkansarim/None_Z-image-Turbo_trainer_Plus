# -*- coding: utf-8 -*-
"""
Multi-GPU Training Wrapper for Z-Image Turbo Trainer

Uses torch.multiprocessing.spawn() for Windows-compatible multi-GPU training.
Based on OneTrainer's proven approach that avoids accelerate launch issues.

Uses FileStore for rendezvous to avoid all TCP/network issues on Windows.

Usage:
    python scripts/train_multi_gpu.py --config configs/current_training.toml --num_gpus 2
"""

import os
import sys
import tempfile
import platform
import datetime
import argparse
import traceback
import signal
import atexit
from pathlib import Path

# === CUSTOM: Track child processes for cleanup ===
_child_processes = []

def _cleanup_children():
    """Kill any remaining child processes on exit"""
    global _child_processes
    for p in _child_processes:
        if p.is_alive():
            print(f"[Cleanup] Terminating child process {p.pid}", flush=True)
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()

def _signal_handler(signum, frame):
    """Handle termination signals"""
    print(f"\n[Launcher] Received signal {signum}, cleaning up...", flush=True)
    _cleanup_children()
    sys.exit(1)

# Register cleanup handlers
atexit.register(_cleanup_children)
if platform.system() != 'Windows':
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
else:
    # Windows has limited signal support
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def train_worker(rank: int, world_size: int, store_file: str, config_path: str, args_dict: dict):
    """
    Worker process entry point.
    
    Args:
        rank: Process rank (0 = master)
        world_size: Total number of processes
        store_file: Path to FileStore file
        config_path: Path to training config
        args_dict: Additional arguments
    """
    import time
    
    # Clean up any existing process group
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
            time.sleep(1.0)
        except:
            pass
    
    # Select backend (Windows = gloo, Linux = nccl)
    backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
    
    # Use FileStore - avoids all TCP/network issues on Windows
    init_method = f'file:///{store_file}'
    
    # Set device
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)
    
    print(f"[Rank {rank}] Initializing {backend} via FileStore", flush=True)
    
    # Initialize process group
    # 24-hour timeout for long caching operations
    timeout = datetime.timedelta(hours=24)
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    
    print(f"[Rank {rank}] Device: {device} ({torch.cuda.get_device_name(device)})", flush=True)
    
    # Synchronize
    dist.barrier()
    if rank == 0:
        print("[Master] All GPUs synchronized!", flush=True)
    
    try:
        # Set environment variables that Accelerator will read
        os.environ['RANK'] = str(rank)
        os.environ['LOCAL_RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        
        # Add scripts directory to path for imports
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        
        # Import here to avoid issues with multiprocessing
        import train_acrf
        
        # Build args for train_acrf
        sys.argv = [
            'train_acrf.py',
            '--config', config_path,
            '--distributed_backend', backend,
        ]
        
        # Add resume flag if specified
        if args_dict.get('resume', False):
            sys.argv.append('--resume')
        
        # Run training
        train_acrf.main()
        
    except Exception as e:
        print(f"[Rank {rank}] Error: {e}", flush=True)
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            try:
                dist.barrier()
                dist.destroy_process_group()
            except:
                pass


def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Training Launcher')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--num_gpus', type=int, default=0, help='Number of GPUs (0=all)')
    parser.add_argument('--resume', action='store_true', help='Resume training')
    args = parser.parse_args()
    
    # Determine number of GPUs
    num_gpus = args.num_gpus
    if num_gpus <= 0:
        num_gpus = torch.cuda.device_count()
    
    print(f"[Launcher] Detected {torch.cuda.device_count()} GPUs, using {num_gpus}")
    
    if num_gpus < 2:
        print("[Launcher] Single GPU mode - running directly")
        # Just run single GPU training
        sys.argv = ['train_acrf.py', '--config', args.config]
        if args.resume:
            sys.argv.append('--resume')
        
        # Add scripts directory to path
        scripts_dir = Path(__file__).parent
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        
        from train_acrf import main as train_main
        train_main()
        return
    
    # Use spawn for Windows compatibility
    if platform.system() == 'Windows':
        os.environ['USE_LIBUV'] = '0'
        mp.set_start_method('spawn', force=True)
    
    # Create FileStore file for rendezvous (avoids all TCP issues)
    store_file = os.path.join(tempfile.gettempdir(), f"zimage_dist_store_{os.getpid()}")
    if os.path.exists(store_file):
        os.remove(store_file)
    
    config_path = str(Path(args.config).absolute())
    args_dict = {'resume': args.resume}
    
    print(f"[Launcher] Using FileStore: {store_file}")
    print(f"[Launcher] Spawning {num_gpus} processes...")
    
    try:
        # Spawn all processes
        mp.spawn(
            train_worker,
            args=(num_gpus, store_file, config_path, args_dict),
            nprocs=num_gpus,
            join=True,
        )
        print("[Launcher] Training complete!")
    finally:
        # Cleanup store file
        if os.path.exists(store_file):
            try:
                os.remove(store_file)
            except:
                pass


if __name__ == '__main__':
    main()

