# -*- coding: utf-8 -*-
"""
Multi-GPU Training Launcher

Uses torch.multiprocessing.spawn() for Windows-compatible multi-GPU training.
Based on OneTrainer's approach which avoids libuv/TCPStore issues.

This module provides a reliable way to launch multi-GPU training on Windows
using the Gloo backend without relying on accelerate launch.
"""

import os
import sys
import socket
import platform
import datetime
import traceback
from typing import Optional, List, Callable, Any

import torch
import torch.multiprocessing as mp


def find_free_port() -> int:
    """Find a free TCP port for distributed training."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(('127.0.0.1', 0))
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            return sock.getsockname()[1]
    except Exception:
        return 29500  # Fallback to common default


def setup_distributed_environment():
    """
    Set up distributed environment variables before spawning processes.
    Must be called from the main process before spawn().
    """
    # Use explicit loopback to avoid kubernetes.docker.internal issues on Windows
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
    
    # Find and set a free port if not already set
    if 'MASTER_PORT' not in os.environ or not os.environ['MASTER_PORT']:
        free_port = find_free_port()
        os.environ['MASTER_PORT'] = str(free_port)
    
    # Disable libuv on Windows
    if platform.system() == 'Windows':
        os.environ['USE_LIBUV'] = '0'
        os.environ['TORCH_DISTRIBUTED_USE_LIBUV'] = '0'
    
    return os.environ['MASTER_ADDR'], int(os.environ['MASTER_PORT'])


def _worker_process(
    spawn_rank: int,
    world_size: int,
    devices: List[torch.device],
    train_fn: Callable,
    train_args: tuple,
    train_kwargs: dict,
):
    """
    Worker process entry point. Initializes distributed and calls the training function.
    
    Args:
        spawn_rank: Rank assigned by spawn (0 to world_size-2 for workers)
        world_size: Total number of processes
        devices: List of torch.device objects for each rank
        train_fn: The training function to call
        train_args: Positional arguments for train_fn
        train_kwargs: Keyword arguments for train_fn
    """
    import time
    
    # Clean up any existing process group
    needs_wait = False
    if torch.distributed.is_initialized():
        try:
            torch.distributed.destroy_process_group()
            needs_wait = True
        except Exception:
            pass
    
    if needs_wait:
        time.sleep(2.0)
    
    # Worker processes get rank 1, 2, ... (rank 0 is main process)
    rank = spawn_rank + 1
    device = devices[rank]
    
    # 24-hour timeout for long caching operations
    timeout = datetime.timedelta(hours=24)
    
    # Get master address and port from environment
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    init_method = f'tcp://{master_addr}:{master_port}'
    
    # Select backend
    backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
    
    print(f"[Worker {rank}] Initializing distributed: {backend} @ {init_method}", flush=True)
    
    # Initialize process group
    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timeout,
        backend=backend,
        init_method=init_method
    )
    torch.cuda.set_device(device.index)
    
    print(f"[Worker {rank}] Device: {device} ({torch.cuda.get_device_name()})", flush=True)
    
    try:
        # Call the training function with rank information
        train_fn(
            rank=rank,
            world_size=world_size,
            device=device,
            *train_args,
            **train_kwargs
        )
    except Exception as e:
        traceback.print_exc()
        raise
    finally:
        # Cleanup
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            except Exception:
                pass


def _master_process(
    world_size: int,
    devices: List[torch.device],
    train_fn: Callable,
    train_args: tuple,
    train_kwargs: dict,
):
    """
    Master process (rank 0) entry point.
    """
    import time
    
    rank = 0
    device = devices[rank]
    
    timeout = datetime.timedelta(hours=24)
    
    master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
    master_port = os.environ.get('MASTER_PORT', '29500')
    init_method = f'tcp://{master_addr}:{master_port}'
    
    backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
    
    print(f"[Master {rank}] Initializing distributed: {backend} @ {init_method}", flush=True)
    
    torch.distributed.init_process_group(
        rank=rank,
        world_size=world_size,
        device_id=device,
        timeout=timeout,
        backend=backend,
        init_method=init_method
    )
    torch.cuda.set_device(device.index)
    
    # Synchronization barrier
    print(f"[Master {rank}] Synchronizing GPUs...", flush=True)
    torch.distributed.barrier()
    
    for r in range(world_size):
        if r == rank:
            print(f"GPU #{r}  device: {devices[r]} ({torch.cuda.get_device_name(devices[r])})  "
                  f"backend: {torch.distributed.get_backend()}  world_size: {world_size}", flush=True)
        torch.distributed.barrier()
    
    print("[Master] GPUs synchronized.", flush=True)
    
    try:
        train_fn(
            rank=rank,
            world_size=world_size,
            device=device,
            *train_args,
            **train_kwargs
        )
    except Exception as e:
        traceback.print_exc()
        raise
    finally:
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
                torch.distributed.destroy_process_group()
            except Exception:
                pass


def launch_multi_gpu(
    train_fn: Callable,
    num_gpus: int = 0,
    device_ids: Optional[List[int]] = None,
    train_args: tuple = (),
    train_kwargs: dict = None,
):
    """
    Launch multi-GPU training using torch.multiprocessing.spawn().
    
    This function works reliably on Windows with Gloo backend.
    
    Args:
        train_fn: Training function to call. Must accept (rank, world_size, device, *args, **kwargs)
        num_gpus: Number of GPUs to use (0 = all available)
        device_ids: Specific GPU indices to use (overrides num_gpus)
        train_args: Additional positional arguments for train_fn
        train_kwargs: Additional keyword arguments for train_fn
    
    Example:
        def train(rank, world_size, device, config_path):
            # Your training code here
            pass
        
        launch_multi_gpu(train, num_gpus=2, train_args=("config.toml",))
    """
    if train_kwargs is None:
        train_kwargs = {}
    
    # Set up environment
    master_addr, master_port = setup_distributed_environment()
    print(f"[Launcher] Master: {master_addr}:{master_port}", flush=True)
    
    # Determine devices
    if device_ids is not None:
        devices = [torch.device('cuda', i) for i in device_ids]
        world_size = len(devices)
    else:
        if num_gpus <= 0:
            num_gpus = torch.cuda.device_count()
        devices = [torch.device('cuda', i) for i in range(num_gpus)]
        world_size = num_gpus
    
    if world_size < 2:
        print("[Launcher] Only 1 GPU available, running single-GPU mode", flush=True)
        # Just run directly on single GPU
        train_fn(rank=0, world_size=1, device=devices[0], *train_args, **train_kwargs)
        return
    
    print(f"[Launcher] Launching {world_size} processes on devices: {devices}", flush=True)
    
    # Use spawn start method for Windows compatibility
    if platform.system() == 'Windows':
        mp.set_start_method('spawn', force=True)
    
    # Spawn worker processes (world_size - 1 workers)
    workers = mp.spawn(
        _worker_process,
        args=(world_size, devices, train_fn, train_args, train_kwargs),
        nprocs=world_size - 1,
        join=False
    )
    
    # Run master process (rank 0) in main process
    try:
        _master_process(world_size, devices, train_fn, train_args, train_kwargs)
    finally:
        # Wait for workers to complete
        workers.join()


# Convenience function for use from command line
def main():
    """Command-line entry point for testing multi-GPU setup."""
    def test_fn(rank, world_size, device):
        print(f"[Test] Rank {rank}/{world_size} on {device}: {torch.cuda.get_device_name(device)}", flush=True)
        
        # Simple all-reduce test
        tensor = torch.ones(10).cuda(device)
        torch.distributed.all_reduce(tensor)
        print(f"[Test] Rank {rank}: all_reduce result = {tensor[0].item()}", flush=True)
    
    launch_multi_gpu(test_fn, num_gpus=0)


if __name__ == "__main__":
    main()

