# -*- coding: utf-8 -*-
"""Simple test for multi-GPU with Gloo on Windows."""

import os
import sys
import socket
import platform
import datetime
from pathlib import Path

# CRITICAL: Fix Docker Desktop's kubernetes.docker.internal issue
# Monkey-patch socket to resolve this hostname to 127.0.0.1
_original_getaddrinfo = socket.getaddrinfo
def _patched_getaddrinfo(host, port, *args, **kwargs):
    if host == 'kubernetes.docker.internal':
        host = '127.0.0.1'
    return _original_getaddrinfo(host, port, *args, **kwargs)
socket.getaddrinfo = _patched_getaddrinfo

_original_gethostbyname = socket.gethostbyname
def _patched_gethostbyname(hostname):
    if hostname == 'kubernetes.docker.internal':
        return '127.0.0.1'
    return _original_gethostbyname(hostname)
socket.gethostbyname = _patched_gethostbyname

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.multiprocessing as mp
import torch.distributed as dist


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return sock.getsockname()[1]


def worker(rank, world_size, store_file):
    """Test worker that just initializes and does a simple all_reduce."""
    print(f"[Rank {rank}] Starting...", flush=True)
    
    # Set device
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    
    # Use FileStore instead of TCP - avoids all network issues on Windows
    backend = 'gloo' if platform.system() == 'Windows' else 'nccl'
    init_method = f'file:///{store_file}'
    
    print(f"[Rank {rank}] Initializing {backend} via FileStore", flush=True)
    
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=60),
    )
    
    print(f"[Rank {rank}] Initialized! Device: {torch.cuda.get_device_name(device)}", flush=True)
    
    # Simple test - all_reduce
    tensor = torch.ones(10).to(device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    
    print(f"[Rank {rank}] all_reduce result: {tensor[0].item()} (expected: {world_size})", flush=True)
    
    # Cleanup
    dist.barrier()
    dist.destroy_process_group()
    
    print(f"[Rank {rank}] Done!", flush=True)


def main():
    import tempfile
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs")
    
    if num_gpus < 2:
        print("Need at least 2 GPUs for this test")
        return
    
    # Use 2 GPUs for test
    world_size = 2
    
    if platform.system() == 'Windows':
        os.environ['USE_LIBUV'] = '0'
        mp.set_start_method('spawn', force=True)
    
    # Use FileStore - create a temp file for the store
    # This avoids all TCP/network issues on Windows
    store_file = os.path.join(tempfile.gettempdir(), f"pytorch_dist_store_{os.getpid()}")
    # Remove if exists from previous run
    if os.path.exists(store_file):
        os.remove(store_file)
    
    print(f"Using FileStore: {store_file}")
    print(f"Spawning {world_size} processes...")
    
    try:
        mp.spawn(worker, args=(world_size, store_file), nprocs=world_size, join=True)
    finally:
        # Cleanup store file
        if os.path.exists(store_file):
            os.remove(store_file)
    
    print("SUCCESS! Multi-GPU test passed.")


if __name__ == '__main__':
    main()

