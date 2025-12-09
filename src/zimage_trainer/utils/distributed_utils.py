# -*- coding: utf-8 -*-
"""
Distributed Training Utilities

Provides cross-platform distributed backend configuration.
- Windows: Uses Gloo backend (libuv-free, works with PyTorch nightly)
- Linux: Uses NCCL backend (optimal GPU-to-GPU performance)

Designed to be modular and easy to merge with upstream updates.

Usage:
    from zimage_trainer.utils.distributed_utils import setup_distributed_backend
    
    # Auto-detect best backend for current platform
    backend = setup_distributed_backend("auto")
    
    # Or force a specific backend
    backend = setup_distributed_backend("gloo")
"""

import os
import sys
import platform
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def setup_distributed_backend(backend: str = "auto") -> str:
    """
    Configure the distributed backend based on platform and user preference.
    
    This function sets environment variables that PyTorch/Accelerate will use
    when initializing distributed training.
    
    Args:
        backend: One of "auto", "gloo", "nccl"
            - "auto": Gloo on Windows, NCCL on Linux
            - "gloo": Force Gloo backend (works on Windows + Linux)
            - "nccl": Force NCCL backend (Linux only, best GPU performance)
    
    Returns:
        The selected backend name
    """
    is_windows = platform.system() == "Windows"
    
    # CRITICAL: Disable libuv FIRST before any torch.distributed imports
    # This must be set before PyTorch's distributed module initializes
    if is_windows:
        os.environ["USE_LIBUV"] = "0"
        # Also set for subprocesses
        os.environ["TORCH_DISTRIBUTED_USE_LIBUV"] = "0"
    
    # Determine backend
    if backend == "auto":
        selected = "gloo" if is_windows else "nccl"
    else:
        selected = backend.lower()
    
    # Validate backend choice
    if selected == "nccl" and is_windows:
        logger.warning("[DIST] NCCL is not supported on Windows, falling back to Gloo")
        selected = "gloo"
    
    if selected not in ("gloo", "nccl"):
        logger.warning(f"[DIST] Unknown backend '{selected}', falling back to auto")
        selected = "gloo" if is_windows else "nccl"
    
    # Set environment variables
    os.environ["ACCELERATE_TORCH_DEVICE"] = "cuda"
    
    if selected == "gloo":
        # Gloo backend configuration
        os.environ["DISTRIBUTED_BACKEND"] = "gloo"
        
        # Gloo-specific optimizations
        os.environ["GLOO_SOCKET_IFNAME"] = os.environ.get("GLOO_SOCKET_IFNAME", "")
        
        # Force file-based store instead of TCP store (avoids libuv entirely)
        os.environ["TORCH_DISTRIBUTED_INIT_METHOD"] = "file"
        
        logger.info(f"[DIST] Configured Gloo backend (platform: {platform.system()}, libuv=disabled)")
        
    elif selected == "nccl":
        # NCCL backend configuration
        os.environ["DISTRIBUTED_BACKEND"] = "nccl"
        
        # NCCL optimizations
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["NCCL_P2P_DISABLE"] = os.environ.get("NCCL_P2P_DISABLE", "0")
        
        logger.info(f"[DIST] Configured NCCL backend (platform: {platform.system()})")
    
    return selected


def get_accelerate_launch_args(
    num_gpus: int,
    backend: str = "auto",
    mixed_precision: str = "fp16",
    main_process_port: int = 29500,
) -> List[str]:
    """
    Build command-line arguments for accelerate launch.
    
    Args:
        num_gpus: Number of GPUs to use
        backend: Distributed backend ("auto", "gloo", "nccl")
        mixed_precision: Mixed precision mode ("no", "fp16", "bf16")
        main_process_port: Port for main process communication
    
    Returns:
        List of command-line arguments for accelerate launch
    """
    is_windows = platform.system() == "Windows"
    
    # Resolve auto backend
    if backend == "auto":
        backend = "gloo" if is_windows else "nccl"
    
    args = [
        "--multi_gpu",
        f"--num_processes={num_gpus}",
        f"--mixed_precision={mixed_precision}",
        f"--main_process_port={main_process_port}",
    ]
    
    # Add backend-specific args for Windows
    if is_windows and backend == "gloo":
        # These help Gloo work better on Windows
        args.extend([
            "--dynamo_backend=no",  # Disable dynamo for stability
        ])
    
    return args


def get_num_available_gpus() -> int:
    """
    Get the number of available CUDA GPUs.
    
    Returns:
        Number of GPUs, or 0 if CUDA is not available
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


def is_distributed_available() -> bool:
    """
    Check if distributed training is available.
    
    Returns:
        True if distributed training can be used
    """
    try:
        import torch.distributed as dist
        return dist.is_available()
    except ImportError:
        return False


def get_distributed_info() -> dict:
    """
    Get current distributed training information.
    
    Returns:
        Dict with rank, world_size, backend info
    """
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return {
                "initialized": True,
                "rank": dist.get_rank(),
                "world_size": dist.get_world_size(),
                "backend": dist.get_backend(),
            }
    except ImportError:
        pass
    
    return {
        "initialized": False,
        "rank": 0,
        "world_size": 1,
        "backend": None,
    }

