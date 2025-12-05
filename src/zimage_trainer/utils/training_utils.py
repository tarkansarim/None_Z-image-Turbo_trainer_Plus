# -*- coding: utf-8 -*-
"""
Training utilities for Z-Image.

Includes:
- Multiple optimizers (AdamW, AdamW8bit, Adafactor, Prodigy)
- Gradient clipping
- Module offloading for low VRAM
"""

import gc
import logging
import math
from typing import Dict, List, Optional, Union, Callable
from contextlib import contextmanager

import torch
from torch import nn
from safetensors.torch import save_file

logger = logging.getLogger(__name__)


# ============================================================================
# Optimizers
# ============================================================================

def get_optimizer(
    params: Union[List[Dict], List[nn.Parameter]],
    optimizer_type: str = "AdamW",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Create optimizer.
    
    Supported optimizers:
    - AdamW: Standard PyTorch AdamW
    - AdamW8bit: 8-bit AdamW from bitsandbytes (memory efficient)
    - Adafactor: Memory-efficient optimizer from transformers (no momentum states!)
    - Prodigy: Adaptive learning rate optimizer
    
    Args:
        params: Model parameters
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        **kwargs: Additional optimizer arguments
        
    Returns:
        Optimizer instance
    """
    opt_type = optimizer_type.lower().replace("_", "").replace("-", "")
    
    if opt_type in ["adamw", "adam"]:
        return torch.optim.AdamW(
            params, lr=learning_rate, weight_decay=weight_decay,
            betas=kwargs.get("betas", (0.9, 0.999)),
        )
    
    elif opt_type in ["adamw8bit", "adam8bit"]:
        try:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(
                params, lr=learning_rate, weight_decay=weight_decay,
            )
        except ImportError:
            logger.warning("bitsandbytes not available, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    
    elif opt_type in ["adafactor", "adafac"]:
        # Adafactor is extremely memory efficient - no momentum states!
        # Good for low VRAM training
        try:
            from transformers import Adafactor
            
            # Adafactor settings for LoRA/Finetune
            optimizer = Adafactor(
                params,
                lr=learning_rate,
                scale_parameter=kwargs.get("scale_parameter", False),
                relative_step=kwargs.get("relative_step", False),
                warmup_init=kwargs.get("warmup_init", False),
                weight_decay=weight_decay,
            )
            logger.info("Using Adafactor optimizer (memory efficient, no momentum states)")
            return optimizer
        except ImportError:
            logger.warning("transformers not available for Adafactor, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    
    elif opt_type in ["prodigy"]:
        try:
            import prodigyopt
            optimizer = prodigyopt.Prodigy(
                params,
                lr=learning_rate if learning_rate != 1e-4 else 1.0,  # Prodigy uses lr=1.0 by default
                weight_decay=weight_decay,
                d_coef=kwargs.get("d_coef", 1.0),
            )
            logger.info("Using Prodigy optimizer (adaptive learning rate)")
            return optimizer
        except ImportError:
            logger.warning("prodigyopt not available, falling back to AdamW")
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    
    elif opt_type in ["sgd"]:
        return torch.optim.SGD(
            params, lr=learning_rate, weight_decay=weight_decay,
            momentum=kwargs.get("momentum", 0.9),
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 0,
    num_cycles: int = 1,
    **kwargs,
):
    """Create learning rate scheduler."""
    from torch.optim.lr_scheduler import LambdaLR
    
    sched_type = scheduler_type.lower().replace("_", "")
    
    if sched_type == "constant":
        return LambdaLR(optimizer, lambda _: 1.0)
    
    elif sched_type in ["cosine", "cosinewithrestarts"]:
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            progress = (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(progress * math.pi)))
        return LambdaLR(optimizer, lr_lambda)
    
    elif sched_type == "linear":
        def lr_lambda(step):
            if step < num_warmup_steps:
                return step / max(1, num_warmup_steps)
            return max(0.0, 1.0 - (step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps))
        return LambdaLR(optimizer, lr_lambda)
    
    else:
        return LambdaLR(optimizer, lambda _: 1.0)


# ============================================================================
# Gradient Clipping
# ============================================================================

class GradientClipper:
    """
    Gradient clipping utilities.
    
    NOTE: For Z-Image training with Norm_opt, gradient clipping is usually
    NOT needed because the loss is already normalized by k².
    Only use this if you encounter training instability (NaN, loss explosion).
    
    Supports:
    - Max norm clipping
    - Value clipping
    - Adaptive clipping based on historical gradients
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        clip_type: str = "norm",
        adaptive: bool = False,
        adaptive_factor: float = 0.9,
    ):
        """
        Initialize gradient clipper.
        
        Args:
            max_norm: Maximum gradient norm (for norm clipping)
            clip_type: "norm" or "value"
            adaptive: Use adaptive clipping based on EMA of gradient norms
            adaptive_factor: EMA factor for adaptive clipping
        """
        self.max_norm = max_norm
        self.clip_type = clip_type
        self.adaptive = adaptive
        self.adaptive_factor = adaptive_factor
        self.grad_norm_ema = None
    
    def clip(self, parameters) -> float:
        """
        Clip gradients and return the original norm.
        
        Args:
            parameters: Model parameters (iterable)
            
        Returns:
            Original gradient norm before clipping
        """
        if self.clip_type == "norm":
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        elif self.clip_type == "value":
            torch.nn.utils.clip_grad_value_(parameters, self.max_norm)
            # Compute norm for logging
            grad_norm = self._compute_grad_norm(parameters)
        else:
            grad_norm = self._compute_grad_norm(parameters)
        
        # Update adaptive threshold
        if self.adaptive:
            if self.grad_norm_ema is None:
                self.grad_norm_ema = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            else:
                gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
                self.grad_norm_ema = self.adaptive_factor * self.grad_norm_ema + (1 - self.adaptive_factor) * gn
                # Adjust max_norm based on EMA
                self.max_norm = max(0.1, min(10.0, self.grad_norm_ema * 2.0))
        
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    
    def _compute_grad_norm(self, parameters) -> float:
        """Compute total gradient norm."""
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5


def clip_grad_norm(
    parameters,
    max_norm: float = 1.0,
    norm_type: float = 2.0,
) -> float:
    """
    Simple gradient clipping function.
    
    NOTE: For Z-Image with Norm_opt, set max_norm=0 to disable.
    Norm_opt already normalizes gradients via loss / k².
    
    Args:
        parameters: Model parameters
        max_norm: Maximum norm value (0 = disabled)
        norm_type: Type of norm (default: L2)
        
    Returns:
        Original gradient norm (or current norm if disabled)
    """
    if max_norm <= 0:
        # Disabled - just compute norm for logging
        total_norm = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm += p.grad.data.norm(norm_type).item() ** norm_type
        return total_norm ** (1.0 / norm_type)
    
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type).item()


def should_use_gradient_clipping(config: Dict) -> bool:
    """
    Check if gradient clipping should be used based on config.
    
    Returns False if:
    - gradient_clip_norm is 0 or not set
    - Using Norm_opt (which already normalizes gradients)
    """
    clip_norm = config.get("training", {}).get("gradient_clip_norm", 0.0)
    return clip_norm > 0


# ============================================================================
# Module Offloading (for low VRAM)
# ============================================================================

class ModuleOffloader:
    """
    Module offloading utility for low VRAM training.
    
    Moves modules between CPU and GPU to save memory.
    Only keeps the active module on GPU during forward/backward.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        offload_device: Union[str, torch.device] = "cpu",
        verbose: bool = False,
    ):
        """
        Initialize offloader.
        
        Args:
            device: Main compute device (GPU)
            offload_device: Offload device (CPU)
            verbose: Print offload operations
        """
        self.device = torch.device(device)
        self.offload_device = torch.device(offload_device)
        self.verbose = verbose
        self._offloaded_modules: Dict[str, nn.Module] = {}
    
    def register(self, name: str, module: nn.Module):
        """Register a module for offloading."""
        self._offloaded_modules[name] = module
        if self.verbose:
            logger.info(f"Registered module for offloading: {name}")
    
    def offload(self, name: str):
        """Move module to CPU."""
        if name in self._offloaded_modules:
            module = self._offloaded_modules[name]
            module.to(self.offload_device)
            if self.verbose:
                logger.debug(f"Offloaded {name} to {self.offload_device}")
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def load(self, name: str):
        """Move module to GPU."""
        if name in self._offloaded_modules:
            module = self._offloaded_modules[name]
            module.to(self.device)
            if self.verbose:
                logger.debug(f"Loaded {name} to {self.device}")
    
    def offload_all(self):
        """Offload all registered modules."""
        for name in self._offloaded_modules:
            self.offload(name)
    
    def load_all(self):
        """Load all registered modules."""
        for name in self._offloaded_modules:
            self.load(name)
    
    @contextmanager
    def use(self, name: str):
        """
        Context manager to temporarily load a module.
        
        Usage:
            with offloader.use("vae"):
                output = vae(input)
        """
        self.load(name)
        try:
            yield self._offloaded_modules.get(name)
        finally:
            self.offload(name)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


class SequentialOffloader:
    """
    Sequential module offloader for transformer blocks.
    
    Keeps only one block on GPU at a time during forward pass.
    Useful for very large models on limited VRAM.
    """
    
    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        offload_device: Union[str, torch.device] = "cpu",
    ):
        self.device = torch.device(device)
        self.offload_device = torch.device(offload_device)
        self.current_block = None
    
    def process_blocks(
        self,
        blocks: nn.ModuleList,
        hidden_states: torch.Tensor,
        block_fn: Callable,
    ) -> torch.Tensor:
        """
        Process transformer blocks sequentially with offloading.
        
        Args:
            blocks: List of transformer blocks
            hidden_states: Input hidden states
            block_fn: Function to call on each block
            
        Returns:
            Output hidden states
        """
        for i, block in enumerate(blocks):
            # Load current block
            block.to(self.device)
            
            # Process
            hidden_states = block_fn(block, hidden_states)
            
            # Offload current block
            block.to(self.offload_device)
            
            # Clear cache
            if (i + 1) % 5 == 0:  # Clear every 5 blocks
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        return hidden_states


# ============================================================================
# Memory utilities
# ============================================================================

def get_gpu_memory_info() -> Dict[str, float]:
    """Get GPU memory usage information."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "total_gb": total,
        "free_gb": total - reserved,
        "available": True,
    }


def print_gpu_memory():
    """Print GPU memory usage."""
    info = get_gpu_memory_info()
    if info.get("available"):
        logger.info(f"GPU Memory: {info['allocated_gb']:.2f}GB allocated, "
                   f"{info['reserved_gb']:.2f}GB reserved, "
                   f"{info['free_gb']:.2f}GB free / {info['total_gb']:.2f}GB total")
    else:
        logger.info("GPU not available")


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ============================================================================
# Checkpoint utilities
# ============================================================================

def save_checkpoint(
    state_dict: Dict[str, torch.Tensor],
    path: str,
    dtype: torch.dtype = torch.bfloat16,
    metadata: Optional[Dict[str, str]] = None,
):
    """Save checkpoint to safetensors."""
    converted = {k: v.to(dtype) if isinstance(v, torch.Tensor) else v for k, v in state_dict.items()}
    save_file(converted, path, metadata=metadata)
    logger.info(f"Saved: {path}")


def load_checkpoint(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load checkpoint from safetensors."""
    from safetensors.torch import load_file
    return load_file(path, device=device)
