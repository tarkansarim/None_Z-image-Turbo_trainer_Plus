# -*- coding: utf-8 -*-
"""
Z-Image Trainer Extensions

Modular extensions for multi-GPU, resume training, and other enhancements.
These are designed to be minimally invasive to the core training scripts,
making it easy to merge upstream changes.

Usage:
    from zimage_trainer.extensions import TrainingExtensions
    
    # In training script, create extensions early
    ext = TrainingExtensions(args, accelerator)
    
    # Use throughout training
    ext.setup_distributed()
    start_epoch, global_step = ext.load_resume_state(optimizer, scheduler, network)
    ext.save_checkpoint(network, path, dtype)
    ext.log_step(message)
"""

from .training_extensions import TrainingExtensions

__all__ = ["TrainingExtensions"]



