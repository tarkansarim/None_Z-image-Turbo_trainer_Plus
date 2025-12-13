# -*- coding: utf-8 -*-
"""
Training Extensions

A single class that provides all training enhancements:
- Multi-GPU / Distributed setup (Windows Gloo, Linux NCCL)
- Resume training with state management
- DDP-safe checkpoint saving
- Main-process-only logging

This keeps all custom logic in ONE place, making upstream merges trivial.
The core training script only needs to call a few methods.

Example minimal integration in train_acrf.py:
    
    from zimage_trainer.extensions import TrainingExtensions
    
    # After accelerator.prepare()
    ext = TrainingExtensions(args.output_dir, accelerator, args.output_name)
    
    # Resume (replaces 25+ lines of inline code)
    start_epoch, global_step = ext.load_resume_state(optimizer, lr_scheduler, network)
    
    # In training loop - logging
    ext.log_step(f"[STEP] {step}...")
    ext.log_epoch(f"Epoch {epoch}")
    
    # Saving (replaces DDP unwrap logic)
    ext.save_lora_checkpoint(network, save_path, dtype)
    ext.save_training_state(optimizer, lr_scheduler, epoch, global_step)
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Any

import torch

from zimage_trainer.utils.checkpoint_manager import CheckpointManager
from zimage_trainer.utils.distributed_utils import setup_distributed_backend

logger = logging.getLogger(__name__)


class TrainingExtensions:
    """
    Unified training extensions for multi-GPU, resume, and logging.
    
    Designed to be minimally invasive - the core training script
    only needs to call a few methods from this class.
    """
    
    def __init__(
        self,
        output_dir: str,
        accelerator: Any,
        output_name: str = "lora",
    ):
        """
        Initialize training extensions.
        
        Args:
            output_dir: Directory for checkpoints and state
            accelerator: HuggingFace Accelerator instance
            output_name: Base name for output files
        """
        self.output_dir = Path(output_dir)
        self.accelerator = accelerator
        self.output_name = output_name
        
        # Initialize checkpoint manager with output_name for per-LoRA state files
        self.ckpt_manager = CheckpointManager(output_dir, output_name)
        
        # Track if we're the main process
        self._is_main = accelerator.is_main_process
    
    # ========================================
    # Distributed Setup
    # ========================================
    
    @staticmethod
    def setup_distributed(backend: str = "auto") -> str:
        """
        Set up distributed backend before Accelerator initialization.
        
        Call this BEFORE creating Accelerator if you need explicit control.
        
        Args:
            backend: "auto", "gloo", or "nccl"
            
        Returns:
            The selected backend name
        """
        return setup_distributed_backend(backend)
    
    # ========================================
    # Resume Training
    # ========================================
    
    def load_resume_state(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        network: Any,
        resume_enabled: bool = True,
    ) -> Tuple[int, int]:
        """
        Load resume state if available and enabled.
        
        Handles all edge cases:
        - No state file: Start fresh
        - State + LoRA checkpoint: Resume properly
        - State + NO LoRA (orphan): Clean up and start fresh
        
        Args:
            optimizer: Optimizer to restore state
            scheduler: LR scheduler to restore state
            network: LoRA network to load weights into
            resume_enabled: Whether resume was requested
            
        Returns:
            Tuple of (start_epoch, global_step)
        """
        if not resume_enabled:
            return 0, 0
        
        start_epoch, global_step, lora_checkpoint = self.ckpt_manager.load_training_state(
            optimizer, scheduler, self.accelerator, output_name=self.output_name
        )
        
        # Load LoRA weights if checkpoint found
        if lora_checkpoint is not None:
            self.log(f"[RESUME] 加载 LoRA 权重: {lora_checkpoint}")
            # Unwrap network for loading (in case it's wrapped by accelerator)
            unwrapped_network = self.accelerator.unwrap_model(network)
            unwrapped_network.load_weights(str(lora_checkpoint))
            self.log("[RESUME] ✅ LoRA 权重加载成功")
            
        elif start_epoch > 0:
            # Training state exists but no LoRA checkpoint - orphaned state
            self.log_warning("[RESUME] ⚠️ 发现孤立的训练状态文件（无对应LoRA检查点）")
            self.log_warning("[RESUME] 这可能是之前训练崩溃导致的，将删除孤立状态并从头开始")
            self.ckpt_manager.delete_state()
            start_epoch = 0
            global_step = 0
            self.log("[RESUME] 已清理，从头开始训练")
        
        return start_epoch, global_step
    
    def is_training_complete(self, start_epoch: int, num_epochs: int) -> bool:
        """Check if training is already complete."""
        if start_epoch >= num_epochs:
            self.log_warning(f"[CKPT] 已完成所有 {num_epochs} 个 epoch，无需继续训练")
            return True
        return False
    
    # ========================================
    # Checkpoint Saving (DDP-safe)
    # ========================================
    
    def save_lora_checkpoint(
        self,
        network: Any,
        save_path: Path,
        dtype: torch.dtype,
    ) -> bool:
        """
        Save LoRA checkpoint (main process only, DDP-safe).
        
        Args:
            network: LoRA network (may be wrapped in DDP)
            save_path: Path to save the checkpoint
            dtype: Data type for saving
            
        Returns:
            True if saved (main process), False otherwise
        """
        if not self._is_main:
            return False
        
        # Unwrap from DDP/accelerate wrapper
        unwrapped_network = self.accelerator.unwrap_model(network)
        unwrapped_network.save_weights(save_path, dtype=dtype)
        return True
    
    def save_training_state(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
    ) -> bool:
        """
        Save training state for resume (main process only).
        
        Args:
            optimizer: Optimizer to save state from
            scheduler: Scheduler to save state from
            epoch: Current epoch (0-indexed)
            global_step: Current global step
            
        Returns:
            True if saved (main process), False otherwise
        """
        if not self._is_main:
            return False
        
        self.ckpt_manager.save_training_state(
            optimizer, scheduler, epoch, global_step, self.accelerator
        )
        return True
    
    def save_epoch_checkpoint(
        self,
        network: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        dtype: torch.dtype,
    ) -> Optional[Path]:
        """
        Save both LoRA checkpoint and training state for an epoch.
        
        Convenience method that combines save_lora_checkpoint and 
        save_training_state with proper synchronization.
        
        Args:
            network: LoRA network
            optimizer: Optimizer
            scheduler: LR scheduler
            epoch: Current epoch (0-indexed)
            global_step: Current global step
            dtype: Data type for LoRA weights
            
        Returns:
            Path to saved checkpoint if main process, None otherwise
        """
        save_path = None
        
        if self._is_main:
            save_path = self.output_dir / f"{self.output_name}_epoch{epoch+1}.safetensors"
            self.save_lora_checkpoint(network, save_path, dtype)
            self.log(f"\n[SAVE] 保存检查点 (Epoch {epoch+1}): {save_path}")
            
            self.save_training_state(optimizer, scheduler, epoch, global_step)
            self.log(f"[SAVE] ✅ 训练状态已保存 (可用于恢复训练)")
        
        # NOTE: No barrier here - following OneTrainer's pattern for performance.
        # Other GPUs continue training while master saves. DDP handles gradient sync.
        
        return save_path
    
    def save_final_checkpoint(
        self,
        network: Any,
        dtype: torch.dtype,
    ) -> Optional[Path]:
        """
        Save final LoRA checkpoint with synchronization.
        
        Args:
            network: LoRA network
            dtype: Data type for saving
            
        Returns:
            Path to saved checkpoint
        """
        final_path = self.output_dir / f"{self.output_name}_final.safetensors"
        
        if self._is_main:
            self.save_lora_checkpoint(network, final_path, dtype)
        
        # NOTE: No barrier here - following OneTrainer's pattern for performance.
        
        return final_path
    
    # ========================================
    # Logging (Main Process Only)
    # ========================================
    
    def log(self, message: str):
        """Log INFO message (main process only)."""
        if self._is_main:
            logger.info(message)
    
    def log_warning(self, message: str):
        """Log WARNING message (main process only)."""
        if self._is_main:
            logger.warning(message)
    
    def log_error(self, message: str):
        """Log ERROR message (all processes - errors are important)."""
        logger.error(message)
    
    def print_step(self, message: str):
        """Print step progress (main process only, for frontend parsing)."""
        if self._is_main:
            print(message, flush=True)
    
    def log_epoch(self, epoch: int, total_epochs: int):
        """Log epoch start (main process only)."""
        if self._is_main:
            logger.info(f"\nEpoch {epoch + 1}/{total_epochs}")
    
    def log_training_start(self, start_epoch: int, global_step: int, num_epochs: int):
        """Log training start with resume info."""
        if self._is_main:
            logger.info("\n" + "="*60)
            if start_epoch > 0:
                logger.info(f"[RESUME] ✅ 从 Epoch {start_epoch + 1}, Step {global_step} 恢复训练")
                logger.info(f"[RESUME] 剩余 Epochs: {num_epochs - start_epoch}")
            else:
                logger.info("[TARGET] 开始全新训练")
            logger.info("="*60)
    
    def log_training_complete(self, final_path: Path):
        """Log training completion."""
        if self._is_main:
            logger.info("\n" + "="*60)
            logger.info(f"[OK] 训练完成！")
            logger.info(f"最终模型: {final_path}")
            logger.info("="*60)

