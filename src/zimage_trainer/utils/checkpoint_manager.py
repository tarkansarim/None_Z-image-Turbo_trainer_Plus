# -*- coding: utf-8 -*-
"""
Checkpoint Manager for Resume Training

Space-efficient checkpoint management that saves only the latest training state.
Designed to be modular and easy to merge with upstream updates.

Usage:
    from zimage_trainer.utils.checkpoint_manager import CheckpointManager
    
    # Create manager
    ckpt_manager = CheckpointManager(output_dir)
    
    # Load state if resuming (including LoRA weights)
    start_epoch, start_step, lora_path = ckpt_manager.load_training_state(optimizer, scheduler)
    if lora_path:
        network.load_weights(lora_path)
    
    # Save state after each epoch
    ckpt_manager.save_training_state(optimizer, scheduler, epoch, step)
"""

import os
import re
import torch
import logging
from pathlib import Path
from typing import Optional, Tuple, Any

logger = logging.getLogger(__name__)

# State file names
TRAINING_STATE_FILE = "training_state.pt"  # Legacy fallback


class CheckpointManager:
    """
    Manages training checkpoints for resume functionality.
    
    Saves optimizer state, scheduler state, epoch, step, and RNG states
    in a single file that gets overwritten each epoch (space efficient).
    
    State files are saved per-LoRA name to allow independent resume for different trainings.
    """
    
    def __init__(self, output_dir: str, output_name: str = None):
        """
        Initialize checkpoint manager.
        
        Args:
            output_dir: Directory where checkpoints are saved
            output_name: Base name for the LoRA (used to create per-LoRA state files)
        """
        self.output_dir = Path(output_dir)
        self.output_name = output_name
        
        # Per-LoRA state file: training_state_{name}.pt
        if output_name:
            self.state_path = self.output_dir / f"training_state_{output_name}.pt"
            logger.info(f"[CKPT] Using per-LoRA state file: {self.state_path.name}")
        else:
            # Fallback to legacy filename if no name provided
            self.state_path = self.output_dir / TRAINING_STATE_FILE
            logger.info(f"[CKPT] Using legacy state file: {self.state_path.name}")
        
    def has_resume_state(self) -> bool:
        """Check if a resume state file exists."""
        return self.state_path.exists()
    
    def find_latest_lora_checkpoint(self, output_name: str = None) -> Optional[Path]:
        """
        Find the latest LoRA checkpoint file in the output directory.
        
        Args:
            output_name: Base name of the output files (uses self.output_name if not provided)
            
        Returns:
            Path to the latest LoRA checkpoint, or None if not found
        """
        if not self.output_dir.exists():
            return None
        
        # Use provided name or fall back to instance name
        name_to_match = output_name or self.output_name
        
        # Find all safetensors files that look like checkpoints
        # Pattern: {name}_epoch{N}.safetensors or {name}_final.safetensors
        checkpoint_files = []
        
        for f in self.output_dir.glob("*.safetensors"):
            # Skip files that don't look like our checkpoints
            name = f.stem
            
            # If we have a specific name to match, only consider files starting with that name
            if name_to_match:
                if not name.startswith(name_to_match):
                    continue
            
            # Match epoch checkpoints: name_epoch123
            epoch_match = re.search(r'_epoch(\d+)$', name)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                checkpoint_files.append((f, epoch_num, False))  # (path, epoch, is_final)
            # Match final checkpoint: name_final
            elif name.endswith('_final'):
                # Final is always the latest if it exists
                checkpoint_files.append((f, float('inf'), True))
        
        if not checkpoint_files:
            if name_to_match:
                logger.info(f"[CKPT] No checkpoints found for '{name_to_match}'")
            return None
        
        # Sort by epoch number (final = infinity, so it comes last)
        checkpoint_files.sort(key=lambda x: x[1], reverse=True)
        latest = checkpoint_files[0][0]
        
        logger.info(f"[CKPT] Found latest LoRA checkpoint for '{name_to_match}': {latest.name}")
        return latest
    
    def save_training_state(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        epoch: int,
        global_step: int,
        accelerator: Optional[Any] = None,
    ) -> None:
        """
        Save training state for resume.
        
        Args:
            optimizer: The optimizer to save state from
            scheduler: The learning rate scheduler to save state from
            epoch: Current epoch number (0-indexed)
            global_step: Current global step count
            accelerator: Optional Accelerator instance for distributed training
        """
        state = {
            "epoch": epoch,
            "global_step": global_step,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            # Save RNG states for reproducibility
            "rng_state": {
                "python": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        
        # For distributed training, only save on main process
        # NOTE: No barrier after save - following OneTrainer's pattern for performance.
        # Other GPUs continue training while master saves. DDP handles gradient sync.
        if accelerator is not None:
            if accelerator.is_main_process:
                torch.save(state, self.state_path)
                logger.info(f"[CKPT] Saved training state: epoch={epoch+1}, step={global_step}")
        else:
            torch.save(state, self.state_path)
            logger.info(f"[CKPT] Saved training state: epoch={epoch+1}, step={global_step}")
    
    def load_training_state(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        accelerator: Optional[Any] = None,
        output_name: str = None,
    ) -> Tuple[int, int, Optional[Path]]:
        """
        Load training state for resume.
        
        Args:
            optimizer: The optimizer to load state into
            scheduler: The learning rate scheduler to load state into
            accelerator: Optional Accelerator instance for distributed training
            output_name: Base name for finding LoRA checkpoints
            
        Returns:
            Tuple of (start_epoch, start_step, lora_checkpoint_path).
            Returns (0, 0, None) if no state found.
            
        IMPORTANT: Caller MUST load the returned LoRA checkpoint into the network!
        """
        # First, find the latest LoRA checkpoint
        lora_checkpoint = self.find_latest_lora_checkpoint(output_name)
        
        if not self.has_resume_state():
            if lora_checkpoint:
                # LoRA checkpoint exists but no training state
                # This is a common case - user trained without resume enabled
                logger.warning("="*60)
                logger.warning("[CKPT] ⚠️  WARNING: Resume requested but NO training state file found!")
                logger.warning(f"[CKPT] Found LoRA checkpoint: {lora_checkpoint.name}")
                logger.warning("[CKPT] This likely means previous training didn't save state (resume was disabled).")
                logger.warning("[CKPT] ⚠️  Starting from EPOCH 0 with fresh optimizer state!")
                logger.warning("[CKPT] The LoRA weights will NOT be loaded (no state to sync with).")
                logger.warning("[CKPT] To continue from that checkpoint properly, you would need to:")
                logger.warning("[CKPT]   1. Delete existing checkpoints OR")
                logger.warning("[CKPT]   2. Use the checkpoint for inference, not resume training")
                logger.warning("="*60)
            else:
                logger.info("[CKPT] No resume state found, starting fresh")
            return 0, 0, None
        
        try:
            # Load state
            state = torch.load(self.state_path, map_location="cpu", weights_only=False)
            
            # Restore optimizer state
            optimizer.load_state_dict(state["optimizer_state_dict"])
            
            # Restore scheduler state
            if state.get("scheduler_state_dict") is not None and hasattr(scheduler, 'load_state_dict'):
                scheduler.load_state_dict(state["scheduler_state_dict"])
            
            # Restore RNG states
            if "rng_state" in state:
                rng = state["rng_state"]
                if rng.get("python") is not None:
                    torch.set_rng_state(rng["python"])
                if rng.get("cuda") is not None and torch.cuda.is_available():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            
            epoch = state["epoch"]
            global_step = state["global_step"]
            
            # Resume from NEXT epoch (the saved epoch was completed)
            start_epoch = epoch + 1
            
            logger.info(f"[CKPT] ✅ Resumed training state: epoch={epoch+1} (completed), step={global_step}")
            logger.info(f"[CKPT] Will continue from epoch={start_epoch+1}")
            
            if lora_checkpoint:
                logger.info(f"[CKPT] ✅ Found matching LoRA checkpoint: {lora_checkpoint.name}")
            else:
                logger.warning("[CKPT] ⚠️  Training state found but no LoRA checkpoint - weights will be random!")
            
            return start_epoch, global_step, lora_checkpoint
            
        except Exception as e:
            logger.error(f"[CKPT] Failed to load training state: {e}")
            logger.warning("[CKPT] Starting fresh due to load failure")
            return 0, 0, None
    
    def delete_state(self) -> bool:
        """
        Delete the training state file.
        
        Returns:
            True if deleted, False if file didn't exist
        """
        if self.state_path.exists():
            self.state_path.unlink()
            logger.info(f"[CKPT] Deleted training state: {self.state_path}")
            return True
        return False
    
    def get_state_info(self) -> Optional[dict]:
        """
        Get information about saved state without loading full tensors.
        
        Returns:
            Dict with epoch, global_step info, or None if no state exists
        """
        if not self.has_resume_state():
            return None
            
        try:
            state = torch.load(self.state_path, map_location="cpu", weights_only=False)
            return {
                "epoch": state.get("epoch", 0),
                "global_step": state.get("global_step", 0),
                "file_size_mb": self.state_path.stat().st_size / (1024 * 1024),
            }
        except Exception as e:
            logger.warning(f"[CKPT] Could not read state info: {e}")
            return None

