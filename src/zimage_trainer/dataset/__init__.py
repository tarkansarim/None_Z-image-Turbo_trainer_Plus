# -*- coding: utf-8 -*-
"""Dataset utilities for Z-Image training."""

from .config_utils import DatasetConfig, load_dataset_config

__all__ = ["DatasetConfig", "load_dataset_config"]


def create_dataloader(dataset_config: DatasetConfig, **kwargs):
    """Create dataloader from config (placeholder)."""
    # This is a simplified placeholder
    # Full implementation would load cached latents and text embeddings
    raise NotImplementedError("Full dataloader implementation needed")
