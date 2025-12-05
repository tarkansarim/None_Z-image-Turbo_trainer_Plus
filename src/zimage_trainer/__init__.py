# -*- coding: utf-8 -*-
"""
Z-Image Trainer - AC-RF (Anchor-Constrained Rectified Flow) LoRA Training

Features:
- AC-RF: 保持 Turbo 模型的直线加速结构
- 离散锚点采样: 只在关键时间步训练
- 速度回归: 直接预测 velocity
- FFT/Cosine 辅助损失

Usage:
    from zimage_trainer import ACRFTrainer, LoRANetwork
    
    # 或使用命令行:
    accelerate launch scripts/train_acrf.py --config config.toml
"""

__version__ = "0.3.0"

# Models
from .utils.zimage_utils import (
    load_transformer as load_zimage_model,
    load_vae,
    load_text_encoder_and_tokenizer,
    load_scheduler,
    create_pipeline_from_components,
)

# Networks
from .networks import LoRANetwork, create_network

# AC-RF Training
from .acrf_trainer import ACRFTrainer

# Inference
from .inference import ZImagePipeline

# Flow Matching utilities
from .flow_matching import (
    sample_timesteps,
    get_noisy_model_input_and_timesteps,
    get_z_t,
    compute_norm_opt_scale,
    compute_target_with_schedule,
    compute_loss_weighting,
)

__all__ = [
    # Version
    "__version__",
    # Models
    "load_zimage_model",
    "load_vae",
    "load_text_encoder_and_tokenizer",
    "load_scheduler",
    "create_pipeline_from_components",
    # Networks
    "LoRANetwork",
    "create_network",
    # Training (AC-RF)
    "ACRFTrainer",
    # Inference
    "ZImagePipeline",
    # Flow Matching
    "sample_timesteps",
    "get_noisy_model_input_and_timesteps",
    "get_z_t",
    "compute_norm_opt_scale",
    "compute_target_with_schedule",
    "compute_loss_weighting",
]
