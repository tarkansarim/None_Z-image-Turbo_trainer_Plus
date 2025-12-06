# -*- coding: utf-8 -*-
"""
Loss Functions for Z-Image Trainer

可用损失函数：
- FrequencyAwareLoss: 频域分离的混合损失（高频 L1 + 低频 Cosine）
- AdaptiveFrequencyLoss: 自适应权重的频域损失
- StyleStructureLoss: 结构锁风格迁移损失（SSIM + Lab 空间统计量）
- LatentStyleStructureLoss: Latent 空间的风格结构损失（省显存）
"""

from .frequency_aware_loss import FrequencyAwareLoss, AdaptiveFrequencyLoss
from .style_structure_loss import StyleStructureLoss, LatentStyleStructureLoss

__all__ = [
    "FrequencyAwareLoss",
    "AdaptiveFrequencyLoss",
    "StyleStructureLoss",
    "LatentStyleStructureLoss",
]

