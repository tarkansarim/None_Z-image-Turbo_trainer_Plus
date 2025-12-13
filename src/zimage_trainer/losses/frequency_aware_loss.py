# -*- coding: utf-8 -*-
"""
🎨 频域感知损失函数 (Frequency-Aware Loss)

基于频域分离的解耦学习策略：
- 高频增强：L1 Loss 强化纹理/边缘细节
- 低频锁定：Cosine Loss 锁定结构/光影方向

数学公式：
L_total = L_base + λ_hf * ||x̂_high - x_high||₁ + λ_lf * (1 - cos(x̂_low, x_low))

核心优势：
- 解决微调时"顾此失彼"问题（提升细节却搞坏构图）
- 高频用 L1（保持边缘锐利），低频用 Cosine（保住光影结构）
- 在 x0 空间做频域分析，避免 v 空间含噪干扰
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class FrequencyAwareLoss(nn.Module):
    """
    频域分离的混合损失函数
    
    设计理念：
    1. 从 v-prediction 反推 x̂₀（在干净 latent 空间做频域分析）
    2. 使用降采样-上采样快速分离高频/低频
    3. 高频用 L1 Loss（边缘锐利），低频用 Cosine Loss（方向一致）
    """
    
    def __init__(
        self,
        alpha_hf: float = 1.0,      # 高频增强权重
        beta_lf: float = 0.2,       # 低频锁定权重
        base_weight: float = 1.0,   # 基础 loss 权重
        downsample_factor: int = 4, # 低频提取的降采样因子
        use_laplacian: bool = False, # 是否使用拉普拉斯金字塔（更精确但更慢）
        lf_magnitude_weight: float = 0.0,  # 低频幅度约束（防止发灰）
    ):
        super().__init__()
        self.alpha_hf = alpha_hf
        self.beta_lf = beta_lf
        self.base_weight = base_weight
        self.downsample_factor = downsample_factor
        self.use_laplacian = use_laplacian
        self.lf_magnitude_weight = lf_magnitude_weight
        
        logger.info(f"[FreqLoss] 初始化频域感知损失")
        logger.info(f"  高频权重 (alpha_hf): {alpha_hf}")
        logger.info(f"  低频权重 (beta_lf): {beta_lf}")
        logger.info(f"  降采样因子: {downsample_factor}")
        
    def get_low_freq(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取低频分量（结构/光影）
        
        使用降采样-上采样方法：
        - 比高斯模糊更快
        - 在 GPU 上极度优化
        - 对 Latent (64x64 或 128x128) 足够过滤纹理
        """
        h, w = x.shape[-2:]
        target_h = max(1, h // self.downsample_factor)
        target_w = max(1, w // self.downsample_factor)
        
        # 降采样（滤除高频）
        x_small = F.adaptive_avg_pool2d(x, (target_h, target_w))
        
        # 上采样还原尺寸
        x_low = F.interpolate(x_small, size=(h, w), mode='bilinear', align_corners=False)
        
        return x_low
    
    def get_high_freq(self, x: torch.Tensor, x_low: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        提取高频分量（纹理/边缘）
        
        高频 = 原始 - 低频
        """
        if x_low is None:
            x_low = self.get_low_freq(x)
        return x - x_low
    
    def reconstruct_x0_from_v(
        self,
        v_pred: torch.Tensor,
        noisy_latents: torch.Tensor,
        sigmas: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 v-prediction 反推 x̂₀
        
        Z-Image 公式：
        x_t = (1 - σ) * x_0 + σ * noise
        v = noise - x_0
        
        反推：
        x_0 = x_t - σ * v
        """
        # 扩展 sigma 维度以匹配 latents
        sigma_broadcast = sigmas.view(-1, 1, 1, 1)
        
        # 反推 x0
        pred_x0 = noisy_latents - sigma_broadcast * v_pred
        
        return pred_x0
    
    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        num_train_timesteps: int = 1000,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算频域感知损失
        
        Args:
            pred_v: 模型预测的速度 v (B, C, H, W)
            target_v: 目标速度 v (B, C, H, W)
            noisy_latents: 加噪后的 latents x_t (B, C, H, W)
            timesteps: 时间步 (B,)
            num_train_timesteps: 总训练时间步数
            return_components: 是否返回各分量 loss
            
        Returns:
            total_loss: 总损失
            components (optional): 各分量 loss 字典
        """
        # 保存原始 dtype，确保所有计算结果都转换回来
        original_dtype = pred_v.dtype
        
        # 转换为 float32 进行计算（避免混合精度问题）
        pred_v_fp32 = pred_v.float()
        target_v_fp32 = target_v.float()
        noisy_latents_fp32 = noisy_latents.float()
        
        # 1. 基础 Loss（保证模型不崩）
        base_loss = F.mse_loss(pred_v_fp32, target_v_fp32, reduction="mean")
        
        # 2. 计算 sigma（Z-Image: sigma = timestep / 1000）
        sigmas = timesteps.float() / num_train_timesteps
        
        # 3. 反推 x0（在干净 latent 空间做频域分析）
        sigma_broadcast = sigmas.view(-1, 1, 1, 1)
        pred_x0 = noisy_latents_fp32 - sigma_broadcast * pred_v_fp32
        target_x0 = noisy_latents_fp32 - sigma_broadcast * target_v_fp32
        
        # 4. 频域分离
        pred_low = self.get_low_freq(pred_x0)
        pred_high = pred_x0 - pred_low
        
        target_low = self.get_low_freq(target_x0)
        target_high = target_x0 - target_low
        
        # 5. 高频 Loss：L1（保持边缘锐利）
        loss_hf = F.l1_loss(pred_high, target_high, reduction="mean")
        
        # 6. 低频 Loss：Cosine Similarity（锁定方向）
        pred_low_flat = pred_low.view(pred_low.shape[0], -1)
        target_low_flat = target_low.view(target_low.shape[0], -1)
        
        cos_sim = F.cosine_similarity(pred_low_flat, target_low_flat, dim=1)
        loss_lf_direction = (1.0 - cos_sim).mean()
        
        # 可选：低频幅度约束（防止发灰）
        loss_lf_magnitude = torch.zeros(1, device=pred_v.device, dtype=torch.float32).squeeze()
        if self.lf_magnitude_weight > 0:
            pred_norm = pred_low_flat.norm(dim=1)
            target_norm = target_low_flat.norm(dim=1)
            loss_lf_magnitude = F.mse_loss(pred_norm, target_norm)
        
        loss_lf = loss_lf_direction + self.lf_magnitude_weight * loss_lf_magnitude
        
        # 7. 总 Loss（在 float32 下计算，然后转回原始 dtype）
        total_loss = (
            self.base_weight * base_loss +
            self.alpha_hf * loss_hf +
            self.beta_lf * loss_lf
        )
        
        # 转换回原始 dtype
        total_loss = total_loss.to(original_dtype)
        
        if return_components:
            components = {
                "base_loss": base_loss.to(original_dtype),
                "loss_hf": loss_hf.to(original_dtype),
                "loss_lf": loss_lf.to(original_dtype),
                "loss_lf_direction": loss_lf_direction.to(original_dtype),
                "loss_lf_magnitude": loss_lf_magnitude.to(original_dtype),
                "total_loss": total_loss,
            }
            return total_loss, components
        
        return total_loss
    
    # === CUSTOM: Per-sample weighted loss for per-dataset loss settings ===
    def forward_per_sample(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        num_train_timesteps: int = 1000,
        sample_weights: Optional[Dict[str, torch.Tensor]] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        计算带有 per-sample 权重的频域感知损失
        
        Args:
            sample_weights: 每个样本的权重，字典包含:
                - 'alpha_hf': (B,) 高频权重
                - 'beta_lf': (B,) 低频权重
        """
        batch_size = pred_v.shape[0]
        original_dtype = pred_v.dtype
        
        if sample_weights is None:
            return self.forward(pred_v, target_v, noisy_latents, timesteps, num_train_timesteps, return_components)
        
        # 转换为 float32
        pred_v_fp32 = pred_v.float()
        target_v_fp32 = target_v.float()
        noisy_latents_fp32 = noisy_latents.float()
        
        # 基础 Loss (per-sample)
        base_loss_per_sample = F.mse_loss(pred_v_fp32, target_v_fp32, reduction='none').mean(dim=[1, 2, 3])
        
        # 计算 sigma
        sigmas = timesteps.float() / num_train_timesteps
        sigma_broadcast = sigmas.view(-1, 1, 1, 1)
        
        # 反推 x0
        pred_x0 = noisy_latents_fp32 - sigma_broadcast * pred_v_fp32
        target_x0 = noisy_latents_fp32 - sigma_broadcast * target_v_fp32
        
        # 频域分离
        pred_low = self.get_low_freq(pred_x0)
        pred_high = pred_x0 - pred_low
        target_low = self.get_low_freq(target_x0)
        target_high = target_x0 - target_low
        
        # 高频 Loss (per-sample L1)
        loss_hf_per_sample = F.l1_loss(pred_high, target_high, reduction='none').mean(dim=[1, 2, 3])
        
        # 低频 Loss (per-sample cosine)
        pred_low_flat = pred_low.view(batch_size, -1)
        target_low_flat = target_low.view(batch_size, -1)
        cos_sim_per_sample = F.cosine_similarity(pred_low_flat, target_low_flat, dim=1)
        loss_lf_per_sample = 1.0 - cos_sim_per_sample
        
        # 获取 per-sample 权重
        w_alpha_hf = sample_weights.get('alpha_hf', torch.ones(batch_size, device=pred_v.device) * self.alpha_hf)
        w_beta_lf = sample_weights.get('beta_lf', torch.ones(batch_size, device=pred_v.device) * self.beta_lf)
        
        # 计算加权 per-sample 损失
        loss_per_sample = (
            self.base_weight * base_loss_per_sample +
            w_alpha_hf * loss_hf_per_sample +
            w_beta_lf * loss_lf_per_sample
        )
        
        total_loss = loss_per_sample.mean().to(original_dtype)
        
        if return_components:
            components = {
                "base_loss": base_loss_per_sample.mean().to(original_dtype),
                "loss_hf": (w_alpha_hf * loss_hf_per_sample).mean().to(original_dtype),
                "loss_lf": (w_beta_lf * loss_lf_per_sample).mean().to(original_dtype),
                "total_loss": total_loss,
            }
            return total_loss, components
        
        return total_loss


class AdaptiveFrequencyLoss(FrequencyAwareLoss):
    """
    自适应频域损失
    
    根据训练阶段动态调整高频/低频权重：
    - 训练初期：侧重低频（学习整体结构）
    - 训练后期：侧重高频（精炼细节）
    """
    
    def __init__(
        self,
        alpha_hf_start: float = 0.1,
        alpha_hf_end: float = 1.0,
        beta_lf_start: float = 0.5,
        beta_lf_end: float = 0.1,
        warmup_steps: int = 100,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.alpha_hf_start = alpha_hf_start
        self.alpha_hf_end = alpha_hf_end
        self.beta_lf_start = beta_lf_start
        self.beta_lf_end = beta_lf_end
        self.warmup_steps = warmup_steps
        self.current_step = 0
        
    def update_step(self, step: int):
        """更新当前步数，调整权重"""
        self.current_step = step
        
        if step < self.warmup_steps:
            # Warmup 阶段：从 start 过渡到 end
            progress = step / self.warmup_steps
            self.alpha_hf = self.alpha_hf_start + progress * (self.alpha_hf_end - self.alpha_hf_start)
            self.beta_lf = self.beta_lf_start + progress * (self.beta_lf_end - self.beta_lf_start)
        else:
            self.alpha_hf = self.alpha_hf_end
            self.beta_lf = self.beta_lf_end

