# -*- coding: utf-8 -*-
"""
xformers 内存高效注意力工具模块

提供 xformers memory_efficient_attention 的封装和管理功能。
支持自动检测、回退机制和性能优化。

Features:
- 自动检测 xformers 可用性
- 支持多种注意力后端切换
- 内存高效的注意力计算
- 与 SDPA/Flash Attention 的兼容性
"""

import logging
from typing import Optional, Tuple, Dict, Any
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# xformers 可用性检测
XFORMERS_AVAILABLE = False
XFORMERS_VERSION = None

try:
    import xformers
    import xformers.ops as xops
    XFORMERS_AVAILABLE = True
    XFORMERS_VERSION = getattr(xformers, "__version__", "unknown")
    logger.info(f"✅ xformers {XFORMERS_VERSION} 已加载")
except ImportError:
    logger.info("⚠️ xformers 未安装，将使用 PyTorch 原生注意力")
except Exception as e:
    logger.warning(f"⚠️ xformers 加载失败: {e}")


@lru_cache(maxsize=1)
def check_xformers_availability() -> Dict[str, Any]:
    """
    检查 xformers 可用性和功能支持
    
    Returns:
        包含可用性信息的字典
    """
    info = {
        "available": XFORMERS_AVAILABLE,
        "version": XFORMERS_VERSION,
        "memory_efficient_attention": False,
        "flash_attention": False,
        "cutlass": False,
        "triton": False,
        "cuda_available": torch.cuda.is_available(),
        "recommended_backend": "torch",
    }
    
    if not XFORMERS_AVAILABLE:
        return info
    
    try:
        # 检查各种后端支持
        info["memory_efficient_attention"] = hasattr(xops, "memory_efficient_attention")
        
        # 检查 Flash Attention 支持
        if hasattr(xops, "MemoryEfficientAttentionFlashAttentionOp"):
            info["flash_attention"] = True
        
        # 检查 CUTLASS 支持
        if hasattr(xops, "MemoryEfficientAttentionCutlassOp"):
            info["cutlass"] = True
        
        # 检查 Triton 支持
        try:
            import triton
            info["triton"] = True
        except ImportError:
            pass
        
        # 推荐后端
        if info["flash_attention"] and torch.cuda.is_available():
            cc = torch.cuda.get_device_capability()
            if cc[0] >= 8:  # SM80+ (A100, etc.)
                info["recommended_backend"] = "xformers_flash"
            else:
                info["recommended_backend"] = "xformers"
        elif info["memory_efficient_attention"]:
            info["recommended_backend"] = "xformers"
        
    except Exception as e:
        logger.warning(f"xformers 功能检测失败: {e}")
    
    return info


def xformers_memory_efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    p: float = 0.0,
) -> torch.Tensor:
    """
    使用 xformers 执行内存高效的注意力计算
    
    Args:
        query: Query tensor (B, N, H, D) 或 (B, H, N, D)
        key: Key tensor (B, M, H, D) 或 (B, H, M, D)
        value: Value tensor (B, M, H, D) 或 (B, H, M, D)
        attn_bias: 可选的注意力偏置
        scale: 缩放因子，默认为 1/sqrt(D)
        p: Dropout 概率
        
    Returns:
        注意力输出 tensor
    """
    if not XFORMERS_AVAILABLE:
        # 回退到 PyTorch SDPA
        return _pytorch_scaled_dot_product_attention(
            query, key, value, attn_bias, scale, p
        )
    
    try:
        # xformers 期望格式: (B, N, H, D)
        # 检测并转换格式
        if query.dim() == 4 and query.shape[1] != query.shape[2]:
            # 可能是 (B, H, N, D) 格式，需要转换
            if query.shape[1] < query.shape[2]:  # H < N
                query = query.transpose(1, 2)  # (B, N, H, D)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)
                need_transpose_back = True
            else:
                need_transpose_back = False
        else:
            need_transpose_back = False
        
        # 调用 xformers memory_efficient_attention
        output = xops.memory_efficient_attention(
            query, key, value,
            attn_bias=attn_bias,
            scale=scale,
            p=p if p > 0 else 0.0,
        )
        
        # 转换回原格式
        if need_transpose_back:
            output = output.transpose(1, 2)
        
        return output
        
    except Exception as e:
        logger.warning(f"xformers attention 失败，回退到 PyTorch: {e}")
        return _pytorch_scaled_dot_product_attention(
            query, key, value, attn_bias, scale, p
        )


def _pytorch_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_bias: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    p: float = 0.0,
) -> torch.Tensor:
    """
    PyTorch 原生 SDPA 实现（回退方案）
    """
    # 确保格式为 (B, H, N, D)
    if query.dim() == 4:
        if query.shape[2] > query.shape[1]:  # (B, N, H, D) -> (B, H, N, D)
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            need_transpose = True
        else:
            need_transpose = False
    else:
        need_transpose = False
    
    # 使用 PyTorch SDPA
    output = F.scaled_dot_product_attention(
        query, key, value,
        attn_mask=attn_bias,
        dropout_p=p if p > 0 else 0.0,
        scale=scale,
    )
    
    if need_transpose:
        output = output.transpose(1, 2)
    
    return output


class XFormersAttention(nn.Module):
    """
    xformers 注意力层封装
    
    可以作为 nn.MultiheadAttention 的替代品使用
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        enable_xformers: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.enable_xformers = enable_xformers and XFORMERS_AVAILABLE
        
        # QKV 投影
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            query: (B, N, C) if batch_first else (N, B, C)
            key: (B, M, C) if batch_first else (M, B, C)
            value: (B, M, C) if batch_first else (M, B, C)
            attn_mask: 可选的注意力掩码
            need_weights: 是否返回注意力权重（xformers 模式下不支持）
            
        Returns:
            (output, attn_weights) 元组
        """
        if not self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        batch_size, seq_len, _ = query.shape
        
        # QKV 投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim)
        
        # 注意力计算
        if self.enable_xformers:
            # xformers 格式: (B, N, H, D)
            attn_output = xformers_memory_efficient_attention(
                q, k, v,
                attn_bias=attn_mask,
                scale=self.scale,
                p=self.dropout if self.training else 0.0,
            )
        else:
            # PyTorch SDPA 格式: (B, H, N, D)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                scale=self.scale,
            )
            attn_output = attn_output.transpose(1, 2)
        
        # 重塑回原始维度
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        
        # xformers 不支持返回权重
        attn_weights = None
        if need_weights and not self.enable_xformers:
            logger.warning("xformers 模式下不支持返回注意力权重")
        
        return output, attn_weights


def enable_xformers_for_model(
    model: nn.Module,
    enable: bool = True,
    force: bool = False,
) -> bool:
    """
    为 diffusers 模型启用/禁用 xformers
    
    Args:
        model: 要修改的模型
        enable: 是否启用 xformers
        force: 是否强制启用（忽略兼容性检查）
        
    Returns:
        是否成功启用
    """
    if not XFORMERS_AVAILABLE and enable:
        logger.warning("xformers 不可用，无法启用")
        return False
    
    # 检查是否是 diffusers 模型
    if hasattr(model, "enable_xformers_memory_efficient_attention"):
        try:
            if enable:
                model.enable_xformers_memory_efficient_attention()
                logger.info(f"✅ 已为 {model.__class__.__name__} 启用 xformers")
            else:
                model.disable_xformers_memory_efficient_attention()
                logger.info(f"✅ 已为 {model.__class__.__name__} 禁用 xformers")
            return True
        except Exception as e:
            if force:
                logger.warning(f"强制启用 xformers 失败: {e}")
            else:
                logger.error(f"xformers 启用失败: {e}")
            return False
    
    # 检查子模块
    enabled_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "enable_xformers_memory_efficient_attention"):
            try:
                if enable:
                    module.enable_xformers_memory_efficient_attention()
                else:
                    module.disable_xformers_memory_efficient_attention()
                enabled_count += 1
            except Exception:
                pass
    
    if enabled_count > 0:
        logger.info(f"✅ 已为 {enabled_count} 个模块{'启用' if enable else '禁用'} xformers")
        return True
    
    logger.warning(f"模型 {model.__class__.__name__} 不支持 xformers")
    return False


def get_optimal_attention_backend(
    gpu_name: Optional[str] = None,
    memory_gb: Optional[float] = None,
    sequence_length: int = 4096,
) -> str:
    """
    根据硬件和任务获取最优注意力后端
    
    Args:
        gpu_name: GPU 名称
        memory_gb: GPU 显存 (GB)
        sequence_length: 序列长度
        
    Returns:
        推荐的后端: "xformers", "flash", "sdpa", "torch"
    """
    # 获取硬件信息
    if gpu_name is None and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    
    if memory_gb is None and torch.cuda.is_available():
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    # 检查 xformers 可用性
    xf_info = check_xformers_availability()
    
    # 决策逻辑
    if not torch.cuda.is_available():
        return "torch"
    
    # 长序列优先使用 xformers
    if sequence_length > 8192:
        if xf_info["available"]:
            return "xformers"
        return "sdpa"
    
    # 检查 GPU 计算能力
    cc = torch.cuda.get_device_capability()
    
    # SM80+ (A100, H100, RTX 30xx/40xx)
    if cc[0] >= 8:
        if xf_info["flash_attention"]:
            return "xformers"  # xformers flash attention
        return "flash"  # PyTorch flash attention
    
    # SM70-79 (V100, T4, RTX 20xx)
    if cc[0] >= 7:
        if xf_info["available"]:
            return "xformers"
        return "sdpa"
    
    # 较旧的 GPU
    return "torch"


def apply_xformers_to_transformer(
    transformer: nn.Module,
    enable: bool = True,
) -> None:
    """
    为 Transformer 模型应用 xformers 优化
    
    专门针对 ZImageTransformer2DModel 等 diffusers 模型
    
    Args:
        transformer: Transformer 模型
        enable: 是否启用
    """
    if not XFORMERS_AVAILABLE and enable:
        logger.warning("xformers 不可用")
        return
    
    # 尝试使用 diffusers 内置方法
    if enable_xformers_for_model(transformer, enable):
        return
    
    # 手动替换注意力层
    if enable:
        _replace_attention_with_xformers(transformer)
    else:
        logger.info("跳过 xformers 替换")


def _replace_attention_with_xformers(model: nn.Module) -> int:
    """
    手动替换模型中的注意力层为 xformers 版本
    
    Returns:
        替换的层数
    """
    replaced = 0
    
    for name, module in model.named_modules():
        # 检查是否是标准注意力层
        if isinstance(module, nn.MultiheadAttention):
            # 创建 xformers 替代
            xf_attn = XFormersAttention(
                embed_dim=module.embed_dim,
                num_heads=module.num_heads,
                dropout=module.dropout,
                bias=module.in_proj_bias is not None,
                batch_first=module.batch_first,
                enable_xformers=True,
            )
            
            # 复制权重
            with torch.no_grad():
                if module.in_proj_weight is not None:
                    # 分离 QKV 权重
                    qkv_weight = module.in_proj_weight
                    d = module.embed_dim
                    xf_attn.q_proj.weight.copy_(qkv_weight[:d])
                    xf_attn.k_proj.weight.copy_(qkv_weight[d:2*d])
                    xf_attn.v_proj.weight.copy_(qkv_weight[2*d:])
                    
                    if module.in_proj_bias is not None:
                        qkv_bias = module.in_proj_bias
                        xf_attn.q_proj.bias.copy_(qkv_bias[:d])
                        xf_attn.k_proj.bias.copy_(qkv_bias[d:2*d])
                        xf_attn.v_proj.bias.copy_(qkv_bias[2*d:])
                
                xf_attn.out_proj.weight.copy_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    xf_attn.out_proj.bias.copy_(module.out_proj.bias)
            
            # 替换模块
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            
            if parent_name:
                parent = model
                for part in parent_name.split("."):
                    parent = getattr(parent, part)
                setattr(parent, attr_name, xf_attn)
            else:
                setattr(model, attr_name, xf_attn)
            
            replaced += 1
            logger.debug(f"替换注意力层: {name}")
    
    if replaced > 0:
        logger.info(f"✅ 替换了 {replaced} 个注意力层为 xformers 版本")
    
    return replaced


def benchmark_attention_backends(
    batch_size: int = 4,
    seq_len: int = 4096,
    num_heads: int = 16,
    head_dim: int = 64,
    num_runs: int = 100,
    warmup_runs: int = 10,
) -> Dict[str, float]:
    """
    对比各注意力后端的性能
    
    Args:
        batch_size: 批次大小
        seq_len: 序列长度
        num_heads: 注意力头数
        head_dim: 头维度
        num_runs: 运行次数
        warmup_runs: 预热次数
        
    Returns:
        各后端的平均时间 (ms)
    """
    import time
    
    if not torch.cuda.is_available():
        logger.warning("CUDA 不可用，跳过基准测试")
        return {}
    
    device = torch.device("cuda")
    dtype = torch.float16
    
    # 准备测试数据
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    
    results = {}
    
    # 测试 PyTorch SDPA
    def run_sdpa():
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        return F.scaled_dot_product_attention(q_t, k_t, v_t)
    
    # 预热
    for _ in range(warmup_runs):
        run_sdpa()
    torch.cuda.synchronize()
    
    # 计时
    start = time.time()
    for _ in range(num_runs):
        run_sdpa()
    torch.cuda.synchronize()
    results["sdpa"] = (time.time() - start) / num_runs * 1000
    
    # 测试 xformers
    if XFORMERS_AVAILABLE:
        def run_xformers():
            return xops.memory_efficient_attention(q, k, v)
        
        # 预热
        for _ in range(warmup_runs):
            run_xformers()
        torch.cuda.synchronize()
        
        # 计时
        start = time.time()
        for _ in range(num_runs):
            run_xformers()
        torch.cuda.synchronize()
        results["xformers"] = (time.time() - start) / num_runs * 1000
    
    # 打印结果
    logger.info("=" * 50)
    logger.info(f"注意力后端基准测试 (B={batch_size}, N={seq_len}, H={num_heads}, D={head_dim})")
    logger.info("=" * 50)
    for backend, time_ms in sorted(results.items(), key=lambda x: x[1]):
        logger.info(f"  {backend}: {time_ms:.3f} ms")
    logger.info("=" * 50)
    
    return results


# 便捷函数
def is_xformers_available() -> bool:
    """检查 xformers 是否可用"""
    return XFORMERS_AVAILABLE


def get_xformers_version() -> Optional[str]:
    """获取 xformers 版本"""
    return XFORMERS_VERSION

