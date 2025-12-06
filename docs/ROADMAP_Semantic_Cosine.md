# 🎯 语义聚焦 Cosine Loss 开发计划

> **目标**：使用 SAM3 (Segment Anything Model 3) 实现语义分割，让 Cosine Loss 可以精准聚焦到特定区域（人脸、服装、光影等），而不是全局约束。

## 📋 功能概述

### 当前问题

现有的 Cosine Loss 是**全局约束**：

```python
# 当前实现
loss_cosine = 1 - cosine_similarity(v_pred.flatten(), v_target.flatten())
```

- 对整个图像统一约束
- 无法区分语义区域
- 高权重会导致整体"锁死"，降低多样性

### 目标效果

实现**语义聚焦的局部 Cosine Loss**：

```python
# 目标实现
loss_cosine = Σ (weight_i * cosine_loss(v_pred * mask_i, v_target * mask_i))
```

| 区域 | 约束强度 | 效果 |
|------|---------|------|
| 人脸 | 高 | 五官精确，表情保持 |
| 身材 | 中 | 体型比例锁定 |
| 服装 | 可调 | 服装细节精确 |
| 背景 | 低/关闭 | 背景自由变化 |
| 光影 | 可调 | 氛围一致性 |

---

## 🔧 技术方案

### Phase 1: SAM3 集成

#### 1.1 模型选择

| 模型 | 参数量 | 速度 | 精度 | 推荐场景 |
|------|--------|------|------|---------|
| SAM3-Tiny | ~10M | 快 | 中 | 实时预览 |
| SAM3-Base | ~90M | 中 | 高 | 缓存生成 |
| SAM3-Large | ~300M | 慢 | 最高 | 高质量训练 |

**推荐**：使用 SAM3-Base，在缓存阶段预生成分割掩码。

#### 1.2 分割类别

```python
SEMANTIC_CATEGORIES = {
    "face": {
        "keywords": ["face", "head", "eyes", "nose", "mouth"],
        "default_weight": 1.0,
        "description": "人脸区域"
    },
    "body": {
        "keywords": ["body", "torso", "arms", "legs"],
        "default_weight": 0.5,
        "description": "身体轮廓"
    },
    "clothing": {
        "keywords": ["dress", "shirt", "pants", "clothes"],
        "default_weight": 0.8,
        "description": "服装区域"
    },
    "hair": {
        "keywords": ["hair", "bangs"],
        "default_weight": 0.6,
        "description": "发型区域"
    },
    "background": {
        "keywords": ["background", "sky", "wall", "floor"],
        "default_weight": 0.1,
        "description": "背景区域"
    },
    "lighting": {
        "keywords": ["light", "shadow", "highlight"],
        "default_weight": 0.3,
        "description": "光影区域（通过边缘检测推断）"
    }
}
```

### Phase 2: 缓存流程改造

#### 2.1 新增分割掩码缓存

```
dataset/
├── image_001.png
├── image_001.txt
├── image_001_zi_latent.safetensors    # 现有：latent 缓存
├── image_001_zi_te.safetensors        # 现有：text encoder 缓存
└── image_001_zi_masks.safetensors     # 新增：语义分割掩码缓存
```

#### 2.2 掩码缓存格式

```python
# image_001_zi_masks.safetensors 内容
{
    "face_mask": torch.Tensor,      # [H, W] 浮点掩码 0~1
    "body_mask": torch.Tensor,
    "clothing_mask": torch.Tensor,
    "hair_mask": torch.Tensor,
    "background_mask": torch.Tensor,
    "metadata": {
        "sam_version": "sam3-base",
        "image_size": [1024, 1024],
        "latent_size": [128, 128],   # 下采样到 latent 尺寸
        "categories_detected": ["face", "body", "clothing"]
    }
}
```

### Phase 3: 训练逻辑改造

#### 3.1 新增参数

```toml
# config.toml 新增配置
[semantic_cosine]
enabled = true
face_weight = 1.0       # 人脸约束强度
body_weight = 0.5       # 身体约束强度
clothing_weight = 0.8   # 服装约束强度
hair_weight = 0.6       # 发型约束强度
background_weight = 0.0 # 背景约束强度（通常关闭）
global_fallback = 0.1   # 未分割区域的全局约束
```

#### 3.2 核心算法

```python
def semantic_cosine_loss(
    v_pred: torch.Tensor,      # [B, C, H, W] 预测速度
    v_target: torch.Tensor,    # [B, C, H, W] 目标速度
    masks: Dict[str, torch.Tensor],  # 语义掩码字典
    weights: Dict[str, float]  # 各区域权重
) -> torch.Tensor:
    """
    语义聚焦的 Cosine Loss
    
    对每个语义区域分别计算 cosine similarity，然后加权求和
    """
    total_loss = 0.0
    total_weight = 0.0
    
    for category, mask in masks.items():
        weight = weights.get(category, 0.0)
        if weight <= 0:
            continue
        
        # 将掩码扩展到 [B, C, H, W]
        mask_expanded = mask.unsqueeze(1).expand_as(v_pred)
        
        # 提取该区域的向量
        pred_region = (v_pred * mask_expanded).view(v_pred.shape[0], -1)
        target_region = (v_target * mask_expanded).view(v_target.shape[0], -1)
        
        # 计算该区域的 cosine similarity
        cos_sim = F.cosine_similarity(pred_region, target_region, dim=1)
        region_loss = (1 - cos_sim).mean()
        
        total_loss += weight * region_loss
        total_weight += weight
    
    # 归一化
    if total_weight > 0:
        total_loss = total_loss / total_weight
    
    return total_loss
```

### Phase 4: 前端界面

#### 4.1 配置界面

```
┌─────────────────────────────────────────────────┐
│ 语义聚焦 Cosine Loss                    [开启] │
├─────────────────────────────────────────────────┤
│                                                 │
│  人脸 Face        [████████████░░░░] 1.0       │
│  身体 Body        [██████░░░░░░░░░░] 0.5       │
│  服装 Clothing    [████████████░░░░] 0.8       │
│  发型 Hair        [████████░░░░░░░░] 0.6       │
│  背景 Background  [░░░░░░░░░░░░░░░░] 0.0       │
│                                                 │
│  [预览分割效果]  [重置为默认值]                 │
│                                                 │
└─────────────────────────────────────────────────┘
```

#### 4.2 分割预览

在数据集页面添加分割预览功能：
- 选择一张图片
- 显示 SAM3 分割结果
- 可视化各区域的掩码覆盖

---

## 📅 开发排期

| 阶段 | 任务 | 预计工时 | 优先级 |
|------|------|---------|--------|
| **Phase 1** | SAM3 模型集成 | 2-3 天 | P0 |
| 1.1 | 下载/加载 SAM3 模型 | 0.5 天 | |
| 1.2 | 实现分割推理接口 | 1 天 | |
| 1.3 | 掩码后处理（平滑、下采样） | 0.5 天 | |
| **Phase 2** | 缓存流程改造 | 2 天 | P0 |
| 2.1 | 新增 cache_masks.py 脚本 | 1 天 | |
| 2.2 | 集成到现有缓存流程 | 0.5 天 | |
| 2.3 | 前端缓存生成按钮更新 | 0.5 天 | |
| **Phase 3** | 训练逻辑改造 | 2 天 | P0 |
| 3.1 | 实现 semantic_cosine_loss | 1 天 | |
| 3.2 | 集成到 acrf_trainer.py | 0.5 天 | |
| 3.3 | 配置文件支持 | 0.5 天 | |
| **Phase 4** | 前端界面 | 1.5 天 | P1 |
| 4.1 | 配置界面（滑块组件） | 0.5 天 | |
| 4.2 | 分割预览功能 | 1 天 | |
| **Phase 5** | 测试与优化 | 2 天 | P1 |
| 5.1 | 功能测试 | 1 天 | |
| 5.2 | 性能优化 | 1 天 | |

**总计**：约 9-10 天

---

## 🔬 技术细节

### SAM3 依赖

```bash
pip install segment-anything-3  # 假设的包名，实际以官方为准
# 或
pip install sam3
```

### 显存估算

| 组件 | 显存占用 | 说明 |
|------|---------|------|
| SAM3-Base | ~1.5 GB | 推理时 |
| 掩码缓存 | ~10 MB/图 | 存储时 |
| 训练额外开销 | ~200 MB | 掩码加载 |

**建议**：在缓存阶段单独运行 SAM3，避免与训练冲突。

### 掩码下采样

图像分割在原图尺寸（如 1024x1024），但 latent 空间是 128x128。需要正确下采样：

```python
def downsample_mask_to_latent(mask: torch.Tensor, latent_size: tuple) -> torch.Tensor:
    """
    将原图尺寸的掩码下采样到 latent 尺寸
    使用双线性插值保持平滑
    """
    return F.interpolate(
        mask.unsqueeze(0).unsqueeze(0).float(),
        size=latent_size,
        mode='bilinear',
        align_corners=False
    ).squeeze()
```

---

## 🎯 预期效果

### 训练人物 LoRA

```toml
[semantic_cosine]
enabled = true
face_weight = 1.0       # 人脸精确
body_weight = 0.3       # 身材轻度约束
clothing_weight = 0.0   # 服装不约束（让模型学习风格而非具体衣服）
background_weight = 0.0 # 背景完全自由
```

**效果**：人脸精确还原，但可以穿不同衣服、不同背景。

### 训练服装 LoRA

```toml
[semantic_cosine]
enabled = true
face_weight = 0.0       # 人脸不约束（可以是任何人）
clothing_weight = 1.0   # 服装精确
body_weight = 0.5       # 身材适度约束（服装合身）
background_weight = 0.0
```

**效果**：服装精确还原，可以穿在不同人身上。

### 训练风格 LoRA

```toml
[semantic_cosine]
enabled = false  # 风格训练通常不需要语义分割
# 或者只约束光影
lighting_weight = 0.5
```

---

## 📝 备注

1. **SAM3 版本**：目前 SAM3 可能还未正式发布，需关注 Meta AI 官方动态。可先用 SAM2 或 SAM 实现原型。

2. **光影分割**：SAM 不直接支持光影分割，可考虑：
   - 使用边缘检测（Sobel/Canny）推断高光/阴影区域
   - 或训练专门的光影分割模型

3. **多人场景**：当图像中有多人时，需要实例分割而非语义分割。SAM3 支持实例分割，但需要额外逻辑处理。

4. **渐进式开发**：建议先实现 Face/Body/Background 三个基础类别，验证效果后再扩展。

---

## 🔗 参考资料

- [Segment Anything Model (SAM)](https://segment-anything.com/)
- [SAM 2 GitHub](https://github.com/facebookresearch/segment-anything-2)
- [Semantic Loss Functions in Deep Learning](https://arxiv.org/abs/xxxx)

---

*文档版本: v1.0*  
*创建日期: 2025-12-06*  
*作者: None Trainer Team*

