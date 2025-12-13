# Custom Changes Documentation

This document tracks all custom modifications made to this fork for easy re-merging with upstream.

**Last Updated:** 2024-12-09

---

## Summary

| Area | Files Modified | Merge Difficulty |
|------|----------------|------------------|
| Training Extensions | 1 import + 8 lines | Easy |
| Multi-GPU Backend | 2 functions | Easy (additive) |
| Multi-GPU Frontend | 1 new file + 2 lines | Easy |
| Per-Dataset Loss | 5 files | Medium (additive logic) |

---

## 1. Training Extensions (Resume, Multi-GPU, Logging)

### New Files (No merge conflicts)
```
src/zimage_trainer/extensions/
├── __init__.py
└── training_extensions.py
```

### Modified: `scripts/train_acrf.py`

**Changes to re-apply after upstream merge:**

1. **Add import** (after other zimage_trainer imports):
```python
# === MODULAR EXTENSION === (easy to merge upstream - only this import needed)
from zimage_trainer.extensions import TrainingExtensions
```

2. **Add distributed setup** (after `os.makedirs(args.output_dir, exist_ok=True)`):
```python
# === MODULAR EXTENSION: distributed setup ===
dist_backend = TrainingExtensions.setup_distributed(args.distributed_backend)
logger.info(f"[DIST] 分布式后端: {dist_backend}")
```

3. **Add extension instance** (after `accelerator.prepare()`):
```python
# 7.5. Training Extensions (modular - handles resume, multi-GPU logging, etc.)
ext = TrainingExtensions(args.output_dir, accelerator, args.output_name)

# Resume training (all logic encapsulated in extension)
start_epoch, global_step = ext.load_resume_state(
    optimizer, lr_scheduler, network, resume_enabled=args.resume
)

if ext.is_training_complete(start_epoch, args.num_train_epochs):
    return

# 8. 训练循环
ext.log_training_start(start_epoch, global_step, args.num_train_epochs)
```

4. **Replace epoch logging**:
```python
# Old:
# if accelerator.is_main_process:
#     logger.info(f"\nEpoch {epoch + 1}/{args.num_train_epochs}")

# New:
ext.log_epoch(epoch, args.num_train_epochs)
```

5. **Replace step printing** (use `ext.print_step()` instead of `print()` with `accelerator.is_main_process` check)

6. **Replace checkpoint saving**:
```python
# Old: Manual unwrap + save + wait_for_everyone
# New:
ext.save_epoch_checkpoint(network, optimizer, lr_scheduler, epoch, global_step, weight_dtype)
```

7. **Replace final save**:
```python
final_path = ext.save_final_checkpoint(network, weight_dtype)
memory_optimizer.stop()
ext.log_training_complete(final_path)
```

---

## 2. Multi-GPU Backend

### Modified: `webui-vue/api/routers/websocket.py`

**Function:** `get_gpu_info()`

**Change:** Returns all GPUs + aggregated summary instead of just first GPU.

**New fields added (backwards compatible):**
```python
{
    # Existing fields unchanged
    "name": "2x NVIDIA RTX...",  # Now shows count if multi-GPU
    "memoryTotal": 191.2,         # Now sum of all GPUs
    "memoryUsed": 45.3,           # Now sum of all GPUs
    "memoryPercent": 24,          # Weighted average
    "utilization": 85,            # Average across GPUs
    "temperature": 72,            # Max temperature
    
    # NEW fields (additive only)
    "numGpus": 2,
    "gpus": [
        {"index": 0, "name": "...", "memoryTotal": 95.6, ...},
        {"index": 1, "name": "...", "memoryTotal": 95.6, ...}
    ]
}
```

### Modified: `webui-vue/api/routers/system.py`

Same change as websocket.py (duplicate function for sync endpoint).

---

## 3. Multi-GPU Frontend

### New File (No merge conflicts)
```
webui-vue/src/components/MultiGpuMonitor.vue
```
Self-contained component that displays individual GPU stats when multiple GPUs detected.

### Modified: `webui-vue/src/views/Monitor.vue`

**Only 2 lines added:**

1. Import (in `<script setup>`):
```typescript
// === CUSTOM: Multi-GPU monitoring component ===
import MultiGpuMonitor from '@/components/MultiGpuMonitor.vue'
```

2. Component tag (before `<!-- GPU 监控 -->`):
```html
<!-- === CUSTOM: Multi-GPU display (shows only if multiple GPUs detected) === -->
<MultiGpuMonitor />
```

---

## 4. Training Config UI

### Modified: `webui-vue/src/views/TrainingConfig.vue`

Added UI controls for:
- Resume training toggle
- Multi-GPU toggle
- GPU count selector

**Search for:** `恢复与分布式` to find the added section.

---

## 5. Multi-GPU Launcher

### New File (No merge conflicts)
```
scripts/train_multi_gpu.py
```
Custom launcher for Windows multi-GPU using FileStore (bypasses accelerate launch issues).

---

## 6. Step Timing Fix (Resume Accuracy)

### Modified: `webui-vue/src/stores/websocket.ts`

Added `sessionStartStep` tracking to fix s/step calculation when resuming:

```typescript
// === CUSTOM: Track session start step for accurate s/step calculation ===
let sessionStartStep: number | null = null

// In progress update:
if (!trainingStartTime) {
  trainingStartTime = Date.now()
  sessionStartStep = progress.step.current  // Remember where we started
}
const stepsThisSession = progress.step.current - (sessionStartStep || 0)
```

### Modified: `webui-vue/src/stores/training.ts`

Added interface fields:
```typescript
// === CUSTOM: For accurate s/step calculation on resume ===
sessionStartStep?: number
stepsThisSession?: number
```

### Modified: `webui-vue/src/views/Monitor.vue`

Updated avgStepTime calculation:
```typescript
// === CUSTOM: Use stepsThisSession for accurate s/step on resume ===
const steps = progress.value.stepsThisSession ?? progress.value.currentStep
```

---

## 7. Process Cleanup for Multi-GPU

### New File: `kill_training.bat`
Cleanup script to kill orphaned training processes.

### Modified: `webui-vue/api/routers/training.py`
`stop_training()` now uses psutil to kill child processes (multi-GPU workers).

### Modified: `scripts/train_multi_gpu.py`
Added atexit and signal handlers to cleanup child processes on exit.

---

## 8. Per-Dataset Loss Settings

Allows each dataset to have its own loss weights (lambda_struct, lambda_light, lambda_color, lambda_tex) instead of a single global setting.

### Modified: `webui-vue/src/views/TrainingConfig.vue`

**Changes:**
1. Added `loss_scope` radio group (after loss_mode selector):
```vue
<!-- === CUSTOM: Loss Scope Switch (Global vs Per-Dataset) === -->
<el-radio-group v-model="config.training.loss_scope">
  <el-radio label="global">Global (全局)</el-radio>
  <el-radio label="per_dataset">Per Dataset (按数据集)</el-radio>
</el-radio-group>
```

2. Added `loss_scope: 'global'` to `getDefaultConfig()`:
```typescript
loss_scope: 'global',
```

3. Added per-dataset loss controls in dataset items (conditionally shown):
```vue
<template v-if="config.training.loss_scope === 'per_dataset' && ['style', 'unified'].includes(config.training.loss_mode)">
  <!-- lambda_struct, lambda_light, lambda_color, lambda_tex sliders -->
</template>
```

4. Added default loss weights to `addDatasetFromCache()` and `addDataset()`:
```typescript
lambda_struct: 1.0,
lambda_light: 0.0,
lambda_color: 0.0,
lambda_tex: 0.0
```

### Modified: `webui-vue/api/routers/training.py`

**Function:** `generate_acrf_toml_config()`

**Changes:**
1. Added `loss_scope` to TOML output
2. Added per-dataset loss weights when `loss_scope == 'per_dataset'`

### Modified: `src/zimage_trainer/dataset/dataloader.py`

**Changes:**
1. `ZImageLatentDataset.__init__`: Track `dataset_indices` for each sample
2. `ZImageLatentDataset.__getitem__`: Return `dataset_idx` in output dict
3. `collate_fn`: Batch `dataset_idx` as tensor

### Modified: `src/zimage_trainer/losses/style_structure_loss.py`

**Added:** `forward_per_sample()` method to `LatentStyleStructureLoss` class
- Computes loss per-sample with custom weights instead of batch-averaging
- Used when `loss_scope == 'per_dataset'`

### Modified: `scripts/train_acrf.py`

**Changes:**
1. Added `--loss_scope` argument
2. Added `args.datasets_loss_weights` parsing from config
3. Modified 'style' and 'unified' loss computation to use `forward_per_sample()` when per-dataset mode is enabled

---

## Quick Re-Merge Checklist

After pulling upstream changes:

1. [ ] Check if `train_acrf.py` has conflicts → re-apply the ~8 lines above + per-dataset loss logic
2. [ ] Check if `websocket.py` / `system.py` `get_gpu_info()` changed → re-apply multi-GPU logic
3. [ ] Check if `Monitor.vue` changed → re-add MultiGpuMonitor import + avgStepTime fix
4. [ ] Check if `TrainingConfig.vue` changed → re-add resume/multi-GPU UI + per-dataset loss UI
5. [ ] Check if `websocket.ts` changed → re-add sessionStartStep tracking
6. [ ] Check if `training.ts` changed → re-add sessionStartStep/stepsThisSession fields
7. [ ] Check if `dataloader.py` changed → re-add dataset_idx tracking
8. [ ] Check if `style_structure_loss.py` changed → re-add `forward_per_sample()` method
9. [ ] Check if `training.py` changed → re-add loss_scope to TOML generation
10. [ ] New files in `extensions/`, `components/`, `scripts/train_multi_gpu.py` → no action needed

---

## Testing After Merge

```bash
# 1. Test imports
.\python\python.exe -c "from zimage_trainer.extensions import TrainingExtensions; print('OK')"

# 2. Test syntax
.\python\python.exe -m py_compile .\scripts\train_acrf.py

# 3. Start app and verify UI
.\start.bat
```

