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

## Quick Re-Merge Checklist

After pulling upstream changes:

1. [ ] Check if `train_acrf.py` has conflicts → re-apply the ~8 lines above
2. [ ] Check if `websocket.py` / `system.py` `get_gpu_info()` changed → re-apply multi-GPU logic
3. [ ] Check if `Monitor.vue` changed → re-add MultiGpuMonitor import + avgStepTime fix
4. [ ] Check if `TrainingConfig.vue` changed → re-add resume/multi-GPU UI section
5. [ ] Check if `websocket.ts` changed → re-add sessionStartStep tracking
6. [ ] Check if `training.ts` changed → re-add sessionStartStep/stepsThisSession fields
7. [ ] New files in `extensions/`, `components/`, `scripts/train_multi_gpu.py` → no action needed

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

