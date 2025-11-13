# Troubleshooting Changes Applied

## Summary
Applied 5 concrete troubleshooting steps to diagnose the NaN gradient issue in Deformable DETR training.

## Changes Made

### 1. ✅ Enhanced Logging of First Batch
**Location**: `run_deformable_detr_commonforms_v2.py` lines ~1358-1393

**What was added**:
- Detailed logging of actual box values from first sample
- Print first 5 boxes with their coordinates
- Box statistics: min, max, mean values
- Warnings if boxes are not normalized (> 1.0 or < 0.0)
- Print first 5 class labels and their range

**Purpose**: Verify that input data is properly normalized and in valid range [0,1] before training starts.

**Expected output**:
```
============================================================
DETAILED INSPECTION: First sample in batch
============================================================
Example boxes (first 5):
  Box 0: [0.123, 0.456, 0.789, 0.234]
  Box 1: [0.345, 0.678, 0.901, 0.123]
  ...
Box statistics:
  Min value: 0.000012
  Max value: 0.987654
  Mean value: 0.456789
Example class_labels (first 5):
  [1, 2, 1, 3, 2]
Class label range: [1, 10]
============================================================
```

---

### 2. ✅ Disabled Custom Matcher Wrapper
**Location**: `run_deformable_detr_commonforms_v2.py` lines ~1250-1260

**What was changed**:
```python
# BEFORE: Matcher wrapper was enabled
criterion.matcher = create_safe_matcher_wrapper(criterion.matcher)

# AFTER: Temporarily commented out
# if hasattr(model, 'model') and hasattr(model.model, 'criterion'):
#     criterion = model.model.criterion
#     ...
logger.info("⚠️  MATCHER WRAPPER DISABLED FOR TROUBLESHOOTING")
```

**Purpose**: Test if NaNs are coming from the custom wrapper vs original matcher. If NaNs disappear, the wrapper has a bug. If NaNs persist, the problem is elsewhere.

**To re-enable**: Uncomment the matcher wrapper code block.

---

### 3. ✅ Simplified Augmentations
**Location**: `run_deformable_detr_commonforms_v2.py` lines ~540-559

**What was changed**:
```python
# BEFORE: Multiple augmentations
A.LongestMaxSize(image_size),
A.PadIfNeeded(image_size, image_size, border_mode=0, value=(0, 0, 0)),
A.HorizontalFlip(p=0.5),
A.RandomBrightnessContrast(p=0.5),
A.HueSaturationValue(p=0.1),
A.Rotate(limit=10, p=0.5),
A.RandomScale(scale_limit=0.2, p=0.5),

# AFTER: Only basic transforms
A.LongestMaxSize(image_size),
A.PadIfNeeded(image_size, image_size, border_mode=0, value=(0, 0, 0)),
# All other augmentations commented out
```

**Purpose**: Test if Rotate/RandomScale/etc. occasionally produce extreme coords that cause GIoU to NaN.

**Strategy**:
- If NaNs vanish → Add augmentations back one by one to identify the culprit
- If NaNs persist → Problem is not in augmentations

**To restore**: Uncomment the augmentation lines.

---

### 4. ✅ Reduced Learning Rate
**Location**: `train_deformable_detr_test.sh` line 37

**What was changed**:
```bash
# BEFORE
--learning_rate "${LEARNING_RATE:-5e-5}"

# AFTER
--learning_rate "${LEARNING_RATE:-1e-5}"
```

**Purpose**: Lower LR (1e-5 instead of 5e-5) reduces gradient magnitudes, making training more stable. If NaNs disappear, the original LR was too high.

---

### 5. ✅ Smaller Test Dataset
**Location**: `train_deformable_detr_test.sh` lines 32-34

**What was changed**:
```bash
# BEFORE
--max_train_samples "${MAX_TRAIN_SAMPLES:-128}"
--max_eval_samples "${MAX_EVAL_SAMPLES:-64}"
--num_train_epochs "${NUM_TRAIN_EPOCHS:-2}"

# AFTER
--max_train_samples "${MAX_TRAIN_SAMPLES:-64}"
--max_eval_samples "${MAX_EVAL_SAMPLES:-32}"
--num_train_epochs "${NUM_TRAIN_EPOCHS:-1}"
```

**Purpose**:
- Faster iteration for troubleshooting (64 samples, 1 epoch)
- Easier to identify which specific sample causes NaN
- Reduces time to reproduce the issue

---

### 6. ✅ Removed fp16
**Location**: `train_deformable_detr_test.sh` line 45

**What was changed**:
```bash
# BEFORE
--fp16 \

# AFTER
(removed completely - train in fp32)
```

**Purpose**: The original issue report stated that fp16 causes GIoU overflow. This was already documented but the test script still had `--fp16` flag.

---

## How to Test

### Run the troubleshooting test:
```bash
cd /workspace/transformers-research-projects/layoutlmv3
./train_deformable_detr_test.sh
```

### What to look for in the output:

1. **Detailed box inspection** - Check if boxes are normalized [0,1]:
   ```
   DETAILED INSPECTION: First sample in batch
   Box 0: [0.123, 0.456, 0.789, 0.234]  ← Should be in [0,1]
   Box statistics:
     Min value: 0.000012  ← Should be >= 0
     Max value: 0.987654  ← Should be <= 1
   ```

2. **Matcher wrapper status** - Should see:
   ```
   ⚠️  MATCHER WRAPPER DISABLED FOR TROUBLESHOOTING
   ```

3. **Training progress** - Watch for:
   ```
   {'loss': 5.234, 'grad_norm': 2.345, ...}  ← grad_norm should be finite
   ```

4. **NaN detection** - If you see:
   ```
   {'loss': nan, 'grad_norm': nan, ...}  ← Problem still exists
   ```

---

## Diagnosis Decision Tree

### If NaNs DISAPPEAR:

1. **Matcher wrapper was the issue**
   - Re-enable matcher wrapper
   - Debug the wrapper logic
   - Check for edge cases in empty sample handling

2. **Augmentations were the issue**
   - Add back augmentations one by one
   - Test: HorizontalFlip → RandomBrightnessContrast → Rotate → RandomScale
   - Identify which augmentation causes extreme coords

3. **Learning rate was too high**
   - Try intermediate LRs: 2e-5, 3e-5, 4e-5
   - Find the maximum stable LR

### If NaNs PERSIST:

**Then the problem is likely in**:
- Model architecture bug (unlikely - pretrained model)
- Deformable attention producing NaN
- Loss calculation (GIoU, L1, class loss)
- Gradient clipping not working

**Next steps**:
1. Add loss component logging (separate giou_loss, l1_loss, class_loss)
2. Check intermediate model outputs (encoder, decoder)
3. Verify optimizer state is not corrupted
4. Test with different batch sizes (1 vs 2 vs 4)

---

## Reverting Changes

### To restore original behavior:

1. **Re-enable matcher wrapper**:
   - Uncomment lines 1252-1259 in `run_deformable_detr_commonforms_v2.py`
   - Remove the warning log line 1260

2. **Restore augmentations**:
   - Uncomment lines 547-551 in `run_deformable_detr_commonforms_v2.py`

3. **Restore original test parameters**:
   ```bash
   MAX_TRAIN_SAMPLES=128 \
   MAX_EVAL_SAMPLES=64 \
   NUM_TRAIN_EPOCHS=2 \
   LEARNING_RATE=5e-5 \
   ./train_deformable_detr_test.sh
   ```

---

## Notes

- The scipy monkey-patch for `linear_sum_assignment` is STILL ACTIVE (lines 65-119)
- Dataset filtering is STILL ACTIVE (requires valid bboxes)
- Sanity check is STILL ACTIVE (asserts all samples have boxes)
- These remain as defensive layers even during troubleshooting

---

## Timeline

All changes applied: 2025-11-12

Ready for testing.
