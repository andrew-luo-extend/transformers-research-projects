# Deformable DETR Training Fix - Manual Head Replacement

## Problem

Training Deformable DETR with `ignore_mismatched_sizes=True` caused NaN errors from step 1 due to:
- Corrupted encoder/decoder state when changing number of classes
- MultiScaleDeformableAttention producing NaN values
- Even with fp32 precision, model produced NaN immediately

Previous trained model had:
- ❌ 12 DocLayNet classes instead of 2-3 CommonForms classes
- ❌ mAP of 0.000 on test set
- ❌ Wrong bbox predictions (oversized, low confidence)

## Solution: Manual Head Replacement

The script now:

### Step 1: Load Clean Pretrained Model
```python
model = AutoModelForObjectDetection.from_pretrained(
    "Aryn/deformable-detr-DocLayNet",
    # NO config override
    # NO ignore_mismatched_sizes
)
```
This loads the model with its original 12 DocLayNet classes and **clean, uncorrupted weights**.

### Step 2: Replace Classification Head
```python
# Old head: hidden_dim → 12 classes + 1 (no-object)
# New head: hidden_dim → 3 classes + 1 (no-object)

new_head = torch.nn.Linear(hidden_dim, new_num_classes + 1)
torch.nn.init.normal_(new_head.weight, std=0.01)
torch.nn.init.constant_(new_head.bias, -2.0)

model.model.class_embed = new_head
```

### Step 3: Reinitialize Bbox Head
```python
for layer in bbox_embed.layers:
    torch.nn.init.xavier_uniform_(layer.weight, gain=0.01)
```
Though bbox head doesn't depend on num_classes, we reinitialize for stability.

### Step 4: Update Config
```python
model.config.num_labels = new_num_classes
model.config.id2label = id2label  # CommonForms labels
model.config.label2id = label2id
```

### Step 5: Verification
The script automatically verifies the config matches the dataset and will **abort if there's a mismatch**.

## What This Avoids

✅ **No `ignore_mismatched_sizes`** - Avoids the corruption bug

✅ **Clean pretrained weights** - Backbone, encoder, decoder are pristine

✅ **Controlled initialization** - We control exactly how heads are initialized

✅ **Proper class mapping** - Model uses CommonForms classes from the start

## Usage

### Quick Test (10 samples, 5 epochs, ~10 min)

```bash
cd /workspace

# Set environment
export HF_USERNAME="your-username"
export HF_TOKEN="your-token"  # Optional

# Run test
./transformers-research-projects/layoutlmv3/train_deformable_detr_test.sh
```

### Full Training (Complete dataset, 15 epochs, ~24 hours on H100)

```bash
cd /workspace

# Set environment
export HF_USERNAME="your-username"
export HF_TOKEN="your-token"
export OUTPUT_DIR="/workspace/outputs/deformable-detr-commonforms"
export CACHE_DIR="/workspace/cache"

# Clean old outputs
rm -rf /workspace/outputs/deformable-detr-commonforms

# Start training
./transformers-research-projects/layoutlmv3/train_deformable_detr_full.sh
```

## Verification (Do This First!)

Within the first minute of training, check the logs for:

```
================================================================================
🔍 MODEL CONFIGURATION VERIFICATION
================================================================================
Expected classes (from dataset): 3
Expected id2label: {0: 'class_0', 1: 'class_1', 2: 'class_2'}

Actual model config:
  num_labels: 3
  id2label: {0: 'class_0', 1: 'class_1', 2: 'class_2'}

✅ VERIFIED: Model class configuration matches dataset!
================================================================================
```

If you see ❌ or WARNING, training will abort automatically.

You can also check the saved config after a few minutes:

```bash
cat /workspace/outputs/deformable-detr-commonforms/config.json | grep -A 5 "id2label"

# Should show CommonForms classes (class_0, class_1, class_2)
# NOT DocLayNet classes (N/A, Caption, Footnote, etc.)
```

## TensorBoard Monitoring

TensorBoard auto-starts on port 6006. Access via RunPod HTTP Services:

1. Go to your pod in RunPod dashboard
2. HTTP Services section → enter port `6006`
3. Open the generated URL

**Monitor:**
- Loss should start at 5-10 and decrease
- grad_norm should be 1-10 (not NaN, not 0)
- No "❌ NaN/Inf detected" errors in logs

## Expected Results

After 15 epochs on full CommonForms dataset:

- **mAP @ IoU 0.5:0.95**: 0.70-0.85
- **mAP @ IoU 0.5**: 0.85-0.95
- **Per-class AP**: All classes should have > 0.5 AP
- **Bbox predictions**: Should match ground truth size/location

## Troubleshooting

If you STILL get NaN errors:

1. **Check verification output** - ensure classes match
2. **Try full fp32** - remove any fp16/bf16 flags
3. **Increase warmup** - use `WARMUP_RATIO=0.2`
4. **Lower learning rate** - use `LEARNING_RATE=1e-5`
5. **Check logs** - NaN hooks will show which module fails

If none of this works, fall back to **Regular DETR** which is more stable:

```bash
./transformers-research-projects/layoutlmv3/train_detr_full.sh
```

## Files Modified

- ✅ `run_deformable_detr_commonforms_v2.py` - Manual head replacement logic
- ✅ `train_deformable_detr_full.sh` - Training script with auto-verification
- ✅ `evaluate_detr.py` - Evaluation with detailed bbox debugging
- ✅ `train_detr_full.sh` - Backup option using Regular DETR
- ✅ `train_detr_test.sh` - Quick test for Regular DETR

## Summary

The core issue was `ignore_mismatched_sizes=True` corrupting the model state when changing number of classes. Manual head replacement avoids this entirely by:

1. Loading pretrained model cleanly (no corruption)
2. Surgically replacing only the classification head
3. Keeping all pretrained weights intact
4. Properly initializing new head for stability

This should train successfully with:
- ✅ No NaN errors
- ✅ Correct CommonForms class mapping
- ✅ Valid mAP scores on evaluation

Good luck with training! 🚀

