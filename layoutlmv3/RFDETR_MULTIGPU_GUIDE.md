# RF-DETR Multi-GPU Training Guide

## Quick Commands

### Single-GPU Training (Default)
```bash
./train_rfdetr_full.sh
```

### 4-GPU Training
```bash
NUM_GPUS=4 ./train_rfdetr_full.sh
```

### 4-GPU Training with Resume
```bash
NUM_GPUS=4 RESUME_CHECKPOINT="auto" ./train_rfdetr_full.sh
```

### 8-GPU Training
```bash
NUM_GPUS=8 ./train_rfdetr_full.sh
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NUM_GPUS` | 1 | Number of GPUs to use |
| `RESUME_CHECKPOINT` | "" | Checkpoint path or "auto" to find latest EMA |
| `BATCH_SIZE` | 24 | Batch size **per GPU** |
| `GRAD_ACCUM_STEPS` | 1 | Gradient accumulation steps |
| `MODEL_SIZE` | medium | Model size: small, medium, or large |
| `EPOCHS` | 30 | Number of training epochs |
| `LEARNING_RATE` | 1e-4 | Learning rate |

## Effective Batch Size Calculation

**Effective Batch Size = BATCH_SIZE × GRAD_ACCUM_STEPS × NUM_GPUS**

### Examples:

#### Single GPU (H200)
```bash
BATCH_SIZE=24 GRAD_ACCUM_STEPS=1 NUM_GPUS=1
# Effective batch size = 24 × 1 × 1 = 24
```

#### 4 GPUs (H100)
```bash
NUM_GPUS=4 BATCH_SIZE=16 GRAD_ACCUM_STEPS=1
# Effective batch size = 16 × 1 × 4 = 64
```

#### 4 GPUs with Gradient Accumulation
```bash
NUM_GPUS=4 BATCH_SIZE=8 GRAD_ACCUM_STEPS=2
# Effective batch size = 8 × 2 × 4 = 64
```

## Recommended Configurations

### 4× H100 (80GB each)
```bash
NUM_GPUS=4 \
BATCH_SIZE=16 \
GRAD_ACCUM_STEPS=1 \
RESUME_CHECKPOINT="auto" \
./train_rfdetr_full.sh

# Effective batch size: 64
```

### 4× A100 (40GB each)
```bash
NUM_GPUS=4 \
BATCH_SIZE=8 \
GRAD_ACCUM_STEPS=2 \
RESUME_CHECKPOINT="auto" \
./train_rfdetr_full.sh

# Effective batch size: 64
```

### 4× L40S (48GB each)
```bash
NUM_GPUS=4 \
BATCH_SIZE=12 \
GRAD_ACCUM_STEPS=1 \
RESUME_CHECKPOINT="auto" \
./train_rfdetr_full.sh

# Effective batch size: 48
```

### 8× H200 (141GB each)
```bash
NUM_GPUS=8 \
BATCH_SIZE=24 \
GRAD_ACCUM_STEPS=1 \
RESUME_CHECKPOINT="auto" \
./train_rfdetr_full.sh

# Effective batch size: 192
```

## Full Example with All Options

```bash
NUM_GPUS=4 \
BATCH_SIZE=16 \
GRAD_ACCUM_STEPS=1 \
MODEL_SIZE=medium \
EPOCHS=30 \
LEARNING_RATE=1e-4 \
NUM_WORKERS=12 \
RESUME_CHECKPOINT="auto" \
OUTPUT_DIR=/workspace/outputs/rfdetr-commonforms \
CACHE_DIR=/workspace/cache \
HF_TOKEN=hf_xxxxx \
HF_USERNAME=your-username \
./train_rfdetr_full.sh
```

## How It Works

When `NUM_GPUS > 1`, the script automatically uses PyTorch's Distributed Data Parallel (DDP):

```bash
python -m torch.distributed.launch \
  --nproc_per_node=4 \
  --use_env \
  run_rfdetr_commonforms.py [args...]
```

This creates one training process per GPU and automatically:
- Splits the dataset across GPUs
- Synchronizes gradients
- Scales the effective batch size
- Handles checkpoint saving/loading

## Notes

- **Batch size is per GPU**: Set `BATCH_SIZE` based on single GPU memory
- **Effective batch size multiplies**: Total batch = `BATCH_SIZE × GRAD_ACCUM × NUM_GPUS`
- **Resume works**: Checkpoints are compatible between single/multi-GPU
- **TensorBoard**: Available on port 6006 during training
- **Early stopping**: Enabled by default

## Monitoring

While training, monitor:
- TensorBoard: `http://localhost:6006` (or RunPod HTTP service)
- GPU usage: `nvidia-smi -l 1`
- Training logs: Printed to console

## Troubleshooting

### OOM (Out of Memory)
Reduce `BATCH_SIZE` or increase `GRAD_ACCUM_STEPS`:
```bash
NUM_GPUS=4 BATCH_SIZE=8 GRAD_ACCUM_STEPS=2 ./train_rfdetr_full.sh
```

### NCCL Errors
Check that all GPUs are visible:
```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

### Slow Data Loading
Increase `NUM_WORKERS`:
```bash
NUM_GPUS=4 NUM_WORKERS=16 ./train_rfdetr_full.sh
```
