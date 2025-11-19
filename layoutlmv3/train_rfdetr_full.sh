#!/usr/bin/env bash
#
# Train RF-DETR on CommonForms (Full Dataset)
#
# RF-DETR is faster than regular DETR while maintaining accuracy.
# Recommended batch_size × grad_accum_steps = 16 for optimal training
#
# GPU Recommendations:
#   H200:      batch_size=24-32, grad_accum_steps=1 (141GB VRAM)
#   H100/A100: batch_size=16, grad_accum_steps=1 (80GB VRAM)
#   L40S:      batch_size=8,  grad_accum_steps=2
#   T4:        batch_size=4,  grad_accum_steps=4
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

# Configuration
: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username.}"

MODEL_ID="${HF_MODEL_ID:-${HF_USERNAME}/rfdetr-commonforms}"  # Production model (no -test suffix)
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/rfdetr-commonforms}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"

# Handle HuggingFace token
if [[ -n "${HF_TOKEN:-}" ]]; then
  HUB_ARGS="--push_to_hub --hub_model_id ${MODEL_ID}"
  echo "Will push to Hub: ${MODEL_ID}"
else
  HUB_ARGS=""
  echo "⚠️  HF_TOKEN not set, skipping Hub upload"
fi

# Create directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

# RF-DETR Training Parameters
MODEL_SIZE="${MODEL_SIZE:-medium}"  # small, medium, or large
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-24}"  # H200: Start with 24, can increase to 32 if no OOM
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"  # batch_size × grad_accum = effective batch size
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-12}"

# Checkpoint resumption
# Set to "auto" to automatically find the latest EMA checkpoint
# Set to a specific path like "/workspace/outputs/rfdetr-commonforms/checkpoint_ema.pt"
# Leave empty to start from scratch
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
 

echo "=========================================="
echo "RF-DETR Training on CommonForms"
echo "=========================================="
echo "Model size: ${MODEL_SIZE}"
echo "Epochs: ${EPOCHS}"
echo "Batch size: ${BATCH_SIZE}"
echo "Gradient accumulation: ${GRAD_ACCUM_STEPS}"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS))"
echo "Learning rate: ${LEARNING_RATE}"
echo "Output: ${OUTPUT_DIR}"
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  echo "Resume checkpoint: ${RESUME_CHECKPOINT}"
else
  echo "Resume checkpoint: None (training from scratch)"
fi
echo "=========================================="

exec "${PYTHON_BIN}" "${DIR}/run_rfdetr_commonforms.py" \
  --dataset_name jbarrow/CommonForms \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_size "${MODEL_SIZE}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_workers "${NUM_WORKERS}" \
  ${RESUME_CHECKPOINT:+--resume_from_checkpoint "${RESUME_CHECKPOINT}"} \
  ${HUB_ARGS}
