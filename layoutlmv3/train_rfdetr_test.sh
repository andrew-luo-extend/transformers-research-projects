#!/usr/bin/env bash
#
# Quick Test - RF-DETR on CommonForms
#
# Tests RF-DETR training with 50 samples for 5 epochs (~10 minutes)
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username.}"

MODEL_ID="${HF_USERNAME}/rfdetr-commonforms-test"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/rfdetr-test}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"

# Handle HuggingFace token
if [[ -n "${HF_TOKEN:-}" ]]; then
  HUB_ARGS="--push_to_hub --hub_model_id ${MODEL_ID}"
  echo "Will push to Hub: ${MODEL_ID}"
else
  HUB_ARGS=""
  echo "HF_TOKEN not set, skipping Hub upload"
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

echo "=========================================="
echo "RF-DETR QUICK TEST"
echo "=========================================="
echo "50 train samples, 10 val samples"
echo "1 epoch"
echo "Should complete in ~2-3 minutes"
echo "Hub model: ${MODEL_ID}"
echo "=========================================="

exec "${PYTHON_BIN}" "${DIR}/run_rfdetr_commonforms.py" \
  --dataset_name jbarrow/CommonForms \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_size small \
  --max_train_samples 50 \
  --max_val_samples 10 \
  --epochs 1 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --learning_rate 1e-4 \
  --num_workers 0 \
  ${HUB_ARGS}

