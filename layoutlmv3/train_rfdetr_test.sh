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

OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/rfdetr-test}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

echo "=========================================="
echo "RF-DETR QUICK TEST"
echo "=========================================="
echo "50 train samples, 10 val samples"
echo "5 epochs"
echo "Should complete in ~10 minutes"
echo "=========================================="

exec "${PYTHON_BIN}" "${DIR}/run_rfdetr_commonforms.py" \
  --dataset_name jbarrow/CommonForms \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_size small \
  --max_train_samples 50 \
  --max_val_samples 10 \
  --epochs 5 \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --learning_rate 1e-4 \
  --num_workers 0

