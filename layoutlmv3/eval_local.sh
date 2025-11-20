#!/usr/bin/env bash
#
# Evaluate local RF-DETR checkpoint
#

set -euo pipefail

# Configuration
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/workspace/outputs/rfdetr-commonforms/checkpoint_best_ema.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/rfdetr-commonforms}"
EVAL_OUTPUT="${EVAL_OUTPUT:-${OUTPUT_DIR}/eval}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
SPLIT="${SPLIT:-test}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"

echo "=========================================="
echo "RF-DETR Local Evaluation"
echo "=========================================="
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Model size: ${MODEL_SIZE}"
echo "Split: ${SPLIT}"
echo "Output: ${EVAL_OUTPUT}"
echo "=========================================="
echo ""

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec python3 "${DIR}/evaluate_rfdetr.py" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --dataset_name jbarrow/CommonForms \
  --test_split "${SPLIT}" \
  --model_size "${MODEL_SIZE}" \
  --cache_dir "${CACHE_DIR}" \
  --output_dir "${EVAL_OUTPUT}"
