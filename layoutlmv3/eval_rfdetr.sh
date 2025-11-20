#!/usr/bin/env bash
#
# Evaluate RF-DETR checkpoint on test set
#
# Usage:
#   ./eval_rfdetr.sh                           # Auto-detect checkpoint and dataset
#   CHECKPOINT=path/to/checkpoint.pth ./eval_rfdetr.sh
#   SPLIT=valid ./eval_rfdetr.sh              # Evaluate on validation set
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

# Configuration
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/rfdetr-commonforms}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
SPLIT="${SPLIT:-test}"  # test, valid, or train
CHECKPOINT="${CHECKPOINT:-}"  # Optional: specify checkpoint path

echo "=========================================="
echo "RF-DETR Evaluation"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Model size: ${MODEL_SIZE}"
echo "Split: ${SPLIT}"
if [[ -n "${CHECKPOINT}" ]]; then
  echo "Checkpoint: ${CHECKPOINT}"
else
  echo "Checkpoint: Auto-detect best EMA checkpoint"
fi
echo "=========================================="
echo ""

# Build evaluation command
EVAL_SCRIPT="${DIR}/eval_rfdetr.py"
EVAL_ARGS="--output_dir ${OUTPUT_DIR} \
  --model_size ${MODEL_SIZE} \
  --split ${SPLIT}"

if [[ -n "${CHECKPOINT}" ]]; then
  EVAL_ARGS="${EVAL_ARGS} --checkpoint_path ${CHECKPOINT}"
fi

# Run evaluation
echo "Running evaluation..."
echo ""
exec "${PYTHON_BIN}" "${EVAL_SCRIPT}" ${EVAL_ARGS}
