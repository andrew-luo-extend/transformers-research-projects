#!/usr/bin/env bash
#
# Evaluate RF-DETR model on CommonForms test split
#
# Usage:
#   ./eval_rfdetr.sh
#
# Environment Variables:
#   HF_USERNAME  - Your HuggingFace username (required)
#   HF_TOKEN     - Your HuggingFace token (optional, for private models)
#   CACHE_DIR    - Cache directory (default: /workspace/cache)
#

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

# Configuration
: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username.}"

MODEL_ID="${HF_MODEL_ID:-${HF_USERNAME}/rfdetr-commonforms}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
OUTPUT_DIR="${OUTPUT_DIR:-./eval_results_rfdetr}"

# Evaluation settings
DATASET_NAME="${DATASET_NAME:-jbarrow/CommonForms}"
TEST_SPLIT="${TEST_SPLIT:-test}"
MODEL_SIZE="${MODEL_SIZE:-medium}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # Empty = evaluate all
THRESHOLD="${THRESHOLD:-0.0}"  # 0.0 for mAP calculation

mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "RF-DETR Evaluation"
echo "=========================================="
echo "Model: ${MODEL_ID}"
echo "Dataset: ${DATASET_NAME}"
echo "Split: ${TEST_SPLIT}"
echo "Model size: ${MODEL_SIZE}"
if [[ -n "${MAX_SAMPLES}" ]]; then
  echo "Max samples: ${MAX_SAMPLES}"
else
  echo "Max samples: ALL (full evaluation)"
fi
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

# Build command
CMD="${PYTHON_BIN} ${DIR}/evaluate_rfdetr.py \
  --hub_model_id ${MODEL_ID} \
  --dataset_name ${DATASET_NAME} \
  --test_split ${TEST_SPLIT} \
  --cache_dir ${CACHE_DIR} \
  --output_dir ${OUTPUT_DIR} \
  --model_size ${MODEL_SIZE} \
  --threshold ${THRESHOLD}"

# Add max_samples if set
if [[ -n "${MAX_SAMPLES}" ]]; then
  CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

# Execute
exec ${CMD}

