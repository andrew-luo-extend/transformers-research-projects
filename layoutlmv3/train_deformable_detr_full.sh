#!/usr/bin/env bash
# Full RunPod training loop for Deformable DETR on CommonForms with Hugging Face push.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username.}"

MODEL_ID="${HF_MODEL_ID:-${HF_USERNAME}/deformable-detr-commonforms}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/deformable-detr-commonforms}"
CACHE_DIR="${CACHE_DIR:-/workspace/cache}"
DATASETS_CACHE="$CACHE_DIR/datasets"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  HUB_TOKEN_ARGS=(--hub_token "${HF_TOKEN}")
else
  HUB_TOKEN_ARGS=()
fi

export TOKENIZERS_PARALLELISM=false

# Kill any existing TensorBoard on port 6006
echo "Checking for existing TensorBoard process on port 6006..."
lsof -ti:6006 | xargs kill -9 2>/dev/null && echo "Killed existing TensorBoard" || echo "No existing TensorBoard found"

# Start TensorBoard in background
TENSORBOARD_LOG_DIR="${OUTPUT_DIR}/runs"
if command -v tensorboard &> /dev/null; then
  echo "Starting TensorBoard on port 6006..."
  tensorboard --logdir "${TENSORBOARD_LOG_DIR}" --host 0.0.0.0 --port 6006 &
  TENSORBOARD_PID=$!
  echo "TensorBoard started with PID ${TENSORBOARD_PID}"
  echo "Access via RunPod HTTP Services on port 6006"
fi

exec "${PYTHON_BIN}" "${DIR}/run_deformable_detr_commonforms_v2.py" \
  --model_name_or_path Aryn/deformable-detr-DocLayNet \
  --dataset_name jbarrow/CommonForms \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-30}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-8}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-8}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --logging_steps "${LOGGING_STEPS:-50}" \
  --save_strategy "${SAVE_STRATEGY:-epoch}" \
  --eval_strategy "${EVAL_STRATEGY:-epoch}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --load_best_model_at_end \
  --metric_for_best_model "${METRIC_FOR_BEST_MODEL:-eval_loss}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS:-8}" \
  --report_to "${REPORT_TO:-tensorboard}" \
  --seed "${SEED:-42}" \
  --do_train \
  --do_eval \
  --push_to_hub \
  --hub_model_id "${MODEL_ID}" \
  --hub_strategy "${HUB_STRATEGY:-every_save}" \
  "${HUB_TOKEN_ARGS[@]}" \
  --use_runpod_defaults
