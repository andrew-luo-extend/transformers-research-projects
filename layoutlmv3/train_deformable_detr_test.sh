#!/usr/bin/env bash
# Quick sanity run for Deformable DETR on CommonForms that uploads the result to Hugging Face.

set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON:-python3}"

: "${HF_USERNAME:?Set HF_USERNAME to your Hugging Face username.}"

MODEL_ID="${HF_MODEL_ID:-${HF_USERNAME}/deformable-detr-commonforms-test}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs/deformable-detr-test}"
CACHE_DIR="${CACHE_DIR:-/workspace/hf-cache}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  HUB_TOKEN_ARGS=(--hub_token "${HF_TOKEN}")
else
  HUB_TOKEN_ARGS=()
fi

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${CACHE_DIR}"

export TOKENIZERS_PARALLELISM=false

exec "${PYTHON_BIN}" "${DIR}/run_deformable_detr_commonforms_v2.py" \
  --model_name_or_path Aryn/deformable-detr-DocLayNet \
  --dataset_name jbarrow/CommonForms \
  --output_dir "${OUTPUT_DIR}" \
  --cache_dir "${CACHE_DIR}" \
  --max_train_samples "${MAX_TRAIN_SAMPLES:-128}" \
  --max_eval_samples "${MAX_EVAL_SAMPLES:-64}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-2}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-2}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE:-2}" \
  --learning_rate "${LEARNING_RATE:-5e-5}" \
  --warmup_ratio "${WARMUP_RATIO:-0.1}" \
  --weight_decay "${WEIGHT_DECAY:-1e-4}" \
  --max_grad_norm "${MAX_GRAD_NORM:-1.0}" \
  --logging_steps "${LOGGING_STEPS:-5}" \
  --save_strategy "no" \
  --evaluation_strategy "no" \
  --seed "${SEED:-42}" \
  --fp16 \
  --do_train \
  --do_eval \
  --push_to_hub \
  --hub_model_id "${MODEL_ID}" \
  --hub_strategy "${HUB_STRATEGY:-end}" \
  "${HUB_TOKEN_ARGS[@]}" \
  --use_runpod_defaults
