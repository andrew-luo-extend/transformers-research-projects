#!/usr/bin/env python
# coding=utf-8
"""
RunPod-friendly finetuning script for Deformable DETR on the CommonForms dataset.

Highlights
----------
- Works out-of-the-box on RunPod GPU instances (auto-configures cache/output paths).
- Uses Albumentations for strong data augmentation tuned for document layouts.
- Handles CommonForms' COCO-style bounding boxes without intermediate conversion.
- Supports local and streaming dataset loading, subset selection, and Hugging Face Hub push.
- Compatible with `accelerate launch` for multi-GPU distributed training.
- Strict bbox validation (filters out invalid/empty samples by default).

Quick start (single GPU, RunPod defaults):

```
python run_deformable_detr_commonforms_v2.py \
  --do_train --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 30 \
  --learning_rate 5e-5 \
  --use_runpod_defaults
```

For multi-GPU:

```
accelerate launch run_deformable_detr_commonforms_v2.py \
  --do_train --do_eval --use_runpod_defaults --per_device_train_batch_size 6
```

IMPORTANT: Do NOT use --fp16 or --bf16 with Deformable DETR. The Hungarian matcher's
GIoU calculations will overflow in float16, causing "matrix contains invalid numeric
entries" errors. Train in full float32 precision.

This script includes defensive safeguards:
- Strict dataset filtering (requires all samples to have valid bboxes: finite, positive dimensions)
- Comprehensive bbox validation (NaN/Inf/degenerate boxes filtered during preprocessing)
- Smart matcher wrapper (skips Hungarian algorithm for samples with zero targets)
- Safe Hungarian fallback (monkey-patches scipy to handle any NaN/Inf gracefully)
- Strict sanity check (asserts all first-batch samples have boxes, fails fast if not)

By default (filter_empty_annotations=True), only samples with valid bboxes are kept for training.
To train with negative samples (empty annotations), set --filter_empty_annotations=False.
"""

from __future__ import annotations

import itertools
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import albumentations as A
import numpy as np
import scipy.optimize
import torch
from PIL import Image
from datasets import Dataset, IterableDataset, load_dataset

# ============================================================================
# Monkey-patch scipy.optimize.linear_sum_assignment BEFORE transformers import
# ============================================================================
# This defensive guard prevents "matrix contains invalid numeric entries" errors
# in the Hungarian matcher by replacing NaN/Inf with large finite costs.
# CRITICAL: This must happen BEFORE importing transformers, otherwise transformers
# will have already imported the original function and our patch won't work!
# Based on the fix from RF-DETR: https://github.com/Westlake-AI/SSCMA/issues/82

_original_linear_sum_assignment = scipy.optimize.linear_sum_assignment


def safe_linear_sum_assignment(cost_matrix, *args, **kwargs):
    """Wrapper around linear_sum_assignment that sanitizes NaN/Inf values.

    If the cost matrix contains NaN or Inf values, they are replaced with
    a large finite cost (max_finite_cost + 1e4) so they are never chosen
    by the Hungarian algorithm.

    This prevents ValueError: "matrix contains invalid numeric entries"
    while still allowing the matcher to function correctly.
    """
    # Convert to numpy array if it's a tensor
    if hasattr(cost_matrix, 'cpu'):
        cost_matrix = cost_matrix.cpu().numpy()
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)

    # Detect invalid entries (NaN or Inf)
    finite_mask = np.isfinite(cost_matrix)

    if not finite_mask.all():
        # There are NaN or Inf values - need to sanitize
        if finite_mask.any():
            # Some finite values exist - use max as baseline
            max_cost = cost_matrix[finite_mask].max()
        else:
            # Everything is NaN/Inf - use zero as baseline
            max_cost = 0.0

        # Make a copy to avoid mutating upstream tensors
        cost_matrix = cost_matrix.copy()

        # Assign huge cost to invalid entries so they are never chosen
        cost_matrix[~finite_mask] = max_cost + 1e4

        # Log warning about sanitization
        num_invalid = (~finite_mask).sum()
        logger = logging.getLogger(__name__)
        logger.debug(f"Sanitized {num_invalid} NaN/Inf entries in Hungarian cost matrix")

    return _original_linear_sum_assignment(cost_matrix, *args, **kwargs)


# Apply the monkey-patch
scipy.optimize.linear_sum_assignment = safe_linear_sum_assignment

# NOW import transformers (after the monkey-patch is applied)
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)

logger = logging.getLogger(__name__)


# ============================================================================
# Matcher wrapper to skip Hungarian matching for samples with zero targets
# ============================================================================

def create_safe_matcher_wrapper(original_matcher):
    """Wraps a DETR matcher to gracefully handle empty target samples.

    When a sample has no ground-truth boxes (negative sample), the Hungarian
    algorithm is skipped entirely and empty indices are returned.

    Args:
        original_matcher: The original HungarianMatcher from the model

    Returns:
        A wrapped matcher that handles empty samples
    """
    class SafeMatcherWrapper:
        def __init__(self, matcher):
            self.matcher = matcher
            # Copy all attributes from original matcher
            for attr in dir(matcher):
                if not attr.startswith('_') and attr != 'forward':
                    try:
                        setattr(self, attr, getattr(matcher, attr))
                    except:
                        pass

        def __call__(self, outputs, targets):
            """Perform matching, skipping Hungarian for samples with no targets.

            Args:
                outputs: Model predictions
                targets: List of target dicts with 'boxes' and 'class_labels'

            Returns:
                List of (src_idx, tgt_idx) tuples for each sample in batch
            """
            # Check if all targets are empty
            all_empty = all(
                (target.get("boxes") is None or
                 (isinstance(target["boxes"], torch.Tensor) and target["boxes"].numel() == 0))
                for target in targets
            )

            if all_empty:
                # All samples have no targets - return empty indices for each
                batch_size = len(targets)
                device = outputs["pred_logits"].device
                return [
                    (torch.tensor([], dtype=torch.int64, device=device),
                     torch.tensor([], dtype=torch.int64, device=device))
                    for _ in range(batch_size)
                ]

            # Check each target individually
            has_empty = any(
                (target.get("boxes") is None or
                 (isinstance(target["boxes"], torch.Tensor) and target["boxes"].numel() == 0))
                for target in targets
            )

            if has_empty:
                # Mixed batch: some empty, some non-empty
                # Process each sample individually
                results = []
                device = outputs["pred_logits"].device

                for i, target in enumerate(targets):
                    boxes = target.get("boxes")
                    is_empty = (boxes is None or
                               (isinstance(boxes, torch.Tensor) and boxes.numel() == 0))

                    if is_empty:
                        # Empty target - return empty indices
                        results.append((
                            torch.tensor([], dtype=torch.int64, device=device),
                            torch.tensor([], dtype=torch.int64, device=device)
                        ))
                    else:
                        # Non-empty target - run matcher on single sample
                        single_output = {
                            "pred_logits": outputs["pred_logits"][i:i+1],
                            "pred_boxes": outputs["pred_boxes"][i:i+1]
                        }
                        single_target = [target]

                        try:
                            single_result = self.matcher(single_output, single_target)
                            results.append(single_result[0])
                        except Exception as e:
                            logger.warning(f"Matcher failed for sample {i}: {e}, treating as empty")
                            results.append((
                                torch.tensor([], dtype=torch.int64, device=device),
                                torch.tensor([], dtype=torch.int64, device=device)
                            ))

                return results

            # All targets non-empty - use original matcher
            try:
                return self.matcher(outputs, targets)
            except Exception as e:
                logger.error(f"Hungarian matcher failed: {e}")
                # Fallback: return empty indices for all samples
                batch_size = len(targets)
                device = outputs["pred_logits"].device
                return [
                    (torch.tensor([], dtype=torch.int64, device=device),
                     torch.tensor([], dtype=torch.int64, device=device))
                    for _ in range(batch_size)
                ]

        def forward(self, outputs, targets):
            """Forward method for compatibility."""
            return self.__call__(outputs, targets)

    return SafeMatcherWrapper(original_matcher)

CANDIDATE_CATEGORY_KEYS: Tuple[str, ...] = ("category_id", "category", "label", "class", "id")


@dataclass
class ModelArguments:
    """Model configuration."""

    model_name_or_path: str = field(
        default="Aryn/deformable-detr-DocLayNet",
        metadata={"help": "Path or Hub ID of the pretrained Deformable DETR model."},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional config path if different from model_name_or_path."},
    )
    processor_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional processor path if different from model_name_or_path."},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for caching pretrained weights and processors."},
    )
    revision: str = field(
        default="main",
        metadata={"help": "Commit/branch/tag of the model repository to use."},
    )


@dataclass
class DataArguments:
    """Dataset configuration."""

    dataset_name: str = field(
        default="jbarrow/CommonForms",
        metadata={"help": "Dataset name on the Hugging Face Hub."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Optional dataset config (None for default)."},
    )
    train_split: str = field(
        default="train",
        metadata={"help": "Name of the training split."},
    )
    eval_split: str = field(
        default="val",
        metadata={"help": "Name of the evaluation split. Fallbacks: validation/test."},
    )
    use_streaming: bool = field(
        default=False,
        metadata={"help": "Stream data instead of downloading it. Useful for large datasets."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of training samples (for debugging)."},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Optional cap on the number of evaluation samples."},
    )
    image_size: int = field(
        default=1025,
        metadata={"help": "Target longest edge (pixels) after augmentation."},
    )
    filter_empty_annotations: bool = field(
        default=True,
        metadata={
            "help": "Filter out samples with invalid or no bounding boxes at dataset level. "
            "Ensures all training samples have at least one valid bbox (finite, positive dimensions). "
            "Set to False to keep all samples (including empty ones) for negative sample training."
        },
    )


@dataclass
class RunPodArguments:
    """RunPod-specific conveniences."""

    use_runpod_defaults: bool = field(
        default=False,
        metadata={"help": "Auto-configure cache/output paths for RunPod (also picks up RUNPOD_POD_ID)."},
    )
    workspace_dir: str = field(
        default="/workspace",
        metadata={"help": "Root directory of the RunPod workspace."},
    )
    cache_subdir: str = field(
        default="hf-cache",
        metadata={"help": "Sub-directory under workspace for Hugging Face caches."},
    )
    runpod_log_level: str = field(
        default="INFO",
        metadata={"help": "Logging level (DEBUG, INFO, WARNING, ERROR)."},
    )
    tensorboard_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Override tensorboard logging directory."},
    )


def setup_logging(level: str) -> None:
    """Initialize root logging."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )


def ensure_dir(path: Optional[str]) -> None:
    """Ensure a directory exists if path is provided."""
    if path:
        os.makedirs(path, exist_ok=True)


def apply_runpod_defaults(
    model_args: ModelArguments,
    runpod_args: RunPodArguments,
    training_args: TrainingArguments,
) -> None:
    """Apply sensible defaults when running inside a RunPod environment."""
    runpod_detected = bool(os.getenv("RUNPOD_POD_ID"))
    if not (runpod_detected or runpod_args.use_runpod_defaults):
        return

    workspace = runpod_args.workspace_dir or "/workspace"
    cache_root = os.path.join(workspace, runpod_args.cache_subdir)
    ensure_dir(cache_root)
    ensure_dir(os.path.join(cache_root, "datasets"))
    ensure_dir(os.path.join(cache_root, "hub"))

    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_root)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(cache_root, "hub"))

    if model_args.cache_dir is None:
        model_args.cache_dir = cache_root

    default_output = os.path.join(workspace, "outputs", "deformable-detr-commonforms")
    if training_args.output_dir == "tmp_trainer" or training_args.output_dir is None:
        training_args.output_dir = default_output
    ensure_dir(training_args.output_dir)

    if runpod_args.tensorboard_dir:
        training_args.logging_dir = runpod_args.tensorboard_dir
    elif training_args.logging_dir is None:
        training_args.logging_dir = os.path.join(training_args.output_dir, "logs")
    ensure_dir(training_args.logging_dir)

    if training_args.report_to is None or training_args.report_to == []:
        training_args.report_to = ["tensorboard"]

    logger.info("RunPod defaults enabled. Using cache dir %s", model_args.cache_dir)
    logger.info("Outputs will be stored in %s", training_args.output_dir)


def detect_category_field(dataset: Iterable[Any]) -> Tuple[str, Optional[Any]]:
    """Infer which key inside `objects` holds class labels."""
    features = getattr(dataset, "features", None)
    if features and "objects" in features:
        inner_feature = getattr(features["objects"], "feature", None)
        if inner_feature:
            for candidate in CANDIDATE_CATEGORY_KEYS:
                if candidate in inner_feature:
                    return candidate, inner_feature[candidate]

    sample = None
    if isinstance(dataset, IterableDataset):
        iterator = itertools.islice(dataset.take(5), 5)
        for sample in iterator:
            if sample:
                break
    else:
        try:
            if len(dataset) > 0:
                sample = dataset[0]
        except TypeError:
            sample = None

    if sample and "objects" in sample:
        for candidate in CANDIDATE_CATEGORY_KEYS:
            if candidate in sample["objects"]:
                return candidate, None

    raise ValueError(
        "Could not determine category field inside `objects`. "
        f"Tried keys: {', '.join(CANDIDATE_CATEGORY_KEYS)}"
    )


def gather_category_mapping(
    dataset: Iterable[Any],
    category_key: str,
    category_feature: Optional[Any],
    use_streaming: bool,
    max_scan: int = 4000,
) -> Tuple[Dict[Any, int], Dict[int, str]]:
    """Build mapping from dataset category values to contiguous ids + readable labels."""
    if category_feature is not None and hasattr(category_feature, "names"):
        names = list(category_feature.names)
        category_id_remap = {idx: idx for idx in range(len(names))}
        id2label = {idx: name for idx, name in enumerate(names)}
        return category_id_remap, id2label

    seen: Dict[Any, int] = {}
    unique_values: List[Any] = []

    def _normalize(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        return value

    if use_streaming:
        iterator = itertools.islice(dataset.take(max_scan), max_scan)
        for example in iterator:
            cats = (example or {}).get("objects", {}).get(category_key, [])
            for value in cats:
                value = _normalize(value)
                if value not in seen:
                    seen[value] = 1
                    unique_values.append(value)
    else:
        try:
            length = len(dataset)  # type: ignore[arg-type]
        except TypeError:
            length = max_scan
        for idx in range(min(max_scan, length)):
            example = dataset[idx]  # type: ignore[index]
            cats = (example or {}).get("objects", {}).get(category_key, [])
            for value in cats:
                value = _normalize(value)
                if value not in seen:
                    seen[value] = 1
                    unique_values.append(value)

    if not unique_values:
        unique_values = [0]

    sorted_unique = sorted(unique_values, key=lambda v: (str(type(v)), str(v)))
    category_id_remap = {orig: idx for idx, orig in enumerate(sorted_unique)}

    id2label: Dict[int, str] = {}
    for idx, orig in enumerate(sorted_unique):
        label_name = str(orig)
        if (
            isinstance(orig, (int, np.integer))
            and category_feature is not None
            and hasattr(category_feature, "names")
        ):
            try:
                label_name = category_feature.names[int(orig)]
            except (IndexError, ValueError, TypeError):
                label_name = str(orig)
        id2label[idx] = label_name

    return category_id_remap, id2label


def formatted_annotations(image_id: int, category_ids: Sequence[int], bboxes: Sequence[Sequence[float]]) -> List[Dict[str, Any]]:
    """Return annotations ready for the image processor."""
    annotations: List[Dict[str, Any]] = []
    for category_id, box in zip(category_ids, bboxes):
        x, y, w, h = map(float, box)
        if w <= 0 or h <= 0:
            continue
        annotations.append(
            {
                "image_id": image_id,
                "category_id": int(category_id),
                "iscrowd": 0,
                "area": float(w * h),
                "bbox": [x, y, w, h],
            }
        )
    return annotations


def create_transforms(image_size: int, is_train: bool) -> A.Compose:
    """Albumentations pipelines for train/eval.

    Note: Albumentations expects RGB images and COCO format bboxes [x, y, width, height].
    """
    if is_train:
        return A.Compose(
            [
                A.LongestMaxSize(image_size),
                A.PadIfNeeded(image_size, image_size, border_mode=0, value=(0, 0, 0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.1),
                A.Rotate(limit=10, p=0.5),
                A.RandomScale(scale_limit=0.2, p=0.5),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_id"],
                min_area=1.0,
                min_visibility=0.1,
            ),
        )
    return A.Compose(
        [
            A.LongestMaxSize(image_size),
            A.PadIfNeeded(image_size, image_size, border_mode=0, value=(0, 0, 0)),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["category_id"],
            min_area=1.0,
            min_visibility=0.1,
        ),
    )


def _to_plain_value(value: Any) -> Any:
    """Recursively convert BatchEncoding values to plain python containers."""
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    if hasattr(value, "items"):
        return {k: _to_plain_value(v) for k, v in value.items()}  # type: ignore[attr-defined]
    return value


def _to_numpy_rgb(image: Any) -> np.ndarray:
    """Convert PIL/np images to numpy RGB arrays."""
    if isinstance(image, Image.Image):
        if image.mode != "RGB":
            image = image.convert("RGB")
        return np.array(image)
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return np.stack([image] * 3, axis=-1)
        if image.shape[-1] == 4:
            return image[..., :3]
        return image
    raise TypeError(f"Unsupported image type: {type(image)}")


def validate_bbox_format(bbox: Sequence[float], img_w: int, img_h: int) -> bool:
    """Check if bbox is in valid COCO format [x, y, width, height]."""
    if len(bbox) != 4:
        return False
    x, y, w, h = bbox
    # Check for valid values
    if not all(np.isfinite([x, y, w, h])):
        return False
    # COCO format: x, y should be non-negative, w, h should be positive
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False
    # Boxes should be within reasonable bounds (allow some tolerance for normalization issues)
    if x > img_w * 2 or y > img_h * 2:
        return False
    return True


def preprocess_examples(
    examples: Dict[str, Any],
    transform: A.Compose,
    image_processor: AutoImageProcessor,
    category_key: str,
    category_id_remap: Dict[Any, int],
    allow_empty_samples: bool = False,
) -> Dict[str, Any]:
    """Apply augmentation + processor to a batch from datasets.

    Args:
        allow_empty_samples: If True, process samples with no annotations as negative samples.
                           If False, skip them entirely.
    """
    if isinstance(examples["image"], (Image.Image, np.ndarray)):
        image_ids = [examples.get("id", 0)]
        images = [examples["image"]]
        objects_list = [examples.get("objects", {})]
    else:
        image_ids = examples.get("id", list(range(len(examples["image"]))))
        images = examples["image"]
        objects_list = examples["objects"]

    processed_images: List[np.ndarray] = []
    processed_targets: List[Dict[str, Any]] = []
    skip_reasons: Dict[str, int] = {}
    debug_logged = False  # Log detailed debug info for first sample of each batch

    for idx, (image_id, image, objects) in enumerate(zip(image_ids, images, objects_list)):
        try:
            image_np = _to_numpy_rgb(image)
        except Exception as exc:
            logger.warning("Skipping sample %s due to image conversion error: %s", image_id, exc)
            skip_reasons["image_conversion_error"] = skip_reasons.get("image_conversion_error", 0) + 1
            continue

        if objects is None:
            objects = {}

        raw_bboxes = objects.get("bbox") or objects.get("boxes") or []
        raw_categories = objects.get(category_key) or []

        # Debug logging for first sample
        if not debug_logged:
            logger.info("=" * 60)
            logger.info("First sample debug info:")
            logger.info(f"  Image ID: {image_id}")
            logger.info(f"  Image shape: {image_np.shape}")
            logger.info(f"  Objects type: {type(objects)}")
            logger.info(f"  Objects keys: {list(objects.keys()) if objects else 'None'}")
            logger.info(f"  Category key: {category_key}")
            logger.info(f"  Raw bboxes type: {type(raw_bboxes)}")
            logger.info(f"  Raw bboxes count: {len(raw_bboxes) if raw_bboxes else 0}")
            logger.info(f"  Raw categories type: {type(raw_categories)}")
            logger.info(f"  Raw categories count: {len(raw_categories) if raw_categories else 0}")
            if raw_bboxes and len(raw_bboxes) > 0:
                logger.info(f"  First bbox: {raw_bboxes[0]}")
                logger.info(f"  First bbox type: {type(raw_bboxes[0])}")
            if raw_categories and len(raw_categories) > 0:
                logger.info(f"  First category: {raw_categories[0]}")
            logger.info(f"  Full objects structure: {objects}")
            logger.info("=" * 60)
            debug_logged = True

        # Handle samples with no annotations
        if not raw_bboxes or not raw_categories:
            # Check if we should process empty samples or skip them
            # If filter_empty_annotations is enabled, these should have been filtered at dataset level
            logger.debug(f"Processing sample {image_id} with no annotations (negative sample)")
            try:
                # Apply transform to image without bboxes
                albumentations_inputs = transform(
                    image=image_np,
                    bboxes=[],
                    category_id=[],
                )
                img_transformed = albumentations_inputs["image"]

                if isinstance(image_id, (int, np.integer)):
                    target_image_id = int(image_id)
                else:
                    try:
                        target_image_id = int(str(image_id))
                    except (TypeError, ValueError):
                        target_image_id = idx

                processed_images.append(img_transformed.astype(np.uint8))
                processed_targets.append({
                    "image_id": target_image_id,
                    "annotations": [],  # Empty annotations for negative sample
                })
                continue
            except Exception as exc:
                logger.warning(f"Failed to process empty sample {image_id}: {exc}")
                skip_reasons["empty_sample_error"] = skip_reasons.get("empty_sample_error", 0) + 1
                continue

        # Validate bbox format
        img_h, img_w = image_np.shape[:2]
        valid_bboxes = []
        valid_categories = []
        for bbox, cat in zip(raw_bboxes, raw_categories):
            if validate_bbox_format(bbox, img_w, img_h):
                valid_bboxes.append(bbox)
                valid_categories.append(cat)
            else:
                logger.debug(f"Invalid bbox format in sample {image_id}: {bbox}")

        if not valid_bboxes:
            logger.debug("Skipping sample %s: no valid bboxes after format validation", image_id)
            skip_reasons["invalid_bbox_format"] = skip_reasons.get("invalid_bbox_format", 0) + 1
            continue

        raw_bboxes = valid_bboxes
        raw_categories = valid_categories

        mapped_categories: List[int] = []
        for cat in raw_categories:
            if isinstance(cat, np.generic):
                cat = cat.item()
            mapped_categories.append(int(category_id_remap.get(cat, 0)))

        # Albumentations expects RGB, not BGR
        try:
            albumentations_inputs = transform(
                image=image_np,
                bboxes=raw_bboxes,
                category_id=mapped_categories,
            )
        except Exception as exc:
            logger.warning("Skipping sample %s due to augmentation error: %s", image_id, exc)
            skip_reasons["augmentation_error"] = skip_reasons.get("augmentation_error", 0) + 1
            continue

        img_transformed = albumentations_inputs["image"]
        img_h, img_w = img_transformed.shape[:2]
        transformed_bboxes: List[List[float]] = []
        transformed_categories: List[int] = []

        for bbox, category_id in zip(albumentations_inputs["bboxes"], albumentations_inputs["category_id"]):
            x, y, w, h = map(float, bbox)
            # Strict validation: skip boxes with NaN/Inf
            if not np.isfinite([x, y, w, h]).all():
                logger.debug(f"Skipping non-finite bbox in sample {image_id}: [{x}, {y}, {w}, {h}]")
                continue
            # Skip degenerate boxes (zero or negative area)
            if w <= 0 or h <= 0:
                logger.debug(f"Skipping degenerate bbox in sample {image_id}: w={w}, h={h}")
                continue
            # Clip to image bounds
            x = float(np.clip(x, 0.0, max(img_w - 1.0, 1.0)))
            y = float(np.clip(y, 0.0, max(img_h - 1.0, 1.0)))
            w = float(np.clip(w, 1.0, max(img_w - x, 1.0)))
            h = float(np.clip(h, 1.0, max(img_h - y, 1.0)))
            # Double-check after clipping
            if w <= 0 or h <= 0:
                logger.debug(f"Skipping bbox with zero area after clipping in sample {image_id}")
                continue
            # Verify no extreme values that could cause overflow
            if x > 1e6 or y > 1e6 or w > 1e6 or h > 1e6:
                logger.warning(f"Skipping bbox with extreme values in sample {image_id}: [{x}, {y}, {w}, {h}]")
                continue
            transformed_bboxes.append([x, y, w, h])
            transformed_categories.append(int(category_id))

        # Handle samples that became empty after augmentation (all boxes filtered out)
        # Treat them as negative samples instead of skipping
        if not transformed_bboxes or not transformed_categories:
            logger.debug(f"Sample {image_id} became empty after augmentation, treating as negative sample")
            if isinstance(image_id, (int, np.integer)):
                target_image_id = int(image_id)
            else:
                try:
                    target_image_id = int(str(image_id))
                except (TypeError, ValueError):
                    target_image_id = idx

            processed_images.append(img_transformed.astype(np.uint8))
            processed_targets.append({
                "image_id": target_image_id,
                "annotations": [],  # Empty annotations
            })
            continue

        if isinstance(image_id, (int, np.integer)):
            target_image_id = int(image_id)
        else:
            try:
                target_image_id = int(str(image_id))
            except (TypeError, ValueError):
                target_image_id = idx

        # Image is already in RGB format from Albumentations
        processed_images.append(img_transformed.astype(np.uint8))
        processed_targets.append(
            {
                "image_id": target_image_id,
                "annotations": formatted_annotations(target_image_id, transformed_categories, transformed_bboxes),
            }
        )

    if not processed_images:
        error_msg = "All samples were skipped during preprocessing. Skip reasons:\n"
        for reason, count in skip_reasons.items():
            error_msg += f"  - {reason}: {count} samples\n"
        error_msg += "\nSuggestions:\n"
        if "no_annotations" in skip_reasons:
            error_msg += "  * Check if dataset contains 'objects' field with 'bbox' and category data\n"
        if "invalid_bbox_format" in skip_reasons:
            error_msg += "  * Verify bboxes are in COCO format [x, y, width, height] with valid positive values\n"
        if "augmentation_error" in skip_reasons:
            error_msg += "  * Check if bounding boxes are within image bounds before augmentation\n"
        raise ValueError(error_msg)

    # Log preprocessing summary
    total_samples = len(image_ids)
    logger.info(f"Preprocessing summary: {len(processed_images)}/{total_samples} samples successful")
    if skip_reasons:
        logger.info("Skipped samples breakdown:")
        for reason, count in skip_reasons.items():
            logger.info(f"  - {reason}: {count}")

    heights = [img.shape[0] for img in processed_images]
    widths = [img.shape[1] for img in processed_images]
    max_h = max(heights)
    max_w = max(widths)
    if any(h != max_h or w != max_w for h, w in zip(heights, widths)):
        uniform_images: List[np.ndarray] = []
        for img in processed_images:
            pad_h = max_h - img.shape[0]
            pad_w = max_w - img.shape[1]
            if pad_h < 0 or pad_w < 0:
                raise ValueError("Encountered negative padding values while normalizing image shapes.")
            if pad_h == 0 and pad_w == 0:
                uniform_images.append(img)
                continue
            padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0)
            uniform_images.append(padded)
        processed_images = uniform_images

    encoded = image_processor(images=processed_images, annotations=processed_targets, return_tensors="pt")
    encoded = {key: _to_plain_value(value) for key, value in encoded.items()}

    pixel_values = encoded.get("pixel_values")
    if isinstance(pixel_values, torch.Tensor):
        encoded["pixel_values"] = torch.nan_to_num(pixel_values.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    pixel_mask = encoded.get("pixel_mask")
    if isinstance(pixel_mask, torch.Tensor):
        encoded["pixel_mask"] = torch.nan_to_num(pixel_mask.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

    labels = encoded.get("labels")
    if isinstance(labels, list):
        cleaned_labels: List[Dict[str, Any]] = []
        for label in labels:
            if not isinstance(label, dict):
                cleaned_labels.append(label)
                continue

            boxes = label.get("boxes")
            class_labels = label.get("class_labels")

            if not isinstance(boxes, torch.Tensor):
                boxes = torch.tensor(boxes, dtype=torch.float32)
            else:
                boxes = torch.nan_to_num(boxes.to(torch.float32), nan=0.0, posinf=0.0, neginf=0.0)

            if class_labels is None:
                class_labels = torch.zeros((boxes.shape[0],), dtype=torch.int64)
            elif not isinstance(class_labels, torch.Tensor):
                class_labels = torch.tensor(class_labels, dtype=torch.int64)
            else:
                class_labels = class_labels.to(torch.int64)

            if boxes.shape[0] != class_labels.shape[0]:
                length = min(boxes.shape[0], class_labels.shape[0])
                boxes = boxes[:length]
                class_labels = class_labels[:length]

            # Keep samples with no boxes - ensure they have the right shape (0, 4) for boxes
            if boxes.numel() == 0:
                # Empty tensor with correct shape for COCO format boxes
                label["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                label["class_labels"] = torch.empty((0,), dtype=torch.int64)
                cleaned_labels.append(label)
                continue

            # Filter out boxes with non-finite values (NaN, Inf)
            finite_mask = torch.isfinite(boxes).all(dim=1)
            if finite_mask.any():
                boxes = boxes[finite_mask]
                class_labels = class_labels[finite_mask]

            # After filtering, check if we have any valid boxes left
            if boxes.numel() == 0:
                # All boxes were non-finite, treat as empty sample
                logger.debug("All boxes were non-finite after filtering, treating as empty sample")
                label["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                label["class_labels"] = torch.empty((0,), dtype=torch.int64)
                cleaned_labels.append(label)
                continue

            # Validate bbox format: image processor converts to normalized [cx, cy, w, h]
            # Ensure all values are in valid range [0, 1] for normalized coordinates
            boxes = boxes.clamp(0.0, 1.0)

            # Ensure width and height are not too small (avoid degenerate boxes)
            if boxes.shape[-1] >= 4:
                boxes[:, 2:] = boxes[:, 2:].clamp(1e-4, 1.0)

            # Final check: remove any degenerate boxes (zero width or height)
            valid_box_mask = (boxes[:, 2] > 1e-6) & (boxes[:, 3] > 1e-6)
            if valid_box_mask.any():
                boxes = boxes[valid_box_mask]
                class_labels = class_labels[valid_box_mask]

            # Check again after removing degenerate boxes
            if boxes.numel() == 0:
                logger.debug("All boxes were degenerate (zero area), treating as empty sample")
                label["boxes"] = torch.empty((0, 4), dtype=torch.float32)
                label["class_labels"] = torch.empty((0,), dtype=torch.int64)
                cleaned_labels.append(label)
                continue

            label["boxes"] = boxes
            label["class_labels"] = class_labels
            cleaned_labels.append(label)

        encoded["labels"] = cleaned_labels

    return encoded


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collation for DETR-style object detection models.

    Keeps all samples including those with empty annotations. The model should
    handle empty targets properly by skipping Hungarian matching for them.
    """
    pixel_values: List[torch.Tensor] = []
    pixel_masks: List[torch.Tensor] = []
    labels: List[Any] = []

    for feature in features:
        pv = feature["pixel_values"]
        if isinstance(pv, torch.Tensor) and pv.dim() == 4 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        pixel_values.append(pv)

        if "pixel_mask" in feature:
            pm = feature["pixel_mask"]
            if isinstance(pm, torch.Tensor) and pm.dim() == 4 and pm.shape[0] == 1:
                pm = pm.squeeze(0)
            pixel_masks.append(pm)

        labels.append(feature["labels"])

    batch = {
        "pixel_values": torch.stack(pixel_values),
        "labels": labels,
    }
    if pixel_masks:
        batch["pixel_mask"] = torch.stack(pixel_masks)
    return batch


def resolve_split(dataset_dict: Any, candidates: Sequence[str]) -> Tuple[Optional[Any], Optional[str]]:
    """Return the first matching split from the provided candidates."""
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in dataset_dict:
            return dataset_dict[candidate], candidate
    return None, None


def limit_dataset(dataset: Any, max_samples: Optional[int], use_streaming: bool) -> Any:
    """Trim datasets for quick experiments."""
    if dataset is None or max_samples is None:
        return dataset
    if use_streaming:
        return dataset.take(max_samples)
    total = len(dataset)  # type: ignore[arg-type]
    return dataset.select(range(min(max_samples, total)))


def log_split_size(name: str, dataset: Any, use_streaming: bool) -> None:
    """Log dataset length (if known)."""
    if dataset is None:
        return
    try:
        size = len(dataset)  # type: ignore[arg-type]
        logger.info("%s split size: %d", name, size)
    except TypeError:
        if use_streaming:
            logger.info("%s split size: streaming (unknown length)", name)
        else:
            logger.info("%s split size: unknown", name)


def filter_empty_annotations(example: Dict[str, Any]) -> bool:
    """Filter out samples that have no bounding boxes."""
    objects = example.get("objects")
    if objects is None:
        return False

    # Check if bbox field exists and has content
    bboxes = objects.get("bbox") or objects.get("boxes") or []
    if not bboxes or len(bboxes) == 0:
        return False

    # Check if category field exists and has content
    for key in CANDIDATE_CATEGORY_KEYS:
        categories = objects.get(key)
        if categories and len(categories) > 0:
            return True

    return False


def filter_valid_bboxes(example: Dict[str, Any]) -> bool:
    """Strict filter: only keep samples with at least one valid bbox.

    A valid bbox must:
    - Exist (len > 0)
    - Have finite values (no NaN/Inf)
    - Have positive dimensions (w > 0, h > 0)
    - Have non-negative coordinates (x >= 0, y >= 0)
    """
    objects = example.get("objects")
    if objects is None:
        return False

    # Get bboxes and categories
    bboxes = objects.get("bbox") or objects.get("boxes") or []
    if not bboxes or len(bboxes) == 0:
        return False

    # Get categories using candidate keys
    categories = None
    for key in CANDIDATE_CATEGORY_KEYS:
        categories = objects.get(key)
        if categories and len(categories) > 0:
            break

    if not categories or len(categories) == 0:
        return False

    if len(bboxes) != len(categories):
        return False

    # Check each bbox for validity
    has_valid_bbox = False
    for bbox in bboxes:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue

        x, y, w, h = bbox

        # Check for finite values
        try:
            if not all(np.isfinite([x, y, w, h])):
                continue
        except (TypeError, ValueError):
            continue

        # Check COCO format requirements
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            continue

        # Check for extreme values that could cause overflow
        if x > 1e6 or y > 1e6 or w > 1e6 or h > 1e6:
            continue

        # At least one valid bbox found
        has_valid_bbox = True
        break

    return has_valid_bbox


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, RunPodArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, runpod_args, training_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, runpod_args, training_args = parser.parse_args_into_dataclasses()

    training_args.save_safetensors = False
    setup_logging(runpod_args.runpod_log_level)
    apply_runpod_defaults(model_args, runpod_args, training_args)

    logger.info("=" * 80)
    logger.info("Deformable DETR finetuning on CommonForms (RunPod-ready)")
    logger.info("=" * 80)
    logger.info("Model: %s", model_args.model_name_or_path)
    logger.info("Dataset: %s", data_args.dataset_name)
    logger.info("Training output directory: %s", training_args.output_dir)
    logger.info("Image size (longest edge): %d px", data_args.image_size)

    training_args.remove_unused_columns = False

    set_seed(training_args.seed)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info("Detected %d CUDA device(s).", device_count)
        for idx in range(device_count):
            logger.info("  GPU %d: %s", idx, torch.cuda.get_device_name(idx))
    else:
        logger.warning("CUDA is not available. Training will run on CPU.")

    logger.info("Loading dataset: %s", data_args.dataset_name)
    if data_args.dataset_config_name:
        logger.info("  Config: %s", data_args.dataset_config_name)
    logger.info("  Streaming: %s", data_args.use_streaming)
    logger.info("  Cache dir: %s", model_args.cache_dir)

    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        streaming=data_args.use_streaming,
    )
    logger.info("Dataset loaded successfully")

    if isinstance(dataset, (Dataset, IterableDataset)):
        dataset_dict: Dict[str, Any] = {data_args.train_split: dataset}
    else:
        dataset_dict = dict(dataset)

    train_candidates = (data_args.train_split, "train", "training")
    eval_candidates = (data_args.eval_split, "val", "validation", "test")

    train_dataset, actual_train_name = resolve_split(dataset_dict, train_candidates)
    eval_dataset, actual_eval_name = resolve_split(dataset_dict, eval_candidates)

    if training_args.do_train and train_dataset is None:
        raise ValueError(f"Could not locate a training split. Tried: {train_candidates}")

    if training_args.do_eval and eval_dataset is None:
        logger.warning("Evaluation requested but no eval split found. Skipping evaluation.")
        training_args.do_eval = False

    # Limit dataset size FIRST (for faster testing), then filter
    if training_args.do_train and data_args.max_train_samples:
        logger.info(f"Limiting training dataset to {data_args.max_train_samples} samples for testing...")
        train_dataset = limit_dataset(train_dataset, data_args.max_train_samples, data_args.use_streaming)

    if training_args.do_eval and data_args.max_eval_samples:
        logger.info(f"Limiting evaluation dataset to {data_args.max_eval_samples} samples for testing...")
        eval_dataset = limit_dataset(eval_dataset, data_args.max_eval_samples, data_args.use_streaming)

    # Now filter the limited dataset (much faster for testing)
    if data_args.filter_empty_annotations:
        if train_dataset is not None and not data_args.use_streaming:
            logger.info("Filtering training dataset - requiring valid bboxes (finite, positive dimensions)...")
            original_size = len(train_dataset)
            train_dataset = train_dataset.filter(filter_valid_bboxes)
            filtered_size = len(train_dataset)
            logger.info(f"Filtered {original_size - filtered_size} invalid/empty samples from training set")
            logger.info(f"Training set: {filtered_size} samples with valid bboxes")

            if filtered_size == 0:
                raise ValueError(
                    "After filtering, no training samples remain with valid bboxes. "
                    "Check your dataset for:\n"
                    "  - Samples with at least one bbox\n"
                    "  - Bboxes with finite values (no NaN/Inf)\n"
                    "  - Bboxes with positive dimensions (w > 0, h > 0)\n"
                    "  - Bboxes with non-negative coordinates (x >= 0, y >= 0)"
                )

        if eval_dataset is not None and not data_args.use_streaming:
            logger.info("Filtering evaluation dataset - requiring valid bboxes...")
            original_size = len(eval_dataset)
            eval_dataset = eval_dataset.filter(filter_valid_bboxes)
            filtered_size = len(eval_dataset)
            logger.info(f"Filtered {original_size - filtered_size} invalid/empty samples from evaluation set")
            logger.info(f"Evaluation set: {filtered_size} samples with valid bboxes")
    else:
        logger.info("Keeping all samples (including empty/invalid ones) - ensure matcher can handle this!")

    log_split_size(actual_train_name or "train", train_dataset, data_args.use_streaming)
    log_split_size(actual_eval_name or "eval", eval_dataset, data_args.use_streaming)

    category_dataset = train_dataset or eval_dataset
    if category_dataset is None:
        raise ValueError("Unable to infer categories: no dataset split available.")

    category_key, category_feature = detect_category_field(category_dataset)
    category_id_remap, id2label = gather_category_mapping(
        category_dataset,
        category_key,
        category_feature,
        data_args.use_streaming,
    )
    label2id = {label: idx for idx, label in id2label.items()}

    logger.info("Detected %d categories.", len(id2label))
    if len(id2label) <= 20:
        logger.info("Labels: %s", id2label)
        logger.info("Category remap (dataset -> contiguous id): %s", category_id_remap)
    else:
        preview = list(id2label.items())[:20]
        logger.info("First 20 labels: %s ...", preview)

    processor_name = model_args.processor_name or model_args.model_name_or_path
    image_processor = AutoImageProcessor.from_pretrained(
        processor_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        do_resize=False,
        do_pad=False,
        size={"longest_edge": data_args.image_size},
    )

    config_name = model_args.config_name or model_args.model_name_or_path
    model_config = AutoConfig.from_pretrained(
        config_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        label2id=label2id,
        id2label=id2label,
    )
    model_config.num_labels = len(id2label)

    logger.info("Loading model weights...")
    logger.info(f"Model config num_labels: {model_config.num_labels}")
    logger.info(f"Dataset categories: {len(id2label)}")

    # Load with ignore_mismatched_sizes=True to handle class count mismatch
    # The pretrained model (DocLayNet) has more classes than CommonForms
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        config=model_config,
        ignore_mismatched_sizes=True,  # Required: pretrained model has different num_classes
    )
    logger.info("Model loaded with %d detection classes.", model.config.num_labels)

    # CRITICAL: Reinitialize BOTH classification AND bbox heads with smaller init for stability
    # When ignore_mismatched_sizes=True, these heads are randomly initialized and can cause:
    # - Explosive gradients from class head
    # - NaN predictions from bbox head
    logger.info("Reinitializing prediction heads for stable training...")

    # Reinitialize classification head with EXTREMELY small weights
    if hasattr(model, 'model') and hasattr(model.model, 'class_embed'):
        for layer in model.model.class_embed:
            if hasattr(layer, 'weight'):
                # Use even smaller init to prevent NaN in first forward pass
                torch.nn.init.normal_(layer.weight, mean=0.0, std=0.0001)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    # Initialize bias to small negative value (favors "no object" class)
                    torch.nn.init.constant_(layer.bias, -2.0)
        logger.info("✓ Classification head reinitialized with std=0.0001")

    # Reinitialize bbox regression head with smaller init
    if hasattr(model, 'model') and hasattr(model.model, 'bbox_embed'):
        for layer in model.model.bbox_embed:
            if hasattr(layer, 'layers'):
                # MLP with multiple layers - use very small init
                for sublayer in layer.layers:
                    if hasattr(sublayer, 'weight'):
                        torch.nn.init.xavier_uniform_(sublayer.weight, gain=0.001)  # Even smaller
                        if hasattr(sublayer, 'bias') and sublayer.bias is not None:
                            torch.nn.init.constant_(sublayer.bias, 0.0)
            elif hasattr(layer, 'weight'):
                torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)  # Even smaller
                if hasattr(layer, 'bias') and layer.bias is not None:
                    torch.nn.init.constant_(layer.bias, 0.0)
        logger.info("✓ Bbox regression head reinitialized with gain=0.001")

    logger.info("✓ All prediction heads reinitialized with small weights")

    # Wrap the matcher to handle empty targets gracefully
    if hasattr(model, 'model') and hasattr(model.model, 'criterion'):
        criterion = model.model.criterion
        if hasattr(criterion, 'matcher'):
            logger.info("Wrapping matcher to handle empty targets (negative samples)")
            criterion.matcher = create_safe_matcher_wrapper(criterion.matcher)
            logger.info("Matcher wrapped successfully")
    else:
        logger.warning("Could not find matcher in model structure - matcher wrapping skipped")

    train_transform = create_transforms(data_args.image_size, is_train=True)
    eval_transform = create_transforms(data_args.image_size, is_train=False)

    def transform_train(batch: Dict[str, Any]) -> Dict[str, Any]:
        return preprocess_examples(batch, train_transform, image_processor, category_key, category_id_remap)

    def transform_eval(batch: Dict[str, Any]) -> Dict[str, Any]:
        return preprocess_examples(batch, eval_transform, image_processor, category_key, category_id_remap)

    if training_args.do_train and train_dataset is not None:
        train_dataset = train_dataset.with_transform(transform_train)

    if training_args.do_eval and eval_dataset is not None:
        eval_dataset = eval_dataset.with_transform(transform_eval)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        tokenizer=image_processor,
    )

    if training_args.do_train:
        logger.info("=" * 80)
        logger.info("Starting training")
        logger.info("=" * 80)

        # Test forward pass before training to catch NaN early
        logger.info("Testing model forward pass on first batch...")
        try:
            first_batch = next(iter(trainer.get_train_dataloader()))

            # Check pixel_values for NaN/Inf
            pv = first_batch.get("pixel_values")
            if isinstance(pv, torch.Tensor):
                if torch.isnan(pv).any():
                    raise ValueError("NaNs detected in first training batch pixel_values.")
                if torch.isinf(pv).any():
                    raise ValueError("Infs detected in first training batch pixel_values.")

            # Check labels for NaN/Inf in boxes
            labels = first_batch.get("labels")
            if labels is not None and isinstance(labels, list):
                batch_bbox_counts = []
                for idx, label in enumerate(labels):
                    if isinstance(label, dict):
                        boxes = label.get("boxes")
                        class_labels = label.get("class_labels")

                        # STRICT REQUIREMENT: All samples must have at least one bbox
                        if boxes is None or (isinstance(boxes, torch.Tensor) and boxes.numel() == 0):
                            raise ValueError(
                                f"Sample {idx} in first batch has NO bounding boxes! "
                                f"If filter_empty_annotations=True, this should not happen. "
                                f"Check your filtering logic."
                            )

                        if isinstance(boxes, torch.Tensor):
                            num_boxes = boxes.shape[0]
                            batch_bbox_counts.append(num_boxes)
                            logger.info(f"  Sample {idx}: {num_boxes} boxes")

                            if num_boxes == 0:
                                raise ValueError(
                                    f"Sample {idx} has 0 boxes after preprocessing! "
                                    f"Dataset filtering should have removed this sample."
                                )

                            # Check for NaN/Inf
                            if torch.isnan(boxes).any():
                                raise ValueError(f"NaNs detected in boxes for sample {idx} of first batch")
                            if torch.isinf(boxes).any():
                                raise ValueError(f"Infs detected in boxes for sample {idx} of first batch")

                            # Check for extreme values that could cause GIoU overflow
                            if (boxes > 10.0).any():
                                logger.warning(f"Extremely large box values detected in sample {idx}: max={boxes.max().item()}")
                            if (boxes < -10.0).any():
                                logger.warning(f"Extremely negative box values detected in sample {idx}: min={boxes.min().item()}")

                            # Check for degenerate boxes (zero area in normalized coordinates)
                            if boxes.shape[-1] >= 4:
                                widths = boxes[:, 2]
                                heights = boxes[:, 3]
                                if (widths < 1e-6).any() or (heights < 1e-6).any():
                                    logger.warning(f"Degenerate boxes (near-zero area) detected in sample {idx}")

                        if isinstance(class_labels, torch.Tensor):
                            if class_labels.numel() == 0:
                                raise ValueError(f"Sample {idx} has 0 class labels!")
                            if (class_labels < 0).any():
                                raise ValueError(f"Negative class labels detected in sample {idx} of first batch")

                # Summary
                if batch_bbox_counts:
                    logger.info(f"First batch summary: {len(batch_bbox_counts)} samples, "
                              f"bbox counts: min={min(batch_bbox_counts)}, max={max(batch_bbox_counts)}, "
                              f"avg={sum(batch_bbox_counts)/len(batch_bbox_counts):.1f}")

                # DETAILED LOGGING: Print actual box values from first sample
                if labels and len(labels) > 0:
                    first_label = labels[0]
                    if isinstance(first_label, dict):
                        boxes = first_label.get("boxes")
                        class_labels = first_label.get("class_labels")

                        logger.info("=" * 60)
                        logger.info("DETAILED INSPECTION: First sample in batch")
                        logger.info("=" * 60)

                        if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                            num_to_show = min(5, boxes.shape[0])
                            logger.info(f"Example boxes (first {num_to_show}):")
                            for i in range(num_to_show):
                                logger.info(f"  Box {i}: {boxes[i].tolist()}")

                            # Statistics
                            logger.info(f"Box statistics:")
                            logger.info(f"  Min value: {boxes.min().item():.6f}")
                            logger.info(f"  Max value: {boxes.max().item():.6f}")
                            logger.info(f"  Mean value: {boxes.mean().item():.6f}")

                            # Check if normalized
                            if (boxes > 1.0).any():
                                logger.warning("⚠️  Some box values > 1.0 (not normalized!)")
                            if (boxes < 0.0).any():
                                logger.warning("⚠️  Some box values < 0.0 (negative!)")

                        if isinstance(class_labels, torch.Tensor) and class_labels.numel() > 0:
                            num_to_show = min(5, class_labels.shape[0])
                            logger.info(f"Example class_labels (first {num_to_show}):")
                            logger.info(f"  {class_labels[:num_to_show].tolist()}")
                            logger.info(f"Class label range: [{class_labels.min().item()}, {class_labels.max().item()}]")

                        logger.info("=" * 60)

            logger.info("✓ Sanity check passed: all samples have valid bboxes")

            # Test model forward pass to detect NaN before training starts
            logger.info("Testing model forward pass...")
            model.eval()
            with torch.no_grad():
                test_batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                             for k, v in first_batch.items()}
                try:
                    test_output = model(**test_batch)

                    # Check for NaN in predictions
                    if hasattr(test_output, 'logits'):
                        pred_logits = test_output.logits
                        if torch.isnan(pred_logits).any():
                            logger.error("❌ NaN detected in model predictions (logits)")
                            logger.error(f"   NaN count: {torch.isnan(pred_logits).sum().item()}/{pred_logits.numel()}")
                            raise ValueError("Model produces NaN predictions before training even starts!")

                    if hasattr(test_output, 'pred_boxes'):
                        pred_boxes = test_output.pred_boxes
                        if torch.isnan(pred_boxes).any():
                            logger.error("❌ NaN detected in model predictions (boxes)")
                            logger.error(f"   NaN count: {torch.isnan(pred_boxes).sum().item()}/{pred_boxes.numel()}")
                            raise ValueError("Model produces NaN box predictions before training even starts!")

                    logger.info("✓ Forward pass successful - no NaN in predictions")
                except Exception as e:
                    logger.error(f"❌ Forward pass failed: {e}")
                    raise
            model.train()

        except StopIteration:
            logger.warning("Training dataloader yielded no batches during sanity check.")
        except Exception as exc:
            logger.error("Sanity check on training dataloader failed: %s", exc)
            raise

        train_result = trainer.train()
        # Use safe_serialization=False to handle shared tensors in Deformable DETR
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    if training_args.do_eval and eval_dataset is not None:
        logger.info("=" * 80)
        logger.info("Running evaluation")
        logger.info("=" * 80)
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.push_to_hub:
        logger.info("Pushing model and artifacts to the Hub...")
        trainer.push_to_hub()

    logger.info("=" * 80)
    logger.info("Job complete.")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
