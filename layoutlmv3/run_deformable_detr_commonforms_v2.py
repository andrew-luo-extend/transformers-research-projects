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

Quick start (single GPU, RunPod defaults):

```
python run_deformable_detr_commonforms_v2.py \
  --do_train --do_eval \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 30 \
  --learning_rate 5e-5 \
  --fp16 \
  --use_runpod_defaults
```

For multi-GPU:

```
accelerate launch --mixed_precision=fp16 run_deformable_detr_commonforms_v2.py \
  --do_train --do_eval --use_runpod_defaults --per_device_train_batch_size 6
```
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
import torch
from PIL import Image
from datasets import Dataset, IterableDataset, load_dataset

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
        default=1000,
        metadata={"help": "Target longest edge (pixels) after augmentation."},
    )
    filter_empty_annotations: bool = field(
        default=False,
        metadata={
            "help": "Filter out samples with no bounding boxes. Set to True to remove negative samples. "
            "When False, samples that cause errors will be caught and skipped during training."
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
    debug_logged = False  # Only log detailed debug info for first sample

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
            if not np.isfinite([x, y, w, h]).all():
                continue
            if w <= 0 or h <= 0:
                continue
            x = float(np.clip(x, 0.0, max(img_w - 1.0, 1.0)))
            y = float(np.clip(y, 0.0, max(img_h - 1.0, 1.0)))
            w = float(np.clip(w, 1.0, max(img_w - x, 1.0)))
            h = float(np.clip(h, 1.0, max(img_h - y, 1.0)))
            if w <= 0 or h <= 0:
                continue
            transformed_bboxes.append([x, y, w, h])
            transformed_categories.append(int(category_id))

        if not transformed_bboxes or not transformed_categories:
            logger.debug("Skipping sample %s after augmentation: no valid boxes.", image_id)
            skip_reasons["no_valid_boxes_after_aug"] = skip_reasons.get("no_valid_boxes_after_aug", 0) + 1
            continue
        if not all(np.isfinite(bbox).all() for bbox in transformed_bboxes):
            logger.debug("Skipping sample %s after augmentation: non-finite bbox detected.", image_id)
            skip_reasons["non_finite_bbox"] = skip_reasons.get("non_finite_bbox", 0) + 1
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

            # Keep samples with no boxes but mark them - they'll be filtered at collation time
            if boxes.numel() == 0:
                label["boxes"] = boxes
                label["class_labels"] = class_labels
                cleaned_labels.append(label)
                continue

            finite_mask = torch.isfinite(boxes).all(dim=1)
            if finite_mask.any():
                boxes = boxes[finite_mask]
                class_labels = class_labels[finite_mask]
            else:
                logger.debug("Replacing non-finite boxes with default placeholder.")
                boxes = torch.tensor([[0.5, 0.5, 1e-3, 1e-3]], dtype=torch.float32)
                class_labels = torch.zeros((1,), dtype=torch.int64)

            boxes = boxes.clamp(0.0, 1.0)
            if boxes.shape[-1] >= 4:
                boxes[:, 2:] = boxes[:, 2:].clamp(1e-4, 1.0 - 1e-4)

            label["boxes"] = boxes
            label["class_labels"] = class_labels
            cleaned_labels.append(label)

        encoded["labels"] = cleaned_labels

    return encoded


def collate_fn(features: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collation for DETR-style object detection models.

    Filters out samples with empty annotations at collation time to avoid
    Hungarian matcher issues.
    """
    # Filter out features with empty labels
    valid_features = []
    skipped_count = 0

    for feature in features:
        label = feature.get("labels")
        has_annotations = False

        if isinstance(label, dict):
            boxes = label.get("boxes")
            if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
                has_annotations = True
        elif isinstance(label, list) and len(label) > 0:
            has_annotations = True

        if has_annotations:
            valid_features.append(feature)
        else:
            skipped_count += 1

    # If we skipped any, log it
    if skipped_count > 0:
        logger.debug(f"Filtered out {skipped_count} samples with empty annotations from batch")

    # If no valid features remain, this shouldn't happen often but just in case,
    # we'll raise an exception which will cause the batch to be skipped
    if not valid_features:
        raise ValueError("Batch contains only samples with empty annotations - skipping")

    pixel_values: List[torch.Tensor] = []
    pixel_masks: List[torch.Tensor] = []
    labels: List[Any] = []

    for feature in valid_features:
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


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, RunPodArguments, TrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, runpod_args, training_args = parser.parse_json_file(os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, runpod_args, training_args = parser.parse_args_into_dataclasses()

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

    # Optionally filter out empty annotations
    # When disabled, the collator will filter them out at batch time
    if data_args.filter_empty_annotations:
        if train_dataset is not None and not data_args.use_streaming:
            logger.info("Filtering training dataset to remove samples without annotations...")
            original_size = len(train_dataset)
            train_dataset = train_dataset.filter(filter_empty_annotations)
            filtered_size = len(train_dataset)
            logger.info(f"Filtered {original_size - filtered_size} empty samples from training set ({filtered_size} remaining)")

            if filtered_size == 0:
                raise ValueError("After filtering empty samples, no training samples remain. Your dataset may only contain empty annotations.")

        if eval_dataset is not None and not data_args.use_streaming:
            logger.info("Filtering evaluation dataset to remove samples without annotations...")
            original_size = len(eval_dataset)
            eval_dataset = eval_dataset.filter(filter_empty_annotations)
            filtered_size = len(eval_dataset)
            logger.info(f"Filtered {original_size - filtered_size} empty samples from evaluation set ({filtered_size} remaining)")
    else:
        logger.info("Keeping all samples (including those with no annotations)")

    if training_args.do_train and data_args.max_train_samples:
        train_dataset = limit_dataset(train_dataset, data_args.max_train_samples, data_args.use_streaming)

    if training_args.do_eval and data_args.max_eval_samples:
        eval_dataset = limit_dataset(eval_dataset, data_args.max_eval_samples, data_args.use_streaming)

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
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.revision,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    logger.info("Model loaded with %d detection classes.", model.config.num_labels)

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
        try:
            first_batch = next(iter(trainer.get_train_dataloader()))
            pv = first_batch.get("pixel_values")
            if isinstance(pv, torch.Tensor) and torch.isnan(pv).any():
                raise ValueError("NaNs detected in first training batch pixel_values.")
        except StopIteration:
            logger.warning("Training dataloader yielded no batches during sanity check.")
        except Exception as exc:
            logger.error("Sanity check on training dataloader failed: %s", exc)
            raise

        train_result = trainer.train()
        # Use safe_serialization=False to handle shared tensors in Deformable DETR
        trainer.save_model(safe_serialization=False)
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
