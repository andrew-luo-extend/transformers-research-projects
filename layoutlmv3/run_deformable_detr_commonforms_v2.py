#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning Deformable DETR for object detection on CommonForms.
Based on the HuggingFace Fashionpedia fine-tuning example.

Model: https://huggingface.co/Aryn/deformable-detr-DocLayNet
Dataset: https://huggingface.co/datasets/jbarrow/CommonForms
"""

import argparse
import logging
import numpy as np
import torch
from PIL import Image
import albumentations as A
from typing import Any

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    
    # Model arguments
    parser.add_argument("--model_name_or_path", type=str, default="Aryn/deformable-detr-DocLayNet")
    parser.add_argument("--cache_dir", type=str, default=None)
    
    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, default="jbarrow/CommonForms")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=1000)
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./deformable-detr-commonforms")
    parser.add_argument("--num_train_epochs", type=int, default=30)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_eval", action="store_true")
    
    return parser.parse_args()


def formatted_anns(image_id, category_ids, bboxes):
    """Format annotations for DETR"""
    annotations = []
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


def create_transforms(image_size, is_train=True):
    """Create Albumentations transforms"""
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
                format="coco",  # [x, y, width, height]
                label_fields=["category_id"],
            ),
        )
    else:
        return A.Compose(
            [
                A.LongestMaxSize(image_size),
                A.PadIfNeeded(image_size, image_size, border_mode=0, value=(0, 0, 0)),
            ],
            bbox_params=A.BboxParams(
                format="coco",
                label_fields=["category_id"],
            ),
        )


def _to_plain_value(value: Any):
    """Recursively convert BatchFeature/maps to plain Python containers."""
    if isinstance(value, list):
        return [_to_plain_value(v) for v in value]
    if hasattr(value, "items"):
        return {k: _to_plain_value(v) for k, v in value.items()}
    return value


def transform_aug_ann(examples, transform, image_processor, category_key, category_id_remap):
    """Apply augmentation and format for DETR"""
    # The datasets library sometimes feeds single examples and sometimes batches.
    # Normalize to lists so downstream code does not have to branch.
    if isinstance(examples["image"], Image.Image):
        image_ids = [examples["id"]]
        images = [examples["image"]]
        objects_list = [examples["objects"]]
    else:
        image_ids = examples["id"]
        images = examples["image"]
        objects_list = examples["objects"]
    
    processed_images = []
    processed_targets = []
    
    for idx, (image_id, image, objects) in enumerate(zip(image_ids, images, objects_list)):
        if isinstance(image, Image.Image):
            image_rgb = image.convert("RGB")
        else:
            image_rgb = Image.fromarray(image).convert("RGB")
        
        image_np = np.array(image_rgb)
        image_bgr = image_np[:, :, ::-1]
        
        raw_bboxes = objects.get("bbox", [])
        raw_categories = objects.get(category_key) or objects.get("category_id") or objects.get("category") or []
        
        # Keep lengths aligned
        if len(raw_bboxes) != len(raw_categories):
            min_len = min(len(raw_bboxes), len(raw_categories))
            raw_bboxes = raw_bboxes[:min_len]
            raw_categories = raw_categories[:min_len]
        
        normalized_categories = []
        for cat in raw_categories:
            if isinstance(cat, np.generic):
                cat = cat.item()
            normalized_categories.append(cat)
        
        mapped_categories = [category_id_remap.get(cat, 0) for cat in normalized_categories]
        
        transform_out = transform(
            image=image_bgr,
            bboxes=raw_bboxes,
            category_id=mapped_categories,
        )
        
        transformed_bboxes = []
        transformed_categories = []
        img_h, img_w = transform_out["image"].shape[:2]
        for bbox, category_id in zip(transform_out["bboxes"], transform_out["category_id"]):
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
        
        if len(transformed_bboxes) == 0:
            transformed_bboxes = [[0.0, 0.0, 1.0, 1.0]]
            default_label = int(mapped_categories[0]) if mapped_categories else 0
            transformed_categories = [default_label]
        
        processed_images.append(transform_out["image"][:, :, ::-1].astype(np.uint8))
        
        if isinstance(image_id, (int, np.integer)):
            target_image_id = int(image_id)
        else:
            try:
                target_image_id = int(str(image_id))
            except (TypeError, ValueError):
                target_image_id = idx
        
        processed_targets.append(
            {"image_id": target_image_id, "annotations": formatted_anns(target_image_id, transformed_categories, transformed_bboxes)}
        )
    
    encoded = image_processor(images=processed_images, annotations=processed_targets, return_tensors="pt")
    encoded = {key: _to_plain_value(value) for key, value in encoded.items()}
    
    # Clean labels to ensure no NaNs or empty tensors propagate to the Trainer
    if "labels" in encoded:
        cleaned_labels = []
        for label in encoded["labels"]:
            if not isinstance(label, dict):
                cleaned_labels.append(label)
                continue
            
            label_copy = dict(label)
            boxes = label_copy.get("boxes")
            class_labels = label_copy.get("class_labels")
            
            if isinstance(boxes, torch.Tensor):
                device = boxes.device
                finite_mask = torch.isfinite(boxes).all(dim=1)
                
                if finite_mask.any():
                    boxes = boxes[finite_mask]
                    if isinstance(class_labels, torch.Tensor):
                        class_labels = class_labels[finite_mask]
                else:
                    boxes = torch.empty(0, 4, dtype=boxes.dtype, device=device)
                    if isinstance(class_labels, torch.Tensor):
                        class_labels = class_labels.new_empty(0)
                
                # Ensure at least one valid annotation remains
                if boxes.numel() == 0:
                    boxes = torch.tensor([[0.0, 0.0, 0.01, 0.01]], dtype=torch.float32, device=device)
                    if isinstance(class_labels, torch.Tensor):
                        class_labels = torch.zeros(1, dtype=torch.int64, device=class_labels.device)
                    else:
                        class_labels = torch.tensor([0], dtype=torch.int64, device=device)
                
                label_copy["boxes"] = boxes
                label_copy["class_labels"] = class_labels
            
            cleaned_labels.append(label_copy)
        
        encoded["labels"] = cleaned_labels
    
    return encoded


def collate_fn(batch):
    """Custom collate function for DETR"""
    pixel_values = []
    for item in batch:
        pv = item["pixel_values"]
        if isinstance(pv, torch.Tensor) and pv.dim() > 3 and pv.shape[0] == 1:
            pv = pv.squeeze(0)
        pixel_values.append(pv)
    pixel_masks = None
    if "pixel_mask" in batch[0]:
        pixel_masks = []
        for item in batch:
            pm = item["pixel_mask"]
            if isinstance(pm, torch.Tensor) and pm.dim() > 3 and pm.shape[0] == 1:
                pm = pm.squeeze(0)
            pixel_masks.append(pm)
    
    batch_dict = {
        "pixel_values": torch.stack(pixel_values),
        "labels": [item["labels"] for item in batch],
    }
    if pixel_masks is not None:
        batch_dict["pixel_mask"] = torch.stack(pixel_masks)
    
    return batch_dict


def main():
    args = get_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    logger.info("="*80)
    logger.info("Deformable DETR Training on CommonForms")
    logger.info("="*80)
    logger.info(f"Model: {args.model_name_or_path}")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Image size: {args.image_size}px")
    logger.info("="*80)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
    )
    
    # Get splits (CommonForms uses "train", "val", "test")
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("val", dataset.get("test"))
    
    # Limit samples
    if args.max_train_samples:
        logger.info(f"Limiting training to {args.max_train_samples} samples")
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    
    if eval_dataset and args.max_eval_samples:
        logger.info(f"Limiting eval to {args.max_eval_samples} samples")
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))
    
    # Determine category field and build label mappings
    logger.info("Extracting categories...")
    objects_feature = train_dataset.features.get("objects")
    category_feature = None
    category_key = None
    
    if objects_feature is not None:
        inner_feature = getattr(objects_feature, "feature", None)
        if inner_feature:
            for candidate in ("category_id", "category", "label", "class"):
                if candidate in inner_feature:
                    category_key = candidate
                    category_feature = inner_feature[candidate]
                    break
    
    if category_key is None:
        sample_example = train_dataset[0]
        if "objects" in sample_example:
            for candidate in ("category_id", "category", "label", "class"):
                if candidate in sample_example["objects"]:
                    category_key = candidate
                    break
    
    if category_key is None:
        raise ValueError("Could not determine category field in dataset. Expected one of ['category_id', 'category', 'label', 'class'].")
    
    logger.info(f"Using '{category_key}' as category field")
    
    # Collect unique category values (limit scanning for speed)
    unique_categories = set()
    max_scan = min(len(train_dataset), 2000)
    for idx in range(max_scan):
        example = train_dataset[idx]
        cats = example.get("objects", {}).get(category_key, [])
        for cat in cats:
            if isinstance(cat, np.generic):
                cat = cat.item()
            unique_categories.add(cat)
        if (
            len(unique_categories) > 0
            and category_feature is not None
            and hasattr(category_feature, "num_classes")
            and len(unique_categories) >= category_feature.num_classes
        ):
            break
    
    if not unique_categories:
        logger.warning("No categories found while scanning data; defaulting to single class.")
        unique_categories = {0}
    
    def sort_key(value):
        return (str(type(value)), str(value))
    
    sorted_unique = sorted(unique_categories, key=sort_key)
    
    # Build mapping from original category value to contiguous IDs
    category_id_remap = {orig: idx for idx, orig in enumerate(sorted_unique)}
    
    id2label = {}
    for new_id, orig_value in enumerate(sorted_unique):
        label_name = None
        if category_feature is not None and hasattr(category_feature, "names"):
            try:
                label_name = category_feature.names[int(orig_value)]
            except (TypeError, ValueError, KeyError, IndexError):
                label_name = None
        if label_name is None:
            label_name = str(orig_value)
        id2label[new_id] = label_name
    
    label2id = {v: k for k, v in id2label.items()}
    
    logger.info(f"Found {len(id2label)} categories: {list(id2label.values())}")
    logger.info(f"Label mapping: {id2label}")
    if len(category_id_remap) <= 20:
        logger.info(f"Category remap (dataset -> model indices): {category_id_remap}")
    else:
        preview_items = list(category_id_remap.items())[:10]
        logger.info(f"Category remap sample (dataset -> model indices): {preview_items} ...")
    
    # Load image processor (same as AutoTrain)
    logger.info(f"Loading image processor from {args.model_name_or_path}")
    image_processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        do_pad=False,  # Padding done in Albumentations
        do_resize=False,  # Resize done in Albumentations
        size={"longest_edge": args.image_size},
    )
    
    # Load model (same as AutoTrain)
    logger.info(f"Loading model from {args.model_name_or_path}")
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        cache_dir=args.cache_dir,
    )
    model_config.num_labels = len(id2label)
    
    try:
        model = AutoModelForObjectDetection.from_pretrained(
            args.model_name_or_path,
            config=model_config,
            cache_dir=args.cache_dir,
            ignore_mismatched_sizes=True,
        )
    except OSError:
        # Fallback for TensorFlow models
        model = AutoModelForObjectDetection.from_pretrained(
            args.model_name_or_path,
            config=model_config,
            cache_dir=args.cache_dir,
            ignore_mismatched_sizes=True,
            from_tf=True,
        )
    
    logger.info(f"Model loaded with {len(id2label)} classes")
    
    # Create transforms
    train_transform = create_transforms(args.image_size, is_train=True)
    eval_transform = create_transforms(args.image_size, is_train=False)
    
    # Apply transforms with datasets
    logger.info("Applying transforms...")
    
    def transform_train(examples):
        return transform_aug_ann(examples, train_transform, image_processor, category_key, category_id_remap)
    
    def transform_eval(examples):
        return transform_aug_ann(examples, eval_transform, image_processor, category_key, category_id_remap)
    
    # Use with_transform for lazy loading (better for large datasets)
    train_dataset_transformed = train_dataset.with_transform(transform_train)
    eval_dataset_transformed = eval_dataset.with_transform(transform_eval) if eval_dataset else None
    
    logger.info("Transforms applied!")
    
    # Training arguments (consistent with AutoTrain)
    training_args_dict = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.num_train_epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "dataloader_num_workers": args.dataloader_num_workers,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "push_to_hub": args.push_to_hub,
        "seed": args.seed,
        "dataloader_pin_memory": True,
        "report_to": "tensorboard",
        "ddp_find_unused_parameters": False,
    }
    
    # Add eval-specific args
    if args.do_eval:
        training_args_dict["eval_steps"] = args.eval_steps
        training_args_dict["eval_strategy"] = "steps"
        training_args_dict["load_best_model_at_end"] = True
        training_args_dict["eval_do_concat_batches"] = False  # Same as AutoTrain
    else:
        training_args_dict["eval_strategy"] = "no"
    
    # Add FP16
    if args.fp16:
        training_args_dict["fp16"] = True
    
    # Add hub model ID if pushing
    if args.push_to_hub and args.hub_model_id:
        training_args_dict["hub_model_id"] = args.hub_model_id
    
    training_args = TrainingArguments(**training_args_dict)
    
    # Initialize trainer
    logger.info("Initializing Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_transformed if args.do_train else None,
        eval_dataset=eval_dataset_transformed if args.do_eval else None,
        data_collator=collate_fn,
        tokenizer=image_processor,  # For saving
    )
    
    # Train
    if args.do_train:
        logger.info("="*80)
        logger.info("Starting training...")
        logger.info("="*80)
        train_result = trainer.train()
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluate
    if args.do_eval:
        logger.info("="*80)
        logger.info("Evaluating...")
        logger.info("="*80)
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Push to hub
    if args.push_to_hub:
        logger.info("="*80)
        logger.info("Pushing to Hub...")
        logger.info("="*80)
        trainer.push_to_hub()
    
    logger.info("="*80)
    logger.info("Done!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
