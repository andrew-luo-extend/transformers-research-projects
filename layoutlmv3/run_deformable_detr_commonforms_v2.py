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


def formatted_anns(image_id, category, area, bbox):
    """Format annotations for DETR"""
    annotations = []
    for i in range(len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)
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
                label_fields=["category"]
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
                label_fields=["category"]
            ),
        )


def transform_aug_ann(examples, transform, image_processor):
    """Apply augmentation and format for DETR"""
    image_ids = examples["id"]
    images, bboxes, areas, categories = [], [], [], []
    
    for image, objects in zip(examples["image"], examples["objects"]):
        # Convert PIL to numpy
        image = np.array(image.convert("RGB"))[:, :, ::-1]
        
        # Apply augmentation
        out = transform(
            image=image,
            bboxes=objects["bbox"],
            category=objects["category"]
        )
        
        areas.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
    
    # Format annotations
    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, areas, bboxes)
    ]
    
    # Process with image processor
    return image_processor(images=images, annotations=targets, return_tensors="pt")


def collate_fn(batch):
    """Custom collate function for DETR"""
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = {}
    encoding["pixel_values"] = torch.stack(pixel_values)
    encoding["pixel_mask"] = torch.stack([item["pixel_mask"] for item in batch])
    encoding["labels"] = [item["labels"] for item in batch]
    return encoding


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
    
    # Get category names from dataset schema (same as AutoTrain)
    logger.info("Extracting categories...")
    categories = train_dataset.features["objects"].feature["category"].names
    id2label = dict(enumerate(categories))
    label2id = {v: k for k, v in id2label.items()}
    
    logger.info(f"Label mapping: {id2label}")
    
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
    from transformers import AutoConfig
    
    model_config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        label2id=label2id,
        id2label=id2label,
        cache_dir=args.cache_dir,
    )
    
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
        return transform_aug_ann(examples, train_transform, image_processor)
    
    def transform_eval(examples):
        return transform_aug_ann(examples, eval_transform, image_processor)
    
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

