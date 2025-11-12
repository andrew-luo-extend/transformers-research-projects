#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning Deformable DETR for object detection on CommonForms.
This uses the Aryn/deformable-detr-DocLayNet model which is:
- Pre-trained on DocLayNet (document layouts)
- Uses deformable attention (better for small objects like checkboxes)
- Faster convergence than regular DETR
- Better multi-scale detection

Model: https://huggingface.co/Aryn/deformable-detr-DocLayNet
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch
from PIL import Image

import datasets
from datasets import load_dataset
import evaluate

from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Model arguments"""
    
    model_name_or_path: str = field(
        default="Aryn/deformable-detr-DocLayNet",
        metadata={"help": "Deformable DETR model. Default: Aryn/deformable-detr-DocLayNet (trained on DocLayNet)"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory for models"}
    )


@dataclass
class DataArguments:
    """Data arguments"""
    
    dataset_name: str = field(
        default="jbarrow/CommonForms",
        metadata={"help": "Dataset name on HuggingFace Hub"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration"}
    )
    use_streaming: bool = field(
        default=False,
        metadata={"help": "Use streaming mode for large datasets"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum training samples (for testing)"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Maximum evaluation samples"}
    )
    image_size: int = field(
        default=1000,
        metadata={"help": "Input image size (Deformable DETR works well with 800-1200)"}
    )


def convert_commonforms_to_coco_format(examples):
    """
    Convert CommonForms format to COCO detection format for Deformable DETR.
    
    Deformable DETR expects same format as DETR:
    {
        'image': PIL.Image,
        'image_id': int,
        'annotations': {
            'image_id': List[int],
            'category_id': List[int],
            'area': List[float],
            'bbox': List[List[float]],  # [x, y, w, h] in absolute pixels
            'iscrowd': List[int],
        }
    }
    """
    batch_size = len(examples["image"])
    images = []
    annotations_list = []
    
    for i in range(batch_size):
        image = examples["image"][i]
        images.append(image)
        
        # Get image dimensions
        if hasattr(image, 'size'):
            img_width, img_height = image.size
        elif hasattr(image, 'shape'):
            img_height, img_width = image.shape[:2]
        else:
            img_width, img_height = 1000, 1000
        
        # Extract objects
        objects = examples["objects"][i]
        num_objects = len(objects["bbox"])
        
        # Convert bboxes to COCO format
        coco_bboxes = []
        for bbox in objects["bbox"]:
            # bbox is already [x, y, w, h] - perfect for COCO format
            x, y, w, h = bbox
            # Ensure valid boxes
            x = max(0, min(float(x), img_width - 1))
            y = max(0, min(float(y), img_height - 1))
            w = max(1, min(float(w), img_width - x))
            h = max(1, min(float(h), img_height - y))
            coco_bboxes.append([x, y, w, h])
        
        # Calculate areas
        areas = [bbox[2] * bbox[3] for bbox in coco_bboxes]
        
        # Create COCO-style annotations
        annotations = {
            'image_id': [i] * num_objects,
            'category_id': [int(cat_id) for cat_id in objects["category_id"]],
            'area': areas,
            'bbox': coco_bboxes,
            'iscrowd': [0] * num_objects,  # No crowd annotations
        }
        
        annotations_list.append(annotations)
    
    return {
        'image': images,
        'image_id': list(range(batch_size)),
        'annotations': annotations_list,
    }


def collate_fn(batch):
    """Custom collate function for Deformable DETR"""
    pixel_values = [item['pixel_values'] for item in batch]
    labels = [item['labels'] for item in batch]
    return {
        'pixel_values': torch.stack(pixel_values),
        'labels': labels
    }


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
    logger.info("="*80)
    logger.info("Deformable DETR Object Detection Training on CommonForms")
    logger.info("="*80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"Image size: {data_args.image_size}px")
    logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Epochs: {training_args.num_train_epochs}")
    logger.info(f"Streaming: {data_args.use_streaming}")
    logger.info("="*80)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        cache_dir=model_args.cache_dir,
        streaming=data_args.use_streaming,
    )
    
    # Get splits (CommonForms uses "train", "val", "test")
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("val", dataset.get("test"))
    
    logger.info(f"Train dataset loaded")
    if eval_dataset:
        logger.info(f"Eval dataset loaded")
    
    # Limit samples for testing
    if data_args.max_train_samples:
        logger.info(f"Limiting training to {data_args.max_train_samples} samples")
        if data_args.use_streaming:
            train_dataset = train_dataset.take(data_args.max_train_samples)
        else:
            train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))
    
    if eval_dataset and data_args.max_eval_samples:
        logger.info(f"Limiting evaluation to {data_args.max_eval_samples} samples")
        if data_args.use_streaming:
            eval_dataset = eval_dataset.take(data_args.max_eval_samples)
        else:
            eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))
    
    # Extract category information from dataset
    logger.info("Extracting category information...")
    categories = set()
    sample_count = 0
    max_samples_for_categories = 1000
    
    dataset_for_categories = train_dataset.take(max_samples_for_categories) if data_args.use_streaming else train_dataset
    
    for example in dataset_for_categories:
        if sample_count >= max_samples_for_categories:
            break
        if "objects" in example and "category_id" in example["objects"]:
            categories.update(example["objects"]["category_id"])
        sample_count += 1
    
    category_list = sorted(list(categories))
    num_labels = len(category_list)
    
    # Create label mappings
    id2label = {i: f"category_{cat}" for i, cat in enumerate(category_list)}
    label2id = {v: k for k, v in id2label.items()}
    
    logger.info(f"Found {num_labels} categories: {category_list}")
    logger.info(f"Label mapping: {id2label}")
    
    # Load model and processor
    logger.info(f"Loading Deformable DETR model: {model_args.model_name_or_path}")
    
    image_processor = AutoImageProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        do_resize=True,
        size={"shortest_edge": data_args.image_size, "longest_edge": data_args.image_size},
    )
    
    model = AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    
    logger.info(f"Model loaded with {num_labels} classes")
    logger.info(f"Model architecture: {model.__class__.__name__}")
    
    # Preprocessing function
    def preprocess_batch(examples):
        """Preprocess batch for Deformable DETR"""
        # Convert CommonForms to COCO format
        coco_data = convert_commonforms_to_coco_format(examples)
        
        # Process with image processor
        images = coco_data['image']
        annotations_list = coco_data['annotations']
        
        # Deformable DETR image processor expects annotations as list of dicts with 'image_id' and 'annotations' keys
        # Each annotation in the 'annotations' list should have: category_id, bbox, area, iscrowd
        formatted_annotations = []
        for i, ann in enumerate(annotations_list):
            # Convert to list of individual annotation dicts
            individual_annotations = []
            for j in range(len(ann['category_id'])):
                individual_annotations.append({
                    'category_id': ann['category_id'][j],
                    'bbox': ann['bbox'][j],
                    'area': ann['area'][j],
                    'iscrowd': ann['iscrowd'][j],
                })
            
            formatted_annotations.append({
                'image_id': i,
                'annotations': individual_annotations
            })
        
        # Process images with formatted annotations
        encoding = image_processor(images=images, annotations=formatted_annotations, return_tensors="pt")
        
        return encoding
    
    # Apply preprocessing
    logger.info("Preprocessing datasets...")
    logger.info("This may take a while for large datasets...")
    
    if data_args.use_streaming:
        # For streaming, we can't use map with num_proc
        train_dataset = train_dataset.map(preprocess_batch, batched=True, remove_columns=train_dataset.column_names)
        if eval_dataset:
            eval_dataset = eval_dataset.map(preprocess_batch, batched=True, remove_columns=eval_dataset.column_names)
    else:
        train_dataset = train_dataset.map(
            preprocess_batch,
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=4,
            desc="Preprocessing training data",
        )
        if eval_dataset:
            eval_dataset = eval_dataset.map(
                preprocess_batch,
                batched=True,
                remove_columns=eval_dataset.column_names,
                num_proc=4,
                desc="Preprocessing evaluation data",
            )
    
    logger.info("Preprocessing complete!")
    
    # Log dataset info
    if not data_args.use_streaming:
        logger.info(f"Training samples: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation samples: {len(eval_dataset)}")
    
    # Initialize Trainer
    logger.info("Initializing Trainer...")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        data_collator=collate_fn,
        tokenizer=image_processor,  # Used for saving
    )
    
    # Train
    if training_args.do_train:
        logger.info("="*80)
        logger.info("Starting training...")
        logger.info("="*80)
        
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("Training complete!")
    
    # Evaluate
    if training_args.do_eval:
        logger.info("="*80)
        logger.info("Evaluating...")
        logger.info("="*80)
        
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info("Evaluation complete!")
    
    # Save final model
    if training_args.push_to_hub:
        logger.info("="*80)
        logger.info("Pushing model to HuggingFace Hub...")
        logger.info("="*80)
        
        trainer.push_to_hub()
        logger.info("Model uploaded successfully!")
    
    logger.info("="*80)
    logger.info("All done!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    if training_args.push_to_hub:
        logger.info(f"Model available at: https://huggingface.co/{training_args.hub_model_id}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

