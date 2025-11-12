#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning DETR (Detection Transformer) for object detection on CommonForms.
This is simpler and more appropriate than LayoutLMv3 for object detection.

DETR outputs: bounding boxes + class labels (true object detection)
LayoutLMv3: Takes bboxes as input, outputs labels only (token classification)

For detecting empty form fields, DETR is the better choice.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List, Dict
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
        default="facebook/detr-resnet-50",
        metadata={"help": "DETR model to use. Options: facebook/detr-resnet-50, facebook/detr-resnet-101"}
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
        default=800,
        metadata={"help": "Input image size"}
    )


def convert_commonforms_to_coco_format(examples):
    """
    Convert CommonForms format to COCO detection format for DETR.
    
    DETR expects:
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
        
        # Convert bboxes to COCO format with validation
        coco_bboxes = []
        valid_categories = []
        valid_areas = []
        
        for idx, bbox in enumerate(objects["bbox"]):
            # bbox is [x, y, w, h]
            x, y, w, h = bbox
            
            # Convert to float
            x, y, w, h = float(x), float(y), float(w), float(h)
            
            # Skip invalid boxes (negative dims, NaN, inf)
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                continue
            if not (0 <= x < img_width and 0 <= y < img_height):
                continue
                
            # Clip to image bounds
            x = max(0.0, min(x, img_width - 1.0))
            y = max(0.0, min(y, img_height - 1.0))
            w = max(1.0, min(w, img_width - x))
            h = max(1.0, min(h, img_height - y))
            
            # Final validation
            if w > 0 and h > 0:
                coco_bboxes.append([x, y, w, h])
                valid_categories.append(int(objects["category_id"][idx]))
                valid_areas.append(w * h)
        
        # Skip images with no valid boxes
        if len(coco_bboxes) == 0:
            # Add a dummy box to avoid empty annotations
            coco_bboxes = [[0.0, 0.0, 1.0, 1.0]]
            valid_categories = [0]
            valid_areas = [1.0]
        
        # Create COCO-style annotations
        annotations = {
            'image_id': [i] * len(coco_bboxes),
            'category_id': valid_categories,
            'area': valid_areas,
            'bbox': coco_bboxes,
            'iscrowd': [0] * len(coco_bboxes),
        }
        
        annotations_list.append(annotations)
    
    return {
        'image': images,
        'image_id': list(range(batch_size)),
        'annotations': annotations_list,
    }


def collate_fn(batch):
    """Custom collate function for DETR"""
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
    logger.info("DETR Object Detection Training on CommonForms")
    logger.info("="*80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name}")
    logger.info(f"Output: {training_args.output_dir}")
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
    
    # Limit samples
    if data_args.max_train_samples:
        if data_args.use_streaming:
            train_dataset = train_dataset.take(data_args.max_train_samples)
        else:
            train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))
    
    if eval_dataset and data_args.max_eval_samples:
        if data_args.use_streaming:
            eval_dataset = eval_dataset.take(data_args.max_eval_samples)
        else:
            eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))
    
    # Extract category information
    logger.info("Extracting category information...")
    
    # Try to get category names from dataset features (like AutoTrain)
    try:
        # CommonForms has category names in the dataset schema
        categories = train_dataset.features["objects"].feature["category"].names
        id2label = dict(enumerate(categories))
        label2id = {v: k for k, v in id2label.items()}
        num_labels = len(categories)
        logger.info(f"Found {num_labels} categories from dataset schema: {categories}")
    except (AttributeError, KeyError):
        # Fallback: scan data for unique category IDs
        logger.info("Dataset schema doesn't have category names, scanning data...")
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
        logger.info(f"Found {num_labels} unique category IDs: {category_list}")
    
    logger.info(f"Label mapping: {id2label}")
    
    # Load model and processor
    logger.info(f"Loading model: {model_args.model_name_or_path}")
    
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
    
    # Preprocessing function
    def preprocess_batch(examples):
        """Preprocess batch for DETR"""
        # Convert CommonForms to COCO format
        coco_data = convert_commonforms_to_coco_format(examples)
        
        # Process with image processor
        images = coco_data['image']
        annotations_list = coco_data['annotations']
        
        # DETR image processor expects annotations as list of dicts with 'image_id' and 'annotations' keys
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
    
    # Determine number of workers (avoid deadlocks with small datasets)
    num_workers = 1 if data_args.max_train_samples and data_args.max_train_samples < 100 else 4
    logger.info(f"Using {num_workers} worker(s) for preprocessing")
    
    if data_args.use_streaming:
        # For streaming, we can't use map with num_proc
        logger.info("Using streaming mode - preprocessing on-the-fly")
        train_dataset = train_dataset.map(preprocess_batch, batched=True, remove_columns=train_dataset.column_names)
        if eval_dataset:
            eval_dataset = eval_dataset.map(preprocess_batch, batched=True, remove_columns=eval_dataset.column_names)
    else:
        logger.info(f"Preprocessing training dataset...")
        train_dataset = train_dataset.map(
            preprocess_batch,
            batched=True,
            remove_columns=train_dataset.column_names,
            num_proc=num_workers,
            desc="Preprocessing training data",
            load_from_cache_file=True,
        )
        if eval_dataset:
            logger.info(f"Preprocessing evaluation dataset...")
            eval_dataset = eval_dataset.map(
                preprocess_batch,
                batched=True,
                remove_columns=eval_dataset.column_names,
                num_proc=num_workers,
                desc="Preprocessing evaluation data",
                load_from_cache_file=True,
            )
    
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
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        trainer.save_model()
        
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    # Evaluate
    if training_args.do_eval:
        logger.info("Evaluating...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Save final model
    if training_args.push_to_hub:
        logger.info("Pushing model to Hub...")
        trainer.push_to_hub()
    
    logger.info("="*80)
    logger.info("Training complete!")
    logger.info(f"Model saved to: {training_args.output_dir}")
    if training_args.push_to_hub:
        logger.info(f"Model uploaded to HuggingFace Hub")
    logger.info("="*80)


if __name__ == "__main__":
    main()

