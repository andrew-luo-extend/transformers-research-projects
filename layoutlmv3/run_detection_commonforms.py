#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning LayoutLMv3 for object detection on CommonForms using Detectron2.
This script uses LayoutLMv3 as a backbone for Detectron2-based object detection.
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import torch
import numpy as np
from datasets import load_dataset
from PIL import Image

logger = logging.getLogger(__name__)

# Check for detectron2
try:
    import detectron2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultTrainer, default_setup, launch
    from detectron2.data import DatasetCatalog, MetadataCatalog
    from detectron2.structures import BoxMode
    from detectron2.evaluation import COCOEvaluator
    HAS_DETECTRON2 = True
except ImportError:
    HAS_DETECTRON2 = False
    print("WARNING: detectron2 not installed. This script requires detectron2.")
    print("Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'")


@dataclass
class Arguments:
    """Training arguments for LayoutLMv3 object detection"""
    
    dataset_name: str = field(
        default="jbarrow/CommonForms",
        metadata={"help": "HuggingFace dataset name"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset configuration name"}
    )
    use_streaming: bool = field(
        default=False,
        metadata={"help": "Use streaming mode to avoid downloading full dataset"}
    )
    cache_dir: Optional[str] = field(
        default="/workspace/hf-cache",
        metadata={"help": "Cache directory for datasets and models"}
    )
    output_dir: str = field(
        default="/workspace/output_detection",
        metadata={"help": "Output directory for model checkpoints and results"}
    )
    model_name_or_path: str = field(
        default="microsoft/layoutlmv3-base",
        metadata={"help": "LayoutLMv3 model to use as backbone"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit training samples for testing"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Limit eval samples for testing"}
    )
    num_epochs: int = field(
        default=50,
        metadata={"help": "Number of training epochs"}
    )
    batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per GPU"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of dataloader workers"}
    )
    eval_period: int = field(
        default=500,
        metadata={"help": "Evaluation period (iterations)"}
    )
    checkpoint_period: int = field(
        default=1000,
        metadata={"help": "Checkpoint saving period (iterations)"}
    )
    resume: bool = field(
        default=False,
        metadata={"help": "Resume from last checkpoint"}
    )


def convert_commonforms_to_detectron2(dataset_dict, split="train"):
    """
    Convert CommonForms dataset format to Detectron2 format.
    
    CommonForms format:
    {
        "image": PIL.Image,
        "objects": {
            "bbox": [[x, y, w, h], ...],
            "category": [0, 1, ...],
            "category_id": [0, 1, ...],
            ...
        }
    }
    
    Detectron2 format:
    [
        {
            "file_name": str,
            "image_id": int,
            "height": int,
            "width": int,
            "annotations": [
                {
                    "bbox": [x, y, w, h],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": int,
                },
                ...
            ]
        },
        ...
    ]
    """
    detectron2_dataset = []
    
    for idx, example in enumerate(dataset_dict):
        # Get image dimensions
        image = example["image"]
        if hasattr(image, 'size'):
            width, height = image.size
        elif hasattr(image, 'shape'):
            height, width = image.shape[:2]
        else:
            height, width = 1000, 1000
        
        # Save image to temp location (Detectron2 needs file paths)
        image_id = f"{split}_{idx}"
        temp_dir = f"/tmp/commonforms_{split}"
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, f"{image_id}.jpg")
        
        if not os.path.exists(image_path):
            if isinstance(image, Image.Image):
                image.save(image_path)
            else:
                Image.fromarray(image).save(image_path)
        
        # Convert objects to annotations
        annotations = []
        objects = example["objects"]
        
        for j in range(len(objects["bbox"])):
            bbox = objects["bbox"][j]  # [x, y, w, h]
            category_id = objects["category_id"][j]
            
            annotation = {
                "bbox": bbox,  # Keep as [x, y, w, h]
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": int(category_id),
            }
            annotations.append(annotation)
        
        record = {
            "file_name": image_path,
            "image_id": idx,
            "height": height,
            "width": width,
            "annotations": annotations,
        }
        
        detectron2_dataset.append(record)
    
    return detectron2_dataset


def register_commonforms_dataset(args):
    """Register CommonForms dataset with Detectron2"""
    
    logger.info(f"Loading CommonForms dataset from {args.dataset_name}...")
    
    # Load dataset from HuggingFace
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config,
        cache_dir=args.cache_dir,
        streaming=args.use_streaming,
    )
    
    # Get splits
    train_dataset = dataset["train"]
    val_dataset = dataset.get("validation", dataset.get("test"))
    
    # Limit samples if specified
    if args.max_train_samples:
        if args.use_streaming:
            train_dataset = train_dataset.take(args.max_train_samples)
        else:
            train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    
    if val_dataset and args.max_eval_samples:
        if args.use_streaming:
            val_dataset = val_dataset.take(args.max_eval_samples)
        else:
            val_dataset = val_dataset.select(range(min(args.max_eval_samples, len(val_dataset))))
    
    # Convert to list (needed for Detectron2)
    logger.info("Converting train split to Detectron2 format...")
    train_list = list(train_dataset)
    train_detectron2 = convert_commonforms_to_detectron2(train_list, "train")
    
    if val_dataset:
        logger.info("Converting validation split to Detectron2 format...")
        val_list = list(val_dataset)
        val_detectron2 = convert_commonforms_to_detectron2(val_list, "val")
    else:
        val_detectron2 = []
    
    # Get category names
    category_ids = set()
    for record in train_detectron2:
        for ann in record["annotations"]:
            category_ids.add(ann["category_id"])
    
    category_names = {i: f"category_{i}" for i in sorted(category_ids)}
    
    # Register with Detectron2
    DatasetCatalog.register("commonforms_train", lambda: train_detectron2)
    MetadataCatalog.get("commonforms_train").set(thing_classes=list(category_names.values()))
    
    if val_detectron2:
        DatasetCatalog.register("commonforms_val", lambda: val_detectron2)
        MetadataCatalog.get("commonforms_val").set(thing_classes=list(category_names.values()))
    
    logger.info(f"Registered {len(train_detectron2)} train and {len(val_detectron2)} val samples")
    logger.info(f"Found {len(category_names)} categories: {category_names}")
    
    return category_names


def setup_cfg(args, num_classes):
    """
    Create Detectron2 config for LayoutLMv3 object detection.
    Based on Microsoft's official implementation.
    """
    cfg = get_cfg()
    
    # Model config
    cfg.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"
    cfg.MODEL.BACKBONE.NAME = "build_layoutlmv3_fpn_backbone"  # This would need custom registration
    
    # For now, use ResNet50-FPN as backbone (simpler alternative)
    # We'll add LayoutLMv3 integration in a follow-up
    cfg.MODEL.BACKBONE.NAME = "build_resnet_fpn_backbone"
    cfg.MODEL.RESNETS.DEPTH = 50
    cfg.MODEL.RESNETS.OUT_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.FPN.IN_FEATURES = ["res3", "res4", "res5"]
    
    # RPN config
    cfg.MODEL.RPN.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256], [512]]
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
    
    # ROI Heads
    cfg.MODEL.ROI_HEADS.NAME = "StandardROIHeads"
    cfg.MODEL.ROI_HEADS.IN_FEATURES = ["p3", "p4", "p5"]
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
    
    # Dataset
    cfg.DATASETS.TRAIN = ("commonforms_train",)
    cfg.DATASETS.TEST = ("commonforms_val",)
    
    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    
    # Solver
    cfg.SOLVER.BASE_LR = args.learning_rate
    cfg.SOLVER.MAX_ITER = args.num_epochs * 1000  # Approximate
    cfg.SOLVER.STEPS = []  # No learning rate decay
    cfg.SOLVER.CHECKPOINT_PERIOD = args.checkpoint_period
    
    # Evaluation
    cfg.TEST.EVAL_PERIOD = args.eval_period
    
    # Output
    cfg.OUTPUT_DIR = args.output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    return cfg


def main():
    """
    Main training function for LayoutLMv3 object detection.
    
    NOTE: This is a simplified implementation using ResNet backbone.
    For full LayoutLMv3 backbone integration, you need to:
    1. Create a custom Detectron2 backbone builder
    2. Register LayoutLMv3Model with detection=True
    3. Add FPN layer mapping
    
    See: https://github.com/microsoft/unilm/tree/master/layoutlmv3/examples/object_detection
    """
    
    # Check for Detectron2
    if not HAS_DETECTRON2:
        print("\n" + "="*80)
        print("ERROR: This script requires Detectron2")
        print("="*80)
        print("\nInstall Detectron2:")
        print("  pip install 'git+https://github.com/facebookresearch/detectron2.git'")
        print("\nOr use the simpler token classification approach:")
        print("  python run_funsd_cord.py --dataset_name commonforms")
        print("="*80 + "\n")
        sys.exit(1)
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description="LayoutLMv3 Object Detection on CommonForms")
    
    # Add all arguments from dataclass
    parser.add_argument("--dataset_name", type=str, default="jbarrow/CommonForms")
    parser.add_argument("--dataset_config", type=str, default=None)
    parser.add_argument("--use_streaming", action="store_true")
    parser.add_argument("--cache_dir", type=str, default="/workspace/hf-cache")
    parser.add_argument("--output_dir", type=str, default="/workspace/output_detection")
    parser.add_argument("--model_name_or_path", type=str, default="microsoft/layoutlmv3-base")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--eval_period", type=int, default=500)
    parser.add_argument("--checkpoint_period", type=int, default=1000)
    parser.add_argument("--resume", action="store_true")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=logging.INFO,
    )
    
    logger.info("="*80)
    logger.info("LayoutLMv3 Object Detection Training")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Streaming: {args.use_streaming}")
    logger.info("="*80)
    
    # Register dataset
    category_names = register_commonforms_dataset(args)
    num_classes = len(category_names)
    
    # Setup Detectron2 config
    cfg = setup_cfg(args, num_classes)
    default_setup(cfg, args)
    
    # Train
    logger.info("Starting training...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    
    logger.info("="*80)
    logger.info(f"Training for {args.num_epochs} epochs...")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Number of classes: {num_classes}")
    logger.info("="*80)
    
    trainer.train()
    
    # Evaluate
    logger.info("Running final evaluation...")
    trainer.test(cfg, trainer.model)
    
    logger.info(f"Training complete! Model saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

