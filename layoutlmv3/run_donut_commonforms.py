#!/usr/bin/env python
# coding=utf-8
"""
Fine-tune Donut on CommonForms for document understanding

Donut (Document Understanding Transformer):
- OCR-free end-to-end model
- Vision encoder (Swin) + text decoder (BART)
- Outputs structured JSON from document images

For CommonForms, we train Donut to generate JSON with detected objects:
{
  "objects": [
    {"category": 0, "bbox": [x, y, w, h]},
    {"category": 1, "bbox": [x, y, w, h]},
    ...
  ]
}

Note: Donut is better suited for semantic extraction (dates, names, amounts)
      For pure object detection, DocLayout-YOLO is recommended.

Based on: https://huggingface.co/docs/transformers/en/model_doc/donut
Tutorial: https://www.philschmid.de/fine-tuning-donut
"""

import logging
import os
import sys
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import torch
import numpy as np
from datasets import load_dataset
from PIL import Image

from transformers import (
    VisionEncoderDecoderModel,
    DonutProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    HfArgumentParser,
    set_seed,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Model arguments"""
    model_name_or_path: str = field(
        default="naver-clova-ix/donut-base",
        metadata={"help": "Donut model to fine-tune"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Cache directory"}
    )


@dataclass
class DataArguments:
    """Data arguments"""
    dataset_name: str = field(
        default="jbarrow/CommonForms",
        metadata={"help": "Dataset name"}
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset config"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max training samples"}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max eval samples"}
    )
    image_size: list = field(
        default_factory=lambda: [1280, 960],
        metadata={"help": "Image size [height, width]"}
    )
    max_length: int = field(
        default=768,
        metadata={"help": "Max sequence length for decoder"}
    )


# Special tokens
task_start_token = "<s_commonforms>"
eos_token = "</s>"
new_special_tokens = [task_start_token, eos_token, "<s_object>", "</s_object>", 
                      "<s_category>", "</s_category>", "<s_bbox>", "</s_bbox>"]


def objects_to_json_sequence(objects):
    """
    Convert CommonForms objects to Donut-style JSON sequence.
    
    Input:
        objects: {
            "bbox": [[x, y, w, h], ...],
            "category_id": [0, 1, ...],
        }
    
    Output:
        "<s_commonforms><s_object><s_category>0</s_category><s_bbox>0.1,0.2,0.3,0.4</s_bbox></s_object>...</s>"
    """
    sequence = task_start_token
    
    for i in range(len(objects["bbox"])):
        bbox = objects["bbox"][i]
        category = objects["category_id"][i]
        
        # Normalize bbox to 0-1 (we'll get image size from the image)
        # For now, keep as-is, will normalize during preprocessing
        sequence += f"<s_object><s_category>{category}</s_category><s_bbox>{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}</s_bbox></s_object>"
    
    sequence += eos_token
    return sequence


def normalize_bbox_in_sequence(sequence, img_width, img_height):
    """Normalize pixel coordinates to 0-1 range in the sequence"""
    import re
    
    def normalize_bbox_match(match):
        coords = match.group(1).split(',')
        x, y, w, h = map(float, coords)
        
        # Clamp to image boundaries
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        if x + w > img_width:
            w = img_width - x
        if y + h > img_height:
            h = img_height - y
        
        # Normalize to 0-1
        x_norm = x / img_width
        y_norm = y / img_height
        w_norm = w / img_width
        h_norm = h / img_height
        
        return f"<s_bbox>{x_norm:.4f},{y_norm:.4f},{w_norm:.4f},{h_norm:.4f}</s_bbox>"
    
    # Replace all bbox values with normalized versions
    return re.sub(r'<s_bbox>([\d.,-]+)</s_bbox>', normalize_bbox_match, sequence)


def preprocess_for_donut(sample):
    """Convert CommonForms sample to Donut format"""
    image = sample["image"]
    
    # Convert to RGB
    if isinstance(image, Image.Image):
        image = image.convert('RGB')
    else:
        image = Image.fromarray(image).convert('RGB')
    
    # Get image dimensions
    img_width, img_height = image.size
    
    # Create sequence
    sequence = objects_to_json_sequence(sample["objects"])
    
    # Normalize bboxes in the sequence
    sequence = normalize_bbox_in_sequence(sequence, img_width, img_height)
    
    return {"image": image, "text": sequence}


def transform_and_tokenize(sample, processor, max_length=768):
    """Transform to model inputs"""
    try:
        # Process image
        pixel_values = processor(
            sample["image"],
            random_padding=False,  # No padding for detection
            return_tensors="pt"
        ).pixel_values.squeeze()
    except Exception as e:
        logger.warning(f"Failed to process image: {e}")
        return {}
    
    # Tokenize text
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)
    
    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    return {
        "pixel_values": pixel_values,
        "labels": labels,
        "target_sequence": sample["text"]
    }


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments))
    
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
    logger.info("Donut Fine-tuning on CommonForms")
    logger.info("="*80)
    logger.info(f"Model: {model_args.model_name_or_path}")
    logger.info(f"Dataset: {data_args.dataset_name}")
    logger.info(f"Output: {training_args.output_dir}")
    logger.info(f"Image size: {data_args.image_size}")
    logger.info("="*80)
    
    set_seed(training_args.seed)
    
    # Load dataset
    logger.info("Loading CommonForms dataset...")
    dataset = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config,
        cache_dir=model_args.cache_dir,
    )
    
    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation", dataset.get("test"))
    
    # Limit samples
    if data_args.max_train_samples:
        train_dataset = train_dataset.select(range(min(data_args.max_train_samples, len(train_dataset))))
    if eval_dataset and data_args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(data_args.max_eval_samples, len(eval_dataset))))
    
    # Preprocess to Donut format
    logger.info("Converting to Donut format...")
    train_dataset = train_dataset.map(preprocess_for_donut, remove_columns=train_dataset.column_names)
    if eval_dataset:
        eval_dataset = eval_dataset.map(preprocess_for_donut, remove_columns=eval_dataset.column_names)
    
    logger.info(f"Sample text: {train_dataset[0]['text'][:200]}...")
    
    # Load processor
    logger.info("Loading processor...")
    processor = DonutProcessor.from_pretrained(model_args.model_name_or_path)
    
    # Add special tokens
    processor.tokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
    
    # Adjust image size
    processor.feature_extractor.size = data_args.image_size  # [height, width]
    processor.feature_extractor.do_align_long_axis = False
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    
    tokenize_fn = lambda x: transform_and_tokenize(x, processor, data_args.max_length)
    
    train_dataset = train_dataset.map(
        tokenize_fn,
        remove_columns=["image", "text"],
        num_proc=4
    )
    
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_fn,
            remove_columns=["image", "text"],
            num_proc=4
        )
    
    # Load model
    logger.info("Loading model...")
    model = VisionEncoderDecoderModel.from_pretrained(model_args.model_name_or_path)
    
    # Resize embeddings for new tokens
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    
    # Configure model
    model.config.encoder.image_size = data_args.image_size
    model.config.decoder.max_length = data_args.max_length
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids([task_start_token])[0]
    
    logger.info(f"Model configured:")
    logger.info(f"  Image size: {model.config.encoder.image_size}")
    logger.info(f"  Max length: {model.config.decoder.max_length}")
    logger.info(f"  Vocabulary size: {len(processor.tokenizer)}")
    
    # Train
    logger.info("Starting training...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,  # For image processor saving
    )
    
    train_result = trainer.train()
    trainer.save_model()
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    
    # Save processor
    processor.save_pretrained(training_args.output_dir)
    
    # Push to hub
    if training_args.push_to_hub:
        logger.info("Pushing to Hub...")
        trainer.push_to_hub()
    
    logger.info("="*80)
    logger.info(f"Training complete! Model saved to: {training_args.output_dir}")
    logger.info("="*80)


if __name__ == "__main__":
    main()

