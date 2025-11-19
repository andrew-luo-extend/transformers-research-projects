#!/usr/bin/env python
# coding=utf-8
"""
Fine-tuning RF-DETR on CommonForms dataset from HuggingFace.

RF-DETR (Real-time Fine-tunable DETR) is faster than regular DETR while maintaining accuracy.

Usage:
    python run_rfdetr_commonforms.py \
        --dataset_name jbarrow/CommonForms \
        --output_dir ./rfdetr-commonforms \
        --epochs 30 \
        --batch_size 8

Dataset: https://huggingface.co/datasets/jbarrow/CommonForms
Model: RF-DETR (https://github.com/saidinesh5/rfdetr or similar)
"""

import argparse
import io
import json
import logging
import os
import sys
import subprocess
import signal
from pathlib import Path
import shutil
import warnings
from typing import List, Tuple, Dict

import torch
from PIL import Image
from datasets import Image as DatasetImage
from datasets import load_dataset
from tqdm.auto import tqdm

# Increase PIL's decompression bomb limit for large document images
Image.MAX_IMAGE_PIXELS = None  # Disable the limit

# Suppress decompression bomb warnings (we're handling large images intentionally)
warnings.filterwarnings('ignore', category=Image.DecompressionBombWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def push_final_model_to_hub(
    checkpoint_path: str,
    hub_model_id: str,
    model_info: Dict,
    hf_token: str = None
) -> bool:
    """Push the final trained model to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi, create_repo

        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("⚠️  HF_TOKEN not found, skipping upload")
                return False

        api = HfApi(token=hf_token)

        # Create model card
        commit_message = "Upload final trained model"
        description_suffix = f"Final model after training"
        model_card = f"""---
tags:
- object-detection
- rf-detr
- commonforms
datasets:
- {model_info.get('dataset', 'jbarrow/CommonForms')}
---

# RF-DETR Fine-tuned on CommonForms - {description_suffix}

This model is an RF-DETR ({model_info.get('model_size', 'medium')}) fine-tuned on the CommonForms dataset for form field detection.

**{description_suffix}**

## Model Details

- **Model Type:** RF-DETR {model_info.get('model_size', 'medium')}
- **Dataset:** {model_info.get('dataset', 'jbarrow/CommonForms')}
- **Classes:** {model_info.get('num_classes', 'N/A')}

## Classes

{json.dumps(model_info.get('categories', []), indent=2)}

## Usage

```python
import torch
from PIL import Image

# Load model
# Note: You'll need the rfdetr library installed
from rfdetr import RFDETR{model_info.get('model_size', 'Medium').capitalize()}

model = RFDETR{model_info.get('model_size', 'Medium').capitalize()}()
# Load from checkpoint
model.load_state_dict(torch.load("path/to/checkpoint.pt"))
model.eval()

# Run inference
image = Image.open("form.jpg")
predictions = model.predict(image)
print(predictions)
```

## Training Details

- Learning Rate: {model_info.get('learning_rate', 'N/A')}
- Batch Size: {model_info.get('batch_size', 'N/A')}
- Effective Batch Size: {model_info.get('batch_size', 1) * model_info.get('grad_accum_steps', 1)}
- Epochs Trained: {model_info.get('epochs_trained', 'N/A')}
"""

        # Create repo if it doesn't exist
        create_repo(
            repo_id=hub_model_id,
            token=hf_token,
            exist_ok=True,
            private=False,
        )

        # Upload model checkpoint
        if Path(checkpoint_path).exists():
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo="rfdetr_model.pt",
                repo_id=hub_model_id,
                token=hf_token,
                commit_message=commit_message,
            )
            logger.info(f"  ✓ Uploaded model checkpoint")
        else:
            logger.warning(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
            return False

        # Upload README
        readme_path = Path(checkpoint_path).parent / "README.md"
        with open(readme_path, "w") as f:
            f.write(model_card)

        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=hub_model_id,
            token=hf_token,
            commit_message=commit_message,
        )

        logger.info(f"  ✓ Updated README")

        return True

    except ImportError:
        logger.error("❌ huggingface_hub not installed!")
        logger.info("   Install with: pip install huggingface_hub")
        return False
    except Exception as e:
        logger.error(f"❌ Failed to push to Hub: {e}")
        return False


def find_latest_ema_checkpoint(output_dir):
    """
    Find the latest EMA checkpoint in the output directory.

    Looks for checkpoints in this priority order:
    1. checkpoint_ema.pt / checkpoint_ema.pth (latest EMA checkpoint)
    2. checkpoint_best_ema.pt / checkpoint_best_ema.pth (best EMA checkpoint)
    3. checkpoint_best_total.pt / checkpoint_best_total.pth (best overall)
    4. Most recent checkpoint*.pt / checkpoint*.pth file

    Args:
        output_dir: Directory to search for checkpoints

    Returns:
        Path to the latest checkpoint, or None if no checkpoints found
    """
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return None

    # Priority list of checkpoint names to look for
    priority_checkpoints = [
        "checkpoint_ema.pt",
        "checkpoint_ema.pth",
        "checkpoint_best_ema.pt",
        "checkpoint_best_ema.pth",
        "checkpoint_best_total.pt",
        "checkpoint_best_total.pth",
        "checkpoint_best.pt",
        "checkpoint_best.pth",
    ]

    # Check for priority checkpoints first
    for checkpoint_name in priority_checkpoints:
        checkpoint_path = output_path / checkpoint_name
        if checkpoint_path.exists():
            logger.info(f"Found priority checkpoint: {checkpoint_path}")
            return str(checkpoint_path)

    # If no priority checkpoint found, look for any checkpoint file
    checkpoint_files = list(output_path.glob("checkpoint*.pt")) + list(output_path.glob("checkpoint*.pth"))

    if checkpoint_files:
        # Sort by modification time, most recent first
        latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found latest checkpoint by modification time: {latest_checkpoint}")
        return str(latest_checkpoint)

    logger.warning(f"No checkpoint files found in {output_dir}")
    return None


def parse_args():
    parser = argparse.ArgumentParser()

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="jbarrow/CommonForms")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)

    # Model args
    parser.add_argument("--model_size", type=str, default="medium", choices=["small", "medium", "large"])

    # Training args
    parser.add_argument("--output_dir", type=str, default="./rfdetr-commonforms")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    # HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)

    # Resume training
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint file to resume from (e.g., checkpoint_ema.pt). "
                            "Use 'auto' to automatically find the latest EMA checkpoint.")

    return parser.parse_args()


def convert_hf_to_coco_format(dataset, output_dir, split_name="train"):
    """
    Convert HuggingFace dataset to COCO format that RF-DETR expects.
    
    RF-DETR expects a directory structure like:
    dataset_dir/
        train/
            _annotations.coco.json
            image1.jpg
            image2.jpg
            ...
        valid/
            _annotations.coco.json
            ...
    """
    logger.info(f"Converting {split_name} split to COCO format...")
    
    split_dir = Path(output_dir) / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    # COCO format structure
    coco_output = {
        "info": {
            "description": f"CommonForms {split_name} split",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Get categories from dataset
    try:
        category_names = dataset.features["objects"]["category"].names
        categories = [
            {
                "id": i, 
                "name": name,
                "supercategory": "form_element"  # Required by RF-DETR
            } 
            for i, name in enumerate(category_names)
        ]
        logger.info(f"Found {len(categories)} categories: {category_names}")
    except:
        # Fallback: extract from data
        logger.warning("Could not extract category names, using generic names")
        sample_cats = set()
        for i, sample in enumerate(dataset):
            if i >= 100:
                break
            sample_cats.update(sample["objects"]["category"])
        categories = [
            {
                "id": i, 
                "name": f"class_{i}",
                "supercategory": "form_element"  # Required by RF-DETR
            } 
            for i in sorted(sample_cats)
        ]
        logger.info(f"Found {len(categories)} categories from data")
    
    coco_output["categories"] = categories
    
    annotation_id = 1
    
    # JPEG maximum dimension
    MAX_JPEG_DIM = 65500
    
    # Check how many images are already converted
    existing_images = list(split_dir.glob("image_*.jpg"))
    logger.info(f"Found {len(existing_images)} existing images in {split_name}, will skip these")
    
    # Try to avoid decoding images unless necessary
    dataset_for_conversion = dataset
    try:
        dataset_for_conversion = dataset.cast_column("image", DatasetImage(decode=False))
    except Exception as e:
        logger.debug(f"Unable to disable image decoding for {split_name}: {e}")

    def load_image_from_sample(image_data):
        """Lazily load a PIL image in RGB mode."""
        if isinstance(image_data, Image.Image):
            return image_data.convert("RGB")
        if isinstance(image_data, dict):
            path = image_data.get("path")
            if path and os.path.exists(path):
                with Image.open(path) as img:
                    return img.convert("RGB")
            image_bytes = image_data.get("bytes")
            if image_bytes is not None:
                with Image.open(io.BytesIO(image_bytes)) as img:
                    return img.convert("RGB")
        # Fallback: assume array-like
        return Image.fromarray(image_data).convert("RGB")

    # Process each sample
    converted_count = 0
    skipped_count = 0
    
    for idx in tqdm(range(len(dataset_for_conversion)), desc=f"Converting {split_name}"):
        sample = dataset_for_conversion[idx]
        image_data = sample["image"]
        image_id = sample.get("id", idx)
        objects = sample.get("objects", {})
        raw_bboxes = list(objects.get("bbox", []))
        categories_list = list(objects.get("category", objects.get("category_id", [])))
        adjusted_bboxes = []
        
        # Check if this image already exists - skip if so
        image_filename = f"image_{image_id:08d}.jpg"
        image_path = split_dir / image_filename
        
        if image_path.exists():
            skipped_count += 1
            width = height = None
            try:
                with Image.open(image_path) as existing_img:
                    width, height = existing_img.size
            except Exception:
                width = height = None
            
            if width is not None and height is not None:
                coco_output["images"].append({
                    "id": image_id,
                    "file_name": image_filename,
                    "width": width,
                    "height": height,
                })
                
                adjusted_bboxes = [
                    [float(x), float(y), float(w), float(h)]
                    for x, y, w, h in raw_bboxes
                ]
                for bbox_vals, category in zip(adjusted_bboxes, categories_list):
                    x, y, w, h = bbox_vals
                    if w <= 0 or h <= 0:
                        continue
                    coco_output["annotations"].append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category,
                        "bbox": [x, y, w, h],
                        "area": float(w * h),
                        "iscrowd": 0,
                    })
                    annotation_id += 1
                continue  # Skip to next image
        
        # Image doesn't exist (or existing image unreadable), process it
        converted_count += 1
        pil_image = load_image_from_sample(image_data)
        orig_width, orig_height = pil_image.size
        width, height = orig_width, orig_height
        scale = None
        
        if width > MAX_JPEG_DIM or height > MAX_JPEG_DIM:
            scale = min(MAX_JPEG_DIM / width, MAX_JPEG_DIM / height)
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            if converted_count <= 5:
                logger.warning(f"Image {image_id} too large ({width}x{height}), resizing to {new_width}x{new_height}")
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            width, height = pil_image.size
        
        if scale is not None:
            adjusted_bboxes = []
            for bbox in raw_bboxes:
                x, y, w, h = bbox
                adjusted_bboxes.append([
                    float(x * scale),
                    float(y * scale),
                    float(w * scale),
                    float(h * scale)
                ])
        else:
            adjusted_bboxes = [
                [float(x), float(y), float(w), float(h)]
                for x, y, w, h in raw_bboxes
            ]
        
        try:
            pil_image.save(image_path, "JPEG", quality=95)
        except Exception as e:
            logger.error(f"Failed to save image {image_id}: {e}")
            if hasattr(pil_image, "close"):
                pil_image.close()
            continue
        finally:
            if hasattr(pil_image, "close"):
                pil_image.close()
        
        coco_output["images"].append({
            "id": image_id,
            "file_name": image_filename,
            "width": width,
            "height": height,
        })
        
        for bbox_vals, category in zip(adjusted_bboxes, categories_list):
            x, y, w, h = bbox_vals
            if w <= 0 or h <= 0:
                continue
            coco_output["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category,
                "bbox": [x, y, w, h],
                "area": float(w * h),
                "iscrowd": 0,
            })
            annotation_id += 1
    
    # Save COCO annotations
    annotations_path = split_dir / "_annotations.coco.json"
    with open(annotations_path, "w") as f:
        json.dump(coco_output, f, indent=2)
    
    logger.info(f"✓ {split_name} split processed:")
    logger.info(f"   Images already converted: {skipped_count}")
    logger.info(f"   Images newly converted: {converted_count}")
    logger.info(f"   Total images: {len(coco_output['images'])}")
    logger.info(f"   Total annotations: {len(coco_output['annotations'])}")
    logger.info(f"   Saved to: {split_dir}")
    
    if skipped_count > 0:
        logger.info(f"   ✓ Resumed from previous run ({skipped_count} images reused)")
    
    return split_dir


def main():
    args = parse_args()
    
    # Setup directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.cache_dir is None:
        args.cache_dir = os.environ.get("CACHE_DIR", "./cache")
    
    # Create temp directory for COCO format conversion
    coco_dataset_dir = output_dir / "coco_format"
    coco_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("RF-DETR Training on CommonForms")
    logger.info("="*80)
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Model size: {args.model_size}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("="*80)
    
    # Load dataset from HuggingFace
    logger.info(f"\nLoading dataset from HuggingFace...")
    dataset = load_dataset(
        args.dataset_name,
        cache_dir=args.cache_dir,
    )
    
    # Get splits
    train_dataset = dataset["train"]
    val_dataset = dataset.get("val", dataset.get("validation", dataset.get("test")))
    
    # Limit samples for testing
    if args.max_train_samples:
        logger.info(f"Limiting training to {args.max_train_samples} samples")
        train_dataset = train_dataset.select(range(min(args.max_train_samples, len(train_dataset))))
    
    if val_dataset and args.max_val_samples:
        logger.info(f"Limiting validation to {args.max_val_samples} samples")
        val_dataset = val_dataset.select(range(min(args.max_val_samples, len(val_dataset))))
    
    logger.info(f"✓ Dataset loaded:")
    logger.info(f"   Train samples: {len(train_dataset)}")
    logger.info(f"   Val samples: {len(val_dataset) if val_dataset else 0}")
    
    # Extract categories for model info
    try:
        category_names = train_dataset.features["objects"]["category"].names
        categories = [
            {
                "id": i, 
                "name": name,
                "supercategory": "form_element"
            } 
            for i, name in enumerate(category_names)
        ]
        logger.info(f"Found {len(categories)} categories: {category_names}")
    except:
        logger.warning("Could not extract category names from schema")
        sample_cats = set()
        for i, sample in enumerate(train_dataset):
            if i >= 100:
                break
            sample_cats.update(sample["objects"]["category"])
        categories = [
            {
                "id": i, 
                "name": f"class_{i}",
                "supercategory": "form_element"
            } 
            for i in sorted(sample_cats)
        ]
        logger.info(f"Found {len(categories)} categories from data")
    
    # Convert to COCO format (with caching)
    logger.info(f"\nPreparing COCO format dataset for RF-DETR...")
    
    # Check if conversion already exists
    train_annotations = coco_dataset_dir / "train" / "_annotations.coco.json"
    valid_annotations = coco_dataset_dir / "valid" / "_annotations.coco.json"
    test_annotations = coco_dataset_dir / "test" / "_annotations.coco.json"
    
    if train_annotations.exists() and valid_annotations.exists() and test_annotations.exists():
        logger.info("✓ COCO format dataset already exists (using cache)")
        logger.info(f"   Directory: {coco_dataset_dir}")
        logger.info("   To force reconversion, delete this directory and rerun")
        
        train_dir = coco_dataset_dir / "train"
        val_dir = coco_dataset_dir / "valid"
        test_dir = coco_dataset_dir / "test"
        
        # Load categories from existing annotations
        with open(train_annotations) as f:
            coco_data = json.load(f)
            logger.info(f"   Found {len(coco_data['images'])} train images")
            logger.info(f"   Found {len(coco_data['annotations'])} train annotations")
            logger.info(f"   Categories: {len(categories)}")
    else:
        logger.info("Converting dataset to COCO format (this may take several minutes)...")
        logger.info("ℹ️  This conversion will be cached for future runs")
        
        train_dir = convert_hf_to_coco_format(train_dataset, coco_dataset_dir, "train")
        
        if val_dataset:
            val_dir = convert_hf_to_coco_format(val_dataset, coco_dataset_dir, "valid")
            # RF-DETR also expects a test split - reuse validation data
            test_dir = convert_hf_to_coco_format(val_dataset, coco_dataset_dir, "test")
        else:
            logger.warning("No validation split found, using train split for validation and test")
            val_dir = convert_hf_to_coco_format(train_dataset, coco_dataset_dir, "valid")
            test_dir = convert_hf_to_coco_format(train_dataset, coco_dataset_dir, "test")
        
        logger.info(f"✓ Dataset conversion complete!")
        logger.info(f"   COCO dataset directory: {coco_dataset_dir}")
        logger.info(f"   Splits created: train, valid, test")
        logger.info(f"   Categories: {categories}")
        logger.info(f"   ✓ Conversion cached for future runs")
    
    # Import RF-DETR
    logger.info(f"\nInitializing RF-DETR model...")
    try:
        from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge

        model_classes = {
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }

        ModelClass = model_classes[args.model_size]
        model = ModelClass()

        logger.info(f"✓ RF-DETR {args.model_size} initialized")

    except ImportError as e:
        logger.error("❌ RF-DETR not installed!")
        logger.error("Install with: pip install rfdetr")
        logger.error(f"Error: {e}")
        sys.exit(1)

    # Handle checkpoint resumption
    checkpoint_path = None
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint.lower() == "auto":
            # Automatically find the latest EMA checkpoint
            logger.info(f"\nSearching for latest checkpoint in {args.output_dir}...")
            checkpoint_path = find_latest_ema_checkpoint(args.output_dir)
            if checkpoint_path:
                logger.info(f"✓ Auto-detected checkpoint: {checkpoint_path}")
            else:
                logger.warning("⚠️  No checkpoint found for auto-resume. Starting from scratch.")
        else:
            # Use the provided checkpoint path
            checkpoint_path = args.resume_from_checkpoint
            if not Path(checkpoint_path).exists():
                logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
                sys.exit(1)
            logger.info(f"✓ Will resume from checkpoint: {checkpoint_path}")

    # Load checkpoint if we have one
    if checkpoint_path:
        logger.info(f"\nLoading checkpoint weights...")
        try:
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')

            # Handle different checkpoint formats
            # Some checkpoints are raw state_dicts, others are wrapped in a dict
            if isinstance(checkpoint_data, dict):
                if 'model' in checkpoint_data:
                    state_dict = checkpoint_data['model']
                elif 'state_dict' in checkpoint_data:
                    state_dict = checkpoint_data['state_dict']
                elif 'model_state_dict' in checkpoint_data:
                    state_dict = checkpoint_data['model_state_dict']
                else:
                    # Assume the dict itself is the state dict
                    state_dict = checkpoint_data
            else:
                state_dict = checkpoint_data

            # Load into model
            if hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
                # RF-DETR wraps the actual model
                model.model.load_state_dict(state_dict, strict=False)
            elif hasattr(model, 'load_state_dict'):
                model.load_state_dict(state_dict, strict=False)
            else:
                logger.error("❌ Unable to load checkpoint: model has no load_state_dict method")
                sys.exit(1)

            logger.info("✓ Checkpoint loaded successfully!")
            logger.info(f"  Resuming training from: {Path(checkpoint_path).name}")

            # Log additional checkpoint info if available
            if isinstance(checkpoint_data, dict):
                if 'epoch' in checkpoint_data:
                    logger.info(f"  Checkpoint epoch: {checkpoint_data['epoch']}")
                if 'best_metric' in checkpoint_data:
                    logger.info(f"  Best metric: {checkpoint_data['best_metric']:.4f}")
                if 'ema' in checkpoint_data:
                    logger.info(f"  EMA checkpoint: Yes")

        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            logger.error("   Starting training from scratch instead")
            import traceback
            traceback.print_exc()
    
    # Prepare model info for HuggingFace uploads
    model_info = {
        "model_type": f"rfdetr-{args.model_size}",
        "model_size": args.model_size,
        "dataset": args.dataset_name,
        "num_classes": len(categories),
        "categories": categories,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "learning_rate": args.learning_rate,
    }

    # Start TensorBoard if available
    def start_tensorboard(logdir, port=6006):
        """Start TensorBoard in the background for monitoring training."""
        try:
            # Check if TensorBoard is available
            result = subprocess.run(
                ["which", "tensorboard"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning("TensorBoard not found in PATH, skipping TensorBoard startup")
                return None
            
            # Kill any existing TensorBoard on the port
            try:
                # Find process using the port
                result = subprocess.run(
                    ["lsof", "-ti", f":{port}"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    for pid in pids:
                        try:
                            os.kill(int(pid), signal.SIGKILL)
                            logger.info(f"Killed existing TensorBoard process (PID: {pid})")
                        except (ProcessLookupError, ValueError):
                            pass
            except (FileNotFoundError, subprocess.SubprocessError):
                # lsof might not be available, try alternative method
                try:
                    result = subprocess.run(
                        ["fuser", "-k", f"{port}/tcp"],
                        capture_output=True,
                        text=True
                    )
                except FileNotFoundError:
                    pass  # fuser also not available, continue anyway
            
            # Start TensorBoard in background
            logdir_path = Path(logdir)
            logdir_path.mkdir(parents=True, exist_ok=True)
            
            # Check for existing event files
            event_files = list(logdir_path.glob("events.out.tfevents.*"))
            if event_files:
                logger.info(f"Found {len(event_files)} existing TensorBoard event files")
            
            logger.info(f"Starting TensorBoard on port {port}...")
            logger.info(f"  Log directory: {logdir_path}")
            logger.info(f"  Access via RunPod HTTP Services on port {port}")
            logger.info(f"  Auto-reload: Every 5 seconds")
            logger.info(f"  Note: Dashboard will appear once training starts logging metrics (after a few steps)")
            
            # Start TensorBoard as a background process
            # Add reload_interval to auto-refresh when new data arrives
            process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir", str(logdir_path),
                    "--host", "0.0.0.0",
                    "--port", str(port),
                    "--reload_interval", "5",  # Reload every 5 seconds
                    "--reload_multifile", "true",  # Watch for multiple event files
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # Create new process group
            )
            
            logger.info(f"✓ TensorBoard started with PID {process.pid}")
            return process
            
        except Exception as e:
            logger.warning(f"Failed to start TensorBoard: {e}")
            logger.info("  Training will continue without TensorBoard")
            return None
    
    # Train
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)

    # Start TensorBoard before training
    # RF-DETR may write logs directly to output_dir or to a subdirectory
    # Try both the output_dir and a runs subdirectory
    tensorboard_logdir = output_dir
    # Check if there's a runs subdirectory (common pattern)
    runs_dir = output_dir / "runs"
    if runs_dir.exists():
        tensorboard_logdir = runs_dir
        logger.info(f"Found runs/ subdirectory, using it for TensorBoard")
    
    tensorboard_process = None
    try:
        tensorboard_process = start_tensorboard(
            logdir=str(tensorboard_logdir),
            port=6006
        )
    except Exception as e:
        logger.warning(f"Could not start TensorBoard: {e}")
        logger.info("Training will continue without TensorBoard")

    # Get HF token if needed
    hf_token = os.environ.get("HF_TOKEN") if args.push_to_hub else None

    try:
        logger.info("Starting training...")
        if args.push_to_hub and args.hub_model_id:
            logger.info(f"Final model will be uploaded to: {args.hub_model_id}")

        # Ensure TensorBoard log directory exists
        # RF-DETR may write logs to output_dir or a subdirectory
        tensorboard_log_path = Path(tensorboard_logdir)
        tensorboard_log_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard log directory: {tensorboard_log_path}")
        logger.info(f"  Event files will appear here during training")

        # RF-DETR's train() method - tensorboard=True enables logging
        # The logs are typically written to output_dir or a subdirectory
        # We'll monitor the output_dir for event files
        logger.info(f"Starting training with TensorBoard logging enabled")
        logger.info(f"  RF-DETR will write logs to: {args.output_dir}")
        logger.info(f"  TensorBoard is monitoring: {tensorboard_log_path}")
        logger.info(f"  If logs appear elsewhere, check subdirectories in {args.output_dir}")

        model.train(
            dataset_dir=str(coco_dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.learning_rate,
            num_workers=args.num_workers,
            output_dir=args.output_dir,
            tensorboard=True,
            early_stopping=True,
        )

        logger.info("="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        logger.info("RF-DETR has saved checkpoints during training to the output directory")

    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        # Clean up TensorBoard on error
        if tensorboard_process:
            try:
                os.killpg(os.getpgid(tensorboard_process.pid), signal.SIGTERM)
                logger.info("Stopped TensorBoard process")
            except (ProcessLookupError, OSError):
                pass
        raise
    finally:
        # Clean up TensorBoard when done
        if tensorboard_process:
            try:
                logger.info("Stopping TensorBoard...")
                os.killpg(os.getpgid(tensorboard_process.pid), signal.SIGTERM)
                logger.info("✓ TensorBoard stopped")
            except (ProcessLookupError, OSError):
                pass
    
    # Save final model
    logger.info(f"\nLocating final trained model...")

    # RF-DETR saves its own checkpoints during training
    # Look for checkpoint files in output directory
    checkpoint_files = list(output_dir.glob("checkpoint*.pth")) + list(output_dir.glob("checkpoint*.pt"))

    if checkpoint_files:
        # Use the latest checkpoint as final model
        final_model_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"✓ Found RF-DETR final checkpoint: {final_model_path}")

        # Create a copy named as final model
        final_model_save_path = output_dir / "rfdetr_model_final.pt"
        if final_model_path != final_model_save_path:
            shutil.copy(final_model_path, final_model_save_path)
            logger.info(f"✓ Copied final model to: {final_model_save_path}")
    else:
        # Try to save manually
        logger.info(f"No checkpoints found, attempting manual save...")
        final_model_save_path = output_dir / "rfdetr_model_final.pt"

        try:
            if hasattr(model, 'save'):
                model.save(str(final_model_save_path))
                logger.info(f"✓ Model saved using model.save()")
            elif hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                # Save the underlying PyTorch model
                torch.save(model.model.state_dict(), final_model_save_path)
                logger.info(f"✓ Model saved using model.model.state_dict()")
            else:
                # Fallback: save entire model object
                torch.save(model, final_model_save_path)
                logger.info(f"✓ Model saved as torch object")
        except Exception as e:
            logger.error(f"❌ Could not save model: {e}")
            logger.warning("   Model may have been saved by RF-DETR's training loop")
            final_model_save_path = None

    # Update model info
    model_info["epochs_trained"] = args.epochs

    # Save model info
    info_path = output_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"✓ Model info saved to {info_path}")

    # Push final model to HuggingFace Hub if requested
    if args.push_to_hub and args.hub_model_id and hf_token:
        logger.info("\n" + "="*80)
        logger.info("Uploading final model to HuggingFace Hub...")
        logger.info("="*80)

        if final_model_save_path and final_model_save_path.exists():
            success = push_final_model_to_hub(
                checkpoint_path=str(final_model_save_path),
                hub_model_id=args.hub_model_id,
                model_info=model_info,
                hf_token=hf_token,
            )

            if success:
                logger.info("="*80)
                logger.info(f"✅ Final model successfully pushed to HuggingFace Hub!")
                logger.info(f"   View at: https://huggingface.co/{args.hub_model_id}")
                logger.info("="*80)
            else:
                logger.warning("⚠️  Final model upload failed")

            # Also upload model info file
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=hf_token)
                api.upload_file(
                    path_or_fileobj=str(info_path),
                    path_in_repo="model_info.json",
                    repo_id=args.hub_model_id,
                    token=hf_token,
                    commit_message="Upload model info",
                )
                logger.info("  ✓ Model info uploaded")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not upload model info: {e}")
        else:
            logger.warning("⚠️  Final model not found, skipping upload")
    elif args.push_to_hub and not hf_token:
        logger.warning("⚠️  HF_TOKEN not found in environment")
        logger.info("   Set it with: export HF_TOKEN=your_token")
        logger.info("   Skipping Hub upload")
    
    # Cleanup COCO format directory if desired
    # Uncomment to remove temporary COCO files:
    # shutil.rmtree(coco_dataset_dir)
    # logger.info(f"✓ Cleaned up temporary COCO directory")

    logger.info("\n✅ All done!")
    logger.info(f"   Model saved to: {output_dir}")
    if args.push_to_hub and args.hub_model_id:
        logger.info(f"   Hub URL: https://huggingface.co/{args.hub_model_id}")
        logger.info(f"   Final model uploaded to Hub")


if __name__ == "__main__":
    main()
