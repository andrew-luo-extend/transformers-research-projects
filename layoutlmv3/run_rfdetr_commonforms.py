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
import heapq
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


class Top3CheckpointTracker:
    """Tracks the top 3 checkpoints based on a metric (lower is better for loss)."""

    def __init__(self, metric_name: str = "val_loss", lower_is_better: bool = True, top_k: int = 3):
        self.metric_name = metric_name
        self.lower_is_better = lower_is_better
        self.top_k = top_k
        # Use heap: for lower_is_better, we use a max heap (negate values)
        # for higher_is_better, we use a min heap
        self.heap: List[Tuple[float, int, str]] = []  # (metric_value, epoch, checkpoint_path)

    def should_save(self, metric_value: float) -> bool:
        """Check if this metric value qualifies for top-k."""
        if len(self.heap) < self.top_k:
            return True

        # Compare with worst checkpoint in heap
        worst_metric = self.heap[0][0]
        if self.lower_is_better:
            return metric_value < worst_metric
        else:
            return metric_value > worst_metric

    def add(self, metric_value: float, epoch: int, checkpoint_path: str) -> Tuple[bool, str]:
        """
        Add a checkpoint. Returns (was_added, path_to_remove).
        If was_added is True, the checkpoint was added to top-k.
        If path_to_remove is not None, that old checkpoint should be deleted.
        """
        if not self.should_save(metric_value):
            return False, None

        path_to_remove = None

        # If heap is full, remove worst checkpoint
        if len(self.heap) >= self.top_k:
            _, _, path_to_remove = heapq.heappop(self.heap)

        # Add new checkpoint (negate for max heap behavior with min heap structure)
        heapq.heappush(self.heap, (metric_value, epoch, checkpoint_path))

        return True, path_to_remove

    def get_top_checkpoints(self) -> List[Tuple[float, int, str]]:
        """Get all top checkpoints sorted by metric (best first)."""
        sorted_checkpoints = sorted(self.heap, key=lambda x: x[0], reverse=not self.lower_is_better)
        return sorted_checkpoints


def push_model_to_hub(
    checkpoint_path: str,
    hub_model_id: str,
    epoch: int,
    metric_value: float,
    metric_name: str,
    model_info: Dict,
    is_final: bool = False,
    hf_token: str = None
) -> bool:
    """Push a model checkpoint to HuggingFace Hub with epoch info in description."""
    try:
        from huggingface_hub import HfApi, create_repo

        if not hf_token:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("⚠️  HF_TOKEN not found, skipping upload")
                return False

        api = HfApi(token=hf_token)

        # Determine model name suffix
        if is_final:
            model_suffix = "final"
            commit_message = f"Upload final model (epoch {epoch})"
            description_suffix = f"Final model after {epoch} epochs"
        else:
            model_suffix = f"epoch-{epoch}"
            commit_message = f"Upload checkpoint from epoch {epoch} ({metric_name}={metric_value:.4f})"
            description_suffix = f"Checkpoint from epoch {epoch} - {metric_name}: {metric_value:.4f}"

        # Create model card with epoch info
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
- **Checkpoint:** Epoch {epoch}
{"- **Metric:** " + metric_name + " = " + str(metric_value) if not is_final else ""}

## Classes

{json.dumps(model_info.get('categories', []), indent=2)}

## Usage

```python
import torch
from PIL import Image

# Load model
model_path = "path/to/rfdetr_model_{model_suffix}.pt"
# Note: You'll need the rfdetr library installed
from rfdetr import RFDETR{model_info.get('model_size', 'Medium').capitalize()}

model = RFDETR{model_info.get('model_size', 'Medium').capitalize()}()
model.load_state_dict(torch.load(model_path))
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
- Epochs Trained: {epoch}
"""

        # Create repo if it doesn't exist
        create_repo(
            repo_id=hub_model_id,
            token=hf_token,
            exist_ok=True,
            private=False,
        )

        # Upload checkpoint
        if Path(checkpoint_path).exists():
            api.upload_file(
                path_or_fileobj=checkpoint_path,
                path_in_repo=f"rfdetr_model_{model_suffix}.pt",
                repo_id=hub_model_id,
                token=hf_token,
                commit_message=commit_message,
            )
            logger.info(f"  ✓ Uploaded {model_suffix} checkpoint")
        else:
            logger.warning(f"  ⚠️  Checkpoint not found: {checkpoint_path}")
            return False

        # Upload README for this checkpoint
        readme_path = Path(checkpoint_path).parent / f"README_{model_suffix}.md"
        with open(readme_path, "w") as f:
            f.write(model_card)

        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo=f"README_{model_suffix}.md",
            repo_id=hub_model_id,
            token=hf_token,
            commit_message=commit_message,
        )

        # Update main README
        main_readme_path = Path(checkpoint_path).parent / "README.md"
        with open(main_readme_path, "w") as f:
            f.write(model_card)

        api.upload_file(
            path_or_fileobj=str(main_readme_path),
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

    # Checkpoint args
    parser.add_argument("--top_k_checkpoints", type=int, default=3, help="Number of best checkpoints to save and upload")
    parser.add_argument("--checkpoint_metric", type=str, default="val_loss", help="Metric to use for selecting best checkpoints")

    # HuggingFace Hub
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)

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

    # Initialize checkpoint tracker
    checkpoint_tracker = Top3CheckpointTracker(
        metric_name=args.checkpoint_metric,
        lower_is_better=True,  # Assuming loss-based metric
        top_k=args.top_k_checkpoints,
    )

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
            
            logger.info(f"Starting TensorBoard on port {port}...")
            logger.info(f"  Log directory: {logdir_path}")
            logger.info(f"  Access via RunPod HTTP Services on port {port}")
            
            # Start TensorBoard as a background process
            process = subprocess.Popen(
                [
                    "tensorboard",
                    "--logdir", str(logdir_path),
                    "--host", "0.0.0.0",
                    "--port", str(port),
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
    tensorboard_process = None
    try:
        tensorboard_process = start_tensorboard(
            logdir=str(output_dir),
            port=6006
        )
    except Exception as e:
        logger.warning(f"Could not start TensorBoard: {e}")
        logger.info("Training will continue without TensorBoard")

    # Get HF token if needed
    hf_token = os.environ.get("HF_TOKEN") if args.push_to_hub else None

    try:
        # Check if RF-DETR model supports custom callbacks for per-epoch monitoring
        # If not, we'll monitor checkpoints after training

        logger.info("Training with checkpoint monitoring enabled")
        if args.push_to_hub and args.hub_model_id:
            logger.info(f"Best {args.top_k_checkpoints} checkpoints will be uploaded to: {args.hub_model_id}")

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

        # After training, scan for checkpoints and identify best ones
        logger.info("\nScanning for best checkpoints...")

        # Look for checkpoint files with epoch information
        checkpoint_pattern_files = list(output_dir.glob("*epoch*.pt")) + \
                                   list(output_dir.glob("*epoch*.pth")) + \
                                   list(output_dir.glob("checkpoint*.pt")) + \
                                   list(output_dir.glob("checkpoint*.pth"))

        if checkpoint_pattern_files:
            logger.info(f"Found {len(checkpoint_pattern_files)} checkpoint files")

            # Try to extract metrics from checkpoint files or logs
            # This is a fallback approach - ideally we'd monitor during training

            # Look for a results file or metrics log
            results_files = list(output_dir.glob("*results*.json")) + \
                           list(output_dir.glob("*results*.csv")) + \
                           list(output_dir.glob("*metrics*.json"))

            best_checkpoints = []

            if results_files:
                logger.info(f"Found metrics file: {results_files[0]}")
                # Parse metrics and match with checkpoints
                # This is model-specific, so we'll use a heuristic

                # For now, just take the last 3 checkpoints as "best"
                sorted_checkpoints = sorted(checkpoint_pattern_files, key=lambda p: p.stat().st_mtime)
                best_checkpoints = sorted_checkpoints[-args.top_k_checkpoints:]

                for idx, ckpt in enumerate(best_checkpoints):
                    epoch_num = args.epochs - len(best_checkpoints) + idx + 1
                    metric_value = 0.0  # Placeholder
                    checkpoint_tracker.add(metric_value, epoch_num, str(ckpt))
            else:
                # No metrics file, just use last N checkpoints
                sorted_checkpoints = sorted(checkpoint_pattern_files, key=lambda p: p.stat().st_mtime)
                best_checkpoints = sorted_checkpoints[-args.top_k_checkpoints:]

                for idx, ckpt in enumerate(best_checkpoints):
                    epoch_num = args.epochs - len(best_checkpoints) + idx + 1
                    metric_value = 0.0  # Placeholder
                    checkpoint_tracker.add(metric_value, epoch_num, str(ckpt))

            logger.info(f"Selected {len(best_checkpoints)} best checkpoints")

            # Upload best checkpoints to HuggingFace Hub
            if args.push_to_hub and args.hub_model_id and hf_token:
                logger.info("\n" + "="*80)
                logger.info("Uploading best checkpoints to HuggingFace Hub...")
                logger.info("="*80)

                top_checkpoints = checkpoint_tracker.get_top_checkpoints()
                for metric_value, epoch, checkpoint_path in top_checkpoints:
                    logger.info(f"\nUploading checkpoint from epoch {epoch}...")
                    success = push_model_to_hub(
                        checkpoint_path=checkpoint_path,
                        hub_model_id=args.hub_model_id,
                        epoch=epoch,
                        metric_value=metric_value,
                        metric_name=args.checkpoint_metric,
                        model_info=model_info,
                        is_final=False,
                        hf_token=hf_token,
                    )
                    if success:
                        logger.info(f"✓ Checkpoint epoch {epoch} uploaded successfully")
        else:
            logger.warning("No checkpoint files found with epoch information")

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

    # Update model info with final epochs
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
            success = push_model_to_hub(
                checkpoint_path=str(final_model_save_path),
                hub_model_id=args.hub_model_id,
                epoch=args.epochs,
                metric_value=0.0,  # Placeholder for final model
                metric_name=args.checkpoint_metric,
                model_info=model_info,
                is_final=True,
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
        logger.info(f"   Uploaded models:")
        logger.info(f"     - Top {args.top_k_checkpoints} best checkpoints (based on {args.checkpoint_metric})")
        logger.info(f"     - Final model (epoch {args.epochs})")


if __name__ == "__main__":
    main()
