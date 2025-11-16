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
from pathlib import Path
import shutil
import warnings

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
    
    # Train
    logger.info("")
    logger.info("="*80)
    logger.info("STARTING TRAINING")
    logger.info("="*80)
    
    try:
        model.train(
            dataset_dir=str(coco_dataset_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            lr=args.learning_rate,
            num_workers=args.num_workers,
            project=args.output_dir,
        )
        
        logger.info("="*80)
        logger.info("✅ TRAINING COMPLETE!")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        raise
    
    # Save model
    logger.info(f"\nLocating trained model...")
    
    # RF-DETR saves its own checkpoints during training
    # Look for checkpoint files in output directory
    checkpoint_files = list(output_dir.glob("checkpoint*.pth")) + list(output_dir.glob("checkpoint*.pt"))
    
    if checkpoint_files:
        # Use the latest checkpoint
        model_save_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"✓ Found RF-DETR checkpoint: {model_save_path}")
    else:
        # Try to save manually
        logger.info(f"No checkpoints found, attempting manual save...")
        model_save_path = output_dir / "rfdetr_model.pt"
        
        try:
            if hasattr(model, 'save'):
                model.save(str(model_save_path))
                logger.info(f"✓ Model saved using model.save()")
            elif hasattr(model, 'model') and hasattr(model.model, 'state_dict'):
                # Save the underlying PyTorch model
                torch.save(model.model.state_dict(), model_save_path)
                logger.info(f"✓ Model saved using model.model.state_dict()")
            else:
                # Fallback: save entire model object
                torch.save(model, model_save_path)
                logger.info(f"✓ Model saved as torch object")
        except Exception as e:
            logger.error(f"❌ Could not save model: {e}")
            logger.warning("   Model may have been saved by RF-DETR's training loop")
            model_save_path = None
    
    # Save model info
    model_info = {
        "model_type": f"rfdetr-{args.model_size}",
        "dataset": args.dataset_name,
        "num_classes": len(categories),
        "categories": categories,
        "epochs_trained": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    
    info_path = output_dir / "model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"✓ Model info saved to {info_path}")
    
    # Push to HuggingFace Hub if requested
    if args.push_to_hub and args.hub_model_id:
        logger.info("\n" + "="*80)
        logger.info("Pushing to HuggingFace Hub...")
        logger.info("="*80)
        
        try:
            from huggingface_hub import HfApi, create_repo
            
            # Get token from environment
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                logger.warning("⚠️  HF_TOKEN not found in environment")
                logger.info("   Set it with: export HF_TOKEN=your_token")
                logger.info("   Skipping Hub upload")
            else:
                api = HfApi(token=hf_token)
                
                # Create repo if it doesn't exist
                logger.info(f"Creating/accessing repository: {args.hub_model_id}")
                create_repo(
                    repo_id=args.hub_model_id,
                    token=hf_token,
                    exist_ok=True,
                    private=False,
                )
                
                # Create model card
                model_card = f"""---
tags:
- object-detection
- rf-detr
- commonforms
datasets:
- {args.dataset_name}
---

# RF-DETR Fine-tuned on CommonForms

This model is an RF-DETR ({args.model_size}) fine-tuned on the [CommonForms]({args.dataset_name}) dataset for form field detection.

## Model Details

- **Model Type:** RF-DETR {args.model_size}
- **Dataset:** {args.dataset_name}
- **Classes:** {len(categories)}
- **Epochs:** {args.epochs}
- **Batch Size:** {args.batch_size} (grad_accum: {args.grad_accum_steps})

## Classes

{json.dumps(categories, indent=2)}

## Usage

```python
import torch
from PIL import Image

# Load model
model_path = "path/to/rfdetr_model.pt"
# Note: You'll need the rfdetr library installed
from rfdetr import RFDETR{args.model_size.capitalize()}

model = RFDETR{args.model_size.capitalize()}()
model.load_state_dict(torch.load(model_path))
model.eval()

# Run inference
image = Image.open("form.jpg")
predictions = model.predict(image)
print(predictions)
```

## Training Details

- Learning Rate: {args.learning_rate}
- Effective Batch Size: {args.batch_size * args.grad_accum_steps}
- Dataset: Trained on CommonForms (form field detection)

## Metrics

(Add your evaluation metrics here after running evaluation)

## Citation

```bibtex
@misc{{rfdetr-commonforms,
  author = {{Your Name}},
  title = {{RF-DETR Fine-tuned on CommonForms}},
  year = {{2024}},
  publisher = {{HuggingFace}},
  howpublished = {{\\url{{https://huggingface.co/{args.hub_model_id}}}}}
}}
```
"""
                
                readme_path = output_dir / "README.md"
                with open(readme_path, "w") as f:
                    f.write(model_card)
                logger.info(f"✓ Model card created")
                
                # Upload files
                logger.info(f"Uploading files to {args.hub_model_id}...")
                
                # Upload model weights (if successfully saved)
                if model_save_path and model_save_path.exists():
                    api.upload_file(
                        path_or_fileobj=str(model_save_path),
                        path_in_repo="rfdetr_model.pt",
                        repo_id=args.hub_model_id,
                        token=hf_token,
                    )
                    logger.info("  ✓ Model weights uploaded")
                else:
                    logger.warning("  ⚠️  Model weights not uploaded (save failed)")
                
                # Upload model info
                api.upload_file(
                    path_or_fileobj=str(info_path),
                    path_in_repo="model_info.json",
                    repo_id=args.hub_model_id,
                    token=hf_token,
                )
                logger.info("  ✓ Model info uploaded")
                
                # Upload README
                api.upload_file(
                    path_or_fileobj=str(readme_path),
                    path_in_repo="README.md",
                    repo_id=args.hub_model_id,
                    token=hf_token,
                )
                logger.info("  ✓ README uploaded")
                
                logger.info("="*80)
                logger.info(f"✅ Model successfully pushed to HuggingFace Hub!")
                logger.info(f"   View at: https://huggingface.co/{args.hub_model_id}")
                logger.info("="*80)
                
        except ImportError:
            logger.error("❌ huggingface_hub not installed!")
            logger.info("   Install with: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"❌ Failed to push to Hub: {e}")
            logger.info(f"   Model saved locally at: {model_save_path}")
    
    # Cleanup COCO format directory if desired
    # Uncomment to remove temporary COCO files:
    # shutil.rmtree(coco_dataset_dir)
    # logger.info(f"✓ Cleaned up temporary COCO directory")
    
    logger.info("\n✅ All done!")
    logger.info(f"   Model saved to: {output_dir}")
    if args.push_to_hub and args.hub_model_id:
        logger.info(f"   Hub URL: https://huggingface.co/{args.hub_model_id}")


if __name__ == "__main__":
    main()
