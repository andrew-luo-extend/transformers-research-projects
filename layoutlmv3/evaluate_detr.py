#!/usr/bin/env python3
"""
Evaluate Deformable DETR model on test split using COCO mAP metrics.

Usage:
    python evaluate_detr.py \
        --model_name_or_path your-username/deformable-detr-commonforms \
        --dataset_name jbarrow/CommonForms \
        --test_split test \
        --output_dir ./eval_results

Environment variables:
    HF_TOKEN: Hugging Face token for private models (optional)
    CACHE_DIR: Directory for caching datasets/models (default: ./cache)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

from datasets import load_dataset
from transformers import AutoModelForObjectDetection, AutoImageProcessor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate object detection model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="HuggingFace model ID or path to local model"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="HuggingFace dataset name (e.g., jbarrow/CommonForms)"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory for caching datasets/models (default: env var CACHE_DIR or ./cache)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run evaluation on (default: cuda if available, else cpu)"
    )
    return parser.parse_args()


def convert_to_xywh(boxes):
    """Convert boxes from [x1, y1, x2, y2] to [x, y, w, h] format."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def prepare_for_coco_detection(predictions):
    """Convert model predictions to COCO format."""
    coco_results = []
    for original_id, prediction in predictions.items():
        if len(prediction) == 0:
            continue

        boxes = prediction["boxes"]
        boxes = convert_to_xywh(boxes).tolist()
        scores = prediction["scores"].tolist()
        labels = prediction["labels"].tolist()

        coco_results.extend(
            [
                {
                    "image_id": original_id,
                    "category_id": labels[k],
                    "bbox": box,
                    "score": scores[k],
                }
                for k, box in enumerate(boxes)
            ]
        )
    return coco_results


def prepare_ground_truth(dataset):
    """Convert dataset annotations to COCO format."""
    coco_annotations = []
    coco_images = []

    annotation_id = 1

    # Detect which key holds the category information
    # Try common candidates: category, category_id, label, class
    category_key = None
    if len(dataset) > 0:
        first_sample = dataset[0]
        objects = first_sample.get("objects", {})
        for candidate in ["category", "category_id", "label", "class", "id"]:
            if candidate in objects:
                category_key = candidate
                logger.debug(f"Using category key: {category_key}")
                break

    if category_key is None:
        category_key = "category"  # Default fallback
        logger.warning(f"Could not detect category key, using default: {category_key}")

    for idx, sample in enumerate(dataset):
        image_id = sample.get("id", idx)

        # Get image info
        image = sample["image"]
        if isinstance(image, Image.Image):
            width, height = image.size
        else:
            height, width = image.shape[:2]

        coco_images.append({
            "id": image_id,
            "width": width,
            "height": height,
        })

        # Get annotations
        objects = sample.get("objects", {})
        boxes = objects.get("bbox", [])
        categories = objects.get(category_key, [])

        for box, category in zip(boxes, categories):
            # Convert [x, y, w, h] to COCO format
            x, y, w, h = box

            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue

            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            })
            annotation_id += 1

    return coco_images, coco_annotations


def run_inference(model, processor, dataset, device):
    """Run inference on all samples in the dataset."""
    logger.info("Running inference on test set...")
    predictions = {}

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Inference"):
            sample = dataset[idx]
            image = sample["image"]
            image_id = sample.get("id", idx)

            # Preprocess single image
            inputs = processor(images=image, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run model
            outputs = model(**inputs)

            # Post-process predictions
            # Get actual image size
            if isinstance(image, Image.Image):
                target_size = torch.tensor([image.size[::-1]]).to(device)
            else:
                target_size = torch.tensor([image.shape[:2]]).to(device)
            
            # Debug first image to check bbox format
            if idx == 0:
                logger.info(f"\n🔍 DEBUG: First image processing")
                logger.info(f"   Image size: {image.size if isinstance(image, Image.Image) else image.shape[:2]}")
                logger.info(f"   Input tensor shape: {inputs['pixel_values'].shape}")
                logger.info(f"   Target size for post-processing: {target_size}")
                logger.info(f"   Raw model output boxes shape: {outputs['pred_boxes'].shape}")
                logger.info(f"   Raw model output boxes (first 3):")
                raw_boxes = outputs['pred_boxes'][0][:3].cpu()
                for i, box in enumerate(raw_boxes):
                    logger.info(f"      Raw box {i}: {box.tolist()}")
                logger.info(f"   These should be in normalized [cx, cy, w, h] format (0-1 range)")

            results = processor.post_process_object_detection(
                outputs,
                threshold=0.0,  # Keep all predictions for mAP calculation
                target_sizes=target_size,
            )
            
            # Debug first image after post-processing
            if idx == 0:
                logger.info(f"   After post_process_object_detection:")
                logger.info(f"   Number of boxes: {len(results[0]['boxes'])}")
                logger.info(f"   Boxes format: should be [x1, y1, x2, y2] in absolute pixels")
                logger.info(f"   First 3 boxes:")
                for i, box in enumerate(results[0]['boxes'][:3]):
                    logger.info(f"      Box {i}: {box.tolist()}")
                logger.info(f"   First 3 scores: {results[0]['scores'][:3].tolist()}")
                logger.info(f"   First 3 labels: {results[0]['labels'][:3].tolist()}")

            # Store predictions (results is a list with one element)
            predictions[image_id] = results[0]

    logger.info(f"✓ Generated predictions for {len(predictions)} images")
    
    # Debug: Show predicted categories
    all_predicted_labels = []
    for pred in predictions.values():
        if len(pred["labels"]) > 0:
            all_predicted_labels.extend(pred["labels"].tolist())
    
    if len(all_predicted_labels) > 0:
        unique_predicted = set(all_predicted_labels)
        logger.info(f"Predicted categories: {sorted(unique_predicted)}")
        logger.info(f"Total predictions: {len(all_predicted_labels)}")
    else:
        logger.warning("⚠️  No objects detected in any image!")
    
    return predictions


def evaluate_coco_metrics(dataset, predictions, output_dir, dataset_name="CommonForms"):
    """Evaluate predictions using COCO metrics."""
    # Prepare ground truth in COCO format
    logger.info("Preparing ground truth annotations...")
    coco_images, coco_annotations = prepare_ground_truth(dataset)

    # Get category names if available
    try:
        # Try different ways to access category names
        if hasattr(dataset.features["objects"], "feature"):
            # Datasets >= 2.0 format
            category_feature = dataset.features["objects"].feature["category"]
        else:
            # Older datasets format - objects is a dict of features
            category_feature = dataset.features["objects"]["category"]

        # Get names if available
        if hasattr(category_feature, "names"):
            category_names = category_feature.names
        elif hasattr(category_feature, "_str2int"):
            # Alternative way to get class names
            category_names = list(category_feature._str2int.keys())
        else:
            raise AttributeError("No category names found")

        categories = [
            {"id": i, "name": name} for i, name in enumerate(category_names)
        ]
        logger.info(f"Found {len(categories)} category names from dataset")
    except (AttributeError, KeyError) as e:
        logger.warning(f"Could not extract category names from dataset: {e}")
        # Use generic category names
        num_categories = max([ann["category_id"] for ann in coco_annotations]) + 1
        categories = [
            {"id": i, "name": f"class_{i}"} for i in range(num_categories)
        ]
        logger.info(f"Using generic category names for {num_categories} classes")

    # Create COCO ground truth object
    coco_gt = COCO()
    coco_gt.dataset = {
        "info": {
            "description": f"Evaluation on {dataset_name}",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories,
    }
    coco_gt.createIndex()

    logger.info(f"✓ Ground truth: {len(coco_images)} images, {len(coco_annotations)} annotations")
    
    # Debug: Show ground truth categories
    gt_categories = set([ann["category_id"] for ann in coco_annotations])
    logger.info(f"Ground truth categories: {sorted(gt_categories)}")
    logger.info(f"Category mapping: {categories}")

    # Prepare predictions in COCO format
    logger.info("Preparing predictions...")
    coco_results = prepare_for_coco_detection(predictions)
    logger.info(f"✓ Predictions: {len(coco_results)} detections")
    
    # Debug: Show predicted categories in COCO results
    if len(coco_results) > 0:
        pred_categories = set([pred["category_id"] for pred in coco_results])
        logger.info(f"Predicted category IDs in detections: {sorted(pred_categories)}")
        logger.info(f"⚠️  CATEGORY MISMATCH CHECK:")
        logger.info(f"   Ground truth has categories: {sorted(gt_categories)}")
        logger.info(f"   Predictions have categories: {sorted(pred_categories)}")
        
        overlap = gt_categories.intersection(pred_categories)
        if len(overlap) == 0:
            logger.warning(f"   ❌ NO OVERLAP! Model is predicting different category IDs than ground truth!")
            logger.warning(f"   This explains why mAP is 0.000")
        else:
            logger.info(f"   ✓ Overlap: {sorted(overlap)}")
        
        # Debug: Show sample bbox values to check coordinate system
        logger.info(f"\n📦 BBOX COORDINATE CHECK:")
        logger.info(f"   Sample ground truth boxes (first 3):")
        for i, ann in enumerate(coco_annotations[:3]):
            logger.info(f"      GT {i}: {ann['bbox']} (image_id={ann['image_id']}, cat={ann['category_id']})")
        
        logger.info(f"   Sample predictions (first 3):")
        for i, pred in enumerate(coco_results[:3]):
            logger.info(f"      Pred {i}: {pred['bbox']} (image_id={pred['image_id']}, cat={pred['category_id']}, score={pred['score']:.3f})")
        
        # Check image size to see if predictions are reasonable
        first_img = coco_images[0]
        logger.info(f"   First image dimensions: {first_img['width']}x{first_img['height']}")
        logger.info(f"   ⚠️  If predicted boxes have values > image dimensions, there's a scaling issue!")

    if len(coco_results) == 0:
        logger.warning("⚠️  No predictions generated! Model may not be detecting any objects.")
        return None, None, categories

    # Run COCO evaluation
    logger.info("\n" + "=" * 80)
    logger.info("Calculating COCO metrics...")
    logger.info("=" * 80)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Calculate per-class AP
    logger.info("\n" + "=" * 80)
    logger.info("Per-Class Average Precision (AP @ IoU 0.5:0.95)")
    logger.info("=" * 80)

    per_class_ap = {}

    for cat_id, cat_info in enumerate(categories):
        cat_name = cat_info["name"]

        # Run evaluation for this category only
        coco_eval_cat = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval_cat.params.catIds = [cat_id]
        coco_eval_cat.evaluate()
        coco_eval_cat.accumulate()

        # Get AP @ IoU 0.5:0.95
        # Check if stats has been populated
        if hasattr(coco_eval_cat, 'stats') and len(coco_eval_cat.stats) > 0:
            ap = coco_eval_cat.stats[0]  # AP @ IoU 0.5:0.95
        else:
            # No detections for this category
            try:
                coco_eval_cat.summarize()
                ap = coco_eval_cat.stats[0] if len(coco_eval_cat.stats) > 0 else -1.0
            except:
                ap = -1.0
        
        per_class_ap[cat_name] = float(ap)

        if ap == -1.0:
            logger.info(f"  {cat_name:30s}: No predictions")
        else:
            logger.info(f"  {cat_name:30s}: {ap:.4f}")

    logger.info("=" * 80)

    return coco_eval, per_class_ap, categories


def save_results(args, coco_eval, per_class_ap, output_dir):
    """Save evaluation results to JSON file."""
    if coco_eval is None:
        logger.warning("No results to save (no predictions generated)")
        return

    results_dict = {
        "model_id": args.model_name_or_path,
        "dataset": args.dataset_name,
        "split": args.test_split,
        "metrics": {
            "mAP_50_95": float(coco_eval.stats[0]),
            "mAP_50": float(coco_eval.stats[1]),
            "mAP_75": float(coco_eval.stats[2]),
            "mAP_small": float(coco_eval.stats[3]),
            "mAP_medium": float(coco_eval.stats[4]),
            "mAP_large": float(coco_eval.stats[5]),
            "mAR_1": float(coco_eval.stats[6]),
            "mAR_10": float(coco_eval.stats[7]),
            "mAR_100": float(coco_eval.stats[8]),
            "mAR_small": float(coco_eval.stats[9]),
            "mAR_medium": float(coco_eval.stats[10]),
            "mAR_large": float(coco_eval.stats[11]),
        },
        "per_class_ap": per_class_ap,
    }

    # Save results
    output_file = Path(output_dir) / "evaluation_results.json"
    with open(output_file, "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"\n✓ Results saved to {output_file}")

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"\nModel: {args.model_name_or_path}")
    logger.info(f"Dataset: {args.dataset_name} ({args.test_split} split)")
    logger.info(f"\nKey Metrics:")
    logger.info(f"  mAP @ IoU 0.5:0.95    : {results_dict['metrics']['mAP_50_95']:.4f}")
    logger.info(f"  mAP @ IoU 0.5         : {results_dict['metrics']['mAP_50']:.4f}")
    logger.info(f"  mAP @ IoU 0.75        : {results_dict['metrics']['mAP_75']:.4f}")
    logger.info("=" * 80)


def main():
    args = parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup cache directory
    if args.cache_dir is None:
        args.cache_dir = os.environ.get("CACHE_DIR", "./cache")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Authenticate with HuggingFace if token is available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token)
        logger.info("✓ Authenticated with HuggingFace")

    # Load model and processor
    logger.info(f"Loading model from {args.model_name_or_path}...")
    model = AutoModelForObjectDetection.from_pretrained(
        args.model_name_or_path,
        cache_dir=str(cache_dir),
    )
    processor = AutoImageProcessor.from_pretrained(
        args.model_name_or_path,
        cache_dir=str(cache_dir),
    )

    model.to(device)
    model.eval()

    logger.info(f"✓ Model loaded with {model.config.num_labels} classes")

    # Load dataset
    logger.info(f"Loading dataset {args.dataset_name}, split: {args.test_split}...")
    dataset = load_dataset(
        args.dataset_name,
        split=args.test_split,
        cache_dir=str(cache_dir),
    )

    if args.max_samples is not None:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        logger.info(f"Using first {len(dataset)} samples for evaluation")
    else:
        logger.info(f"Using full {args.test_split} split: {len(dataset)} samples")

    # Run inference
    predictions = run_inference(model, processor, dataset, device)

    # Evaluate
    coco_eval, per_class_ap, categories = evaluate_coco_metrics(
        dataset, predictions, output_dir, dataset_name=args.dataset_name
    )

    # Save results
    save_results(args, coco_eval, per_class_ap, output_dir)

    logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
