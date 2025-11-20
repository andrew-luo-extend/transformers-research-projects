#!/usr/bin/env python3
"""
Evaluate RF-DETR model on CommonForms test split.

Uses supervision library for mAP calculation (IoU 0.5:0.95).

Usage:
    python evaluate_rfdetr.py \
        --hub_model_id your-username/rfdetr-commonforms \
        --dataset_name jbarrow/CommonForms \
        --test_split test

Environment:
    HF_TOKEN: Optional, for private models
    CACHE_DIR: Optional, default=/workspace/cache
"""

import argparse
import gc
import json
import logging
import os
import sys
import weakref
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm

# HuggingFace
from datasets import load_dataset
from huggingface_hub import hf_hub_download

# Supervision for mAP metrics
import supervision as sv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def cleanup_gpu_memory(obj=None, verbose=False):
    """Clean up GPU memory by clearing cache and garbage collecting."""
    if not torch.cuda.is_available():
        if verbose:
            logger.info("CUDA not available, no GPU cleanup needed")
        return
    
    def get_memory_stats():
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        return allocated, reserved
    
    torch.cuda.synchronize()
    
    if verbose:
        alloc, reserv = get_memory_stats()
        logger.info(f"Before cleanup - Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")
    
    # Drop references
    if obj is not None:
        ref = weakref.ref(obj)
        del obj
        if ref() is not None and verbose:
            logger.warning("Object not fully garbage collected")
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, 'ipc_collect'):
        torch.cuda.ipc_collect()
    torch.cuda.synchronize()
    
    if verbose:
        alloc, reserv = get_memory_stats()
        logger.info(f"After cleanup  - Allocated: {alloc / 1024**2:.2f} MB | Reserved: {reserv / 1024**2:.2f} MB")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR model")
    
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="HuggingFace model ID (e.g., your-username/rfdetr-commonforms)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to local checkpoint file (e.g., checkpoint_best_ema.pth)"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jbarrow/CommonForms",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--test_split",
        type=str,
        default="test",
        help="Dataset split to evaluate (test, val, or validation)"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory (default: env CACHE_DIR or /workspace/cache)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results_rfdetr",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate (for quick testing)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help="Detection confidence threshold (0.0 for mAP calculation)"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="RF-DETR model size"
    )
    
    return parser.parse_args()


def download_model_from_hub(hub_model_id, cache_dir):
    """Download RF-DETR checkpoint from HuggingFace Hub."""
    logger.info(f"Downloading model from {hub_model_id}...")
    
    # Try different possible checkpoint filenames
    checkpoint_files = [
        "rfdetr_model.pt",
        "checkpoint_best_total.pth",
        "checkpoint_best.pth",
        "model.pt",
        "pytorch_model.bin",
    ]
    
    checkpoint_path = None
    for filename in checkpoint_files:
        try:
            checkpoint_path = hf_hub_download(
                repo_id=hub_model_id,
                filename=filename,
                cache_dir=cache_dir,
            )
            logger.info(f"✓ Downloaded: {filename}")
            break
        except Exception as e:
            logger.debug(f"  {filename} not found: {e}")
            continue
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint found in {hub_model_id}. Tried: {checkpoint_files}")
    
    return checkpoint_path


def load_commonforms_as_supervision_dataset(dataset_name, split, cache_dir, max_samples=None):
    """Load CommonForms dataset and convert to supervision format."""
    logger.info(f"Loading {dataset_name} ({split} split)...")
    
    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Using {len(dataset)} samples")
    else:
        logger.info(f"Using full split: {len(dataset)} samples")
    
    # Extract category names
    try:
        category_names = dataset.features["objects"]["category"].names
        logger.info(f"Categories: {category_names}")
    except:
        logger.warning("Could not extract category names")
        category_names = None
    
    # Convert to list of (image, annotations) tuples
    samples = []
    
    for idx in tqdm(range(len(dataset)), desc="Loading samples"):
        sample = dataset[idx]
        image = sample["image"]
        objects = sample["objects"]
        
        # Convert to supervision Detections format
        # supervision expects [x1, y1, x2, y2] format
        bboxes = []
        class_ids = []
        
        for bbox, category in zip(objects["bbox"], objects["category"]):
            x, y, w, h = bbox
            # Skip invalid boxes
            if w <= 0 or h <= 0:
                continue
            # Convert [x, y, w, h] to [x1, y1, x2, y2]
            x1, y1, x2, y2 = x, y, x + w, y + h
            bboxes.append([x1, y1, x2, y2])
            class_ids.append(category)
        
        if len(bboxes) > 0:
            # Create supervision Detections object
            detections = sv.Detections(
                xyxy=torch.tensor(bboxes, dtype=torch.float32).numpy(),
                class_id=torch.tensor(class_ids, dtype=torch.int64).numpy(),
            )
            samples.append((image, detections))
        else:
            # Empty annotation
            samples.append((image, sv.Detections.empty()))
    
    logger.info(f"✓ Loaded {len(samples)} samples")
    return samples, category_names


def main():
    args = parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.cache_dir is None:
        args.cache_dir = os.environ.get("CACHE_DIR", "/workspace/cache")
    
    logger.info("="*80)
    logger.info("RF-DETR Evaluation on CommonForms")
    logger.info("="*80)

    # Determine checkpoint path
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if not Path(checkpoint_path).exists():
            logger.error(f"❌ Checkpoint file not found: {checkpoint_path}")
            sys.exit(1)
        logger.info(f"Using local checkpoint: {checkpoint_path}")
    elif args.hub_model_id:
        logger.info(f"Model: {args.hub_model_id}")
        # Download model from Hub
        checkpoint_path = download_model_from_hub(args.hub_model_id, args.cache_dir)
    else:
        logger.error("❌ Must specify either --checkpoint_path or --hub_model_id")
        sys.exit(1)

    logger.info(f"Dataset: {args.dataset_name} ({args.test_split} split)")
    logger.info(f"Output: {args.output_dir}")
    logger.info("="*80)
    
    # Load RF-DETR model
    logger.info(f"\nLoading RF-DETR {args.model_size} model...")
    try:
        from rfdetr import RFDETRSmall, RFDETRMedium, RFDETRLarge
        
        model_classes = {
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "large": RFDETRLarge,
        }
        
        ModelClass = model_classes[args.model_size]
        model = ModelClass(pretrain_weights=checkpoint_path)

        # Skip optimize_for_inference() - it can fail with TorchScript tracing errors
        # model.optimize_for_inference()

        logger.info(f"✓ Model loaded successfully")
        
    except ImportError:
        logger.error("❌ RF-DETR not installed!")
        logger.info("   Install with: pip install rfdetr")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise
    
    # Load test dataset
    test_samples, category_names = load_commonforms_as_supervision_dataset(
        args.dataset_name,
        args.test_split,
        args.cache_dir,
        args.max_samples
    )
    
    # Run inference
    logger.info(f"\nRunning inference on {len(test_samples)} images...")
    logger.info(f"Detection threshold: {args.threshold}")
    
    predictions = []
    targets = []
    
    for image, gt_detections in tqdm(test_samples, desc="Inference"):
        # Run model
        pred_detections = model.predict(image, threshold=args.threshold)
        
        predictions.append(pred_detections)
        targets.append(gt_detections)
    
    logger.info(f"✓ Inference complete")
    
    # Calculate mAP using supervision
    logger.info("\n" + "="*80)
    logger.info("Calculating Mean Average Precision (mAP)")
    logger.info("="*80)
    
    map_metric = sv.metrics.MeanAveragePrecision()
    map_result = map_metric.update(predictions, targets).compute()
    
    # Print results
    logger.info("\nOverall mAP Metrics:")
    logger.info(f"  mAP @ IoU 0.50:0.95 (all)   : {map_result.map50_95:.4f}")
    logger.info(f"  mAP @ IoU 0.50              : {map_result.map50:.4f}")
    logger.info(f"  mAP @ IoU 0.75              : {map_result.map75:.4f}")
    
    if hasattr(map_result, 'small_objects'):
        logger.info(f"  mAP @ IoU 0.50:0.95 (small) : {map_result.small_objects.map50_95:.4f}")
    if hasattr(map_result, 'medium_objects'):
        logger.info(f"  mAP @ IoU 0.50:0.95 (medium): {map_result.medium_objects.map50_95:.4f}")
    if hasattr(map_result, 'large_objects'):
        logger.info(f"  mAP @ IoU 0.50:0.95 (large) : {map_result.large_objects.map50_95:.4f}")
    
    # Per-class mAP if available
    if hasattr(map_result, 'per_class') and category_names:
        logger.info("\nPer-Class mAP @ IoU 0.50:0.95:")
        for i, cat_name in enumerate(category_names):
            if i < len(map_result.per_class):
                logger.info(f"  {cat_name:30s}: {map_result.per_class[i]:.4f}")
    
    # Save results
    results_dict = {
        "model_id": args.hub_model_id,
        "dataset": args.dataset_name,
        "split": args.test_split,
        "num_samples": len(test_samples),
        "threshold": args.threshold,
        "metrics": {
            "mAP_50_95": float(map_result.map50_95),
            "mAP_50": float(map_result.map50),
            "mAP_75": float(map_result.map75),
        }
    }
    
    if category_names and hasattr(map_result, 'per_class'):
        results_dict["per_class_map"] = {
            cat_name: float(map_result.per_class[i])
            for i, cat_name in enumerate(category_names)
            if i < len(map_result.per_class)
        }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_path}")
    logger.info("="*80)
    logger.info(f"✅ Evaluation complete!")
    logger.info(f"   Final mAP @ IoU 0.50:0.95: {map_result.map50_95:.4f}")
    logger.info("="*80)
    
    # Cleanup
    cleanup_gpu_memory(model, verbose=True)


if __name__ == "__main__":
    main()

