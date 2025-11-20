#!/usr/bin/env python
# coding=utf-8
"""
Evaluate RF-DETR checkpoint on test set and compute mAP metrics.

Usage:
    python eval_rfdetr.py \
        --checkpoint_path /workspace/outputs/rfdetr-commonforms/checkpoint_best_ema.pth \
        --dataset_dir /workspace/outputs/rfdetr-commonforms/coco_format \
        --model_size medium

Or use auto-detection:
    python eval_rfdetr.py \
        --output_dir /workspace/outputs/rfdetr-commonforms \
        --model_size medium
"""

import argparse
import json
import logging
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_latest_ema_checkpoint(output_dir):
    """Find the latest EMA checkpoint in the output directory."""
    output_path = Path(output_dir)

    if not output_path.exists():
        logger.warning(f"Output directory does not exist: {output_dir}")
        return None

    # Priority list of checkpoint names to look for
    priority_checkpoints = [
        "checkpoint_best_ema.pth",
        "checkpoint_best_ema.pt",
        "checkpoint_ema.pth",
        "checkpoint_ema.pt",
        "checkpoint_best_total.pth",
        "checkpoint_best_total.pt",
        "checkpoint_best.pth",
        "checkpoint_best.pt",
    ]

    # Check for priority checkpoints first
    for checkpoint_name in priority_checkpoints:
        checkpoint_path = output_path / checkpoint_name
        if checkpoint_path.exists():
            logger.info(f"Found checkpoint: {checkpoint_path}")
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
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR checkpoint")

    # Checkpoint args
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to checkpoint file (e.g., checkpoint_best_ema.pth)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory to search for checkpoint (if checkpoint_path not specified)")

    # Model args
    parser.add_argument("--model_size", type=str, default="medium",
                       choices=["small", "medium", "large"],
                       help="RF-DETR model size")

    # Dataset args
    parser.add_argument("--dataset_dir", type=str, default=None,
                       help="Path to COCO format dataset directory")
    parser.add_argument("--split", type=str, default="test",
                       choices=["test", "valid", "train"],
                       help="Dataset split to evaluate on")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("="*80)
    logger.info("RF-DETR Checkpoint Evaluation")
    logger.info("="*80)

    # Find checkpoint
    if args.checkpoint_path:
        checkpoint_path = args.checkpoint_path
        if not Path(checkpoint_path).exists():
            logger.error(f"❌ Checkpoint not found: {checkpoint_path}")
            return 1
    elif args.output_dir:
        logger.info(f"Searching for checkpoint in: {args.output_dir}")
        checkpoint_path = find_latest_ema_checkpoint(args.output_dir)
        if not checkpoint_path:
            logger.error(f"❌ No checkpoint found in {args.output_dir}")
            return 1
    else:
        logger.error("❌ Must specify either --checkpoint_path or --output_dir")
        return 1

    logger.info(f"Using checkpoint: {checkpoint_path}")

    # Auto-detect dataset directory if not specified
    if not args.dataset_dir:
        if args.output_dir:
            # Try to find coco_format directory in output_dir
            coco_dir = Path(args.output_dir) / "coco_format"
            if coco_dir.exists():
                args.dataset_dir = str(coco_dir)
                logger.info(f"Auto-detected dataset directory: {args.dataset_dir}")
            else:
                logger.error(f"❌ Could not find coco_format directory in {args.output_dir}")
                logger.info("   Please specify --dataset_dir explicitly")
                return 1
        else:
            logger.error("❌ Must specify --dataset_dir")
            return 1

    # Verify dataset directory exists
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        logger.error(f"❌ Dataset directory not found: {args.dataset_dir}")
        return 1

    # Check if split exists
    split_dir = dataset_path / args.split
    if not split_dir.exists():
        logger.error(f"❌ Split '{args.split}' not found in {args.dataset_dir}")
        logger.info(f"   Available splits: {[d.name for d in dataset_path.iterdir() if d.is_dir()]}")
        return 1

    logger.info(f"Dataset directory: {args.dataset_dir}")
    logger.info(f"Evaluating on split: {args.split}")

    # Initialize RF-DETR model
    logger.info(f"\nInitializing RF-DETR {args.model_size} model...")
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
        return 1

    # Load checkpoint
    logger.info(f"\nLoading checkpoint: {checkpoint_path}")
    try:
        # RF-DETR models typically have a load method or can load via state_dict
        if hasattr(model, 'load'):
            model.load(checkpoint_path)
            logger.info("✓ Checkpoint loaded using model.load()")
        elif hasattr(model, 'model') and hasattr(model.model, 'load_state_dict'):
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if 'model' in checkpoint:
                model.model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.model.load_state_dict(checkpoint['state_dict'])
            else:
                model.model.load_state_dict(checkpoint)
            logger.info("✓ Checkpoint loaded using model.model.load_state_dict()")
        else:
            logger.error("❌ Don't know how to load checkpoint into this model")
            return 1
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        return 1

    # Run evaluation
    logger.info("\n" + "="*80)
    logger.info("RUNNING EVALUATION")
    logger.info("="*80)

    try:
        # Check if RF-DETR has an evaluate method
        if hasattr(model, 'evaluate'):
            logger.info(f"Running model.evaluate() on {args.split} split...")
            results = model.evaluate(
                dataset_dir=str(args.dataset_dir),
                split=args.split
            )
        elif hasattr(model, 'val'):
            logger.info(f"Running model.val() on {args.split} split...")
            results = model.val(
                dataset_dir=str(args.dataset_dir),
                split=args.split
            )
        else:
            logger.error("❌ Model doesn't have evaluate() or val() method")
            logger.info("   Available methods: " + ", ".join([m for m in dir(model) if not m.startswith('_')]))
            return 1

        logger.info("\n" + "="*80)
        logger.info("✅ EVALUATION COMPLETE!")
        logger.info("="*80)

        # Print results
        if results:
            logger.info("\nResults:")
            if isinstance(results, dict):
                for key, value in results.items():
                    logger.info(f"  {key}: {value}")

                # Save results to file
                if args.output_dir:
                    results_path = Path(args.output_dir) / f"eval_results_{args.split}.json"
                    with open(results_path, 'w') as f:
                        json.dump(results, f, indent=2)
                    logger.info(f"\n✓ Results saved to: {results_path}")
            else:
                logger.info(f"  {results}")
        else:
            logger.warning("⚠️  No results returned from evaluation")

        # Look for mAP metrics
        logger.info("\n" + "="*80)
        logger.info("mAP METRICS SUMMARY")
        logger.info("="*80)

        if isinstance(results, dict):
            # Common mAP metric keys
            map_keys = [
                'mAP', 'map', 'mAP_50_95', 'map_50_95', 'metrics/mAP_0.5:0.95',
                'mAP_50', 'map_50', 'metrics/mAP_0.5',
                'mAP_75', 'map_75', 'metrics/mAP_0.75',
            ]

            found_metrics = False
            for key in map_keys:
                if key in results:
                    logger.info(f"  {key}: {results[key]:.4f}")
                    found_metrics = True

            if not found_metrics:
                logger.info("  No standard mAP metrics found in results")
                logger.info("  All available metrics:")
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        logger.info(f"    {key}: {value}")

        logger.info("="*80)

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
