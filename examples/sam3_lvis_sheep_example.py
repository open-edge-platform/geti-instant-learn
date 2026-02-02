# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 Exemplar Feature Injection example using LVIS sheep dataset.

This script demonstrates cross-image visual prompting with SAM3 where:
1. Reference samples provide visual exemplars (images + masks) of sheep
2. SAM3 extracts visual features from reference regions
3. Target images are segmented using those features - NO boxes or text needed!

This validates that the Exemplar Feature Injection approach works across
different images with the same semantic category.

Requirements:
    - LVIS dataset downloaded to ./datasets/LVIS or specified path
    - LVIS v1 annotations (lvis_v1_val.json)
    - COCO images (val2017/ and/or train2017/)

Usage:
    python examples/sam3_lvis_sheep_example.py

    # With custom dataset path:
    python examples/sam3_lvis_sheep_example.py --lvis-root /path/to/LVIS

    # With different confidence threshold:
    python examples/sam3_lvis_sheep_example.py --confidence 0.2
"""

import argparse
from pathlib import Path

import torch
from torchmetrics.segmentation import MeanIoU

from getiprompt.data.base import Batch, Sample
from getiprompt.data.lvis import LVISDataset
from getiprompt.models import SAM3
from getiprompt.visualizer import visualize_single_image


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SAM3 Exemplar Feature Injection with LVIS sheep dataset"
    )
    parser.add_argument(
        "--lvis-root",
        type=Path,
        default=Path("~/datasets/lvis").expanduser(),
        help="Path to LVIS dataset root directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./output/sam3_lvis_sheep"),
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--n-shots",
        type=int,
        default=10,
        help="Number of reference shots to load (pool of available reference images)",
    )
    parser.add_argument(
        "--ref-index",
        type=int,
        default=1,
        help="Index of the reference image to use (0-based). Use different values to try different reference images.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=5,
        help="Maximum number of target images to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for inference",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "bf16"],
        help="Model precision (fp32 recommended for SAM3)",
    )
    return parser.parse_args()


def compare_all_methods():
    args = parse_args()

    comparison_dir = args.output_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = LVISDataset(
        root=args.lvis_root,
        split="val",
        categories=["sheep"],
        n_shots=args.n_shots,
    )

    ref_dataset = dataset.get_reference_dataset()
    target_dataset = dataset.get_target_dataset()

    n_targets = min(args.max_targets, len(target_dataset))
    ref_idx = args.ref_index % len(ref_dataset)
    print(f"  Using reference image index: {ref_idx} (of {len(ref_dataset)} available)")
    reference_batch = Batch.collate([ref_dataset[ref_idx]])
    target_batch = Batch.collate([target_dataset[i] for i in range(n_targets)])

    # Get category info
    sheep_cat_id = reference_batch.samples[0].category_ids[0]
    category_colors = {int(sheep_cat_id): (255, 0, 0)}

    print(f"\nConfiguration:")
    print(f"  Reference samples: {len(ref_dataset)}")
    print(f"  Target samples: {n_targets}")
    print(f"  Category: sheep (ID: {sheep_cat_id})")

    # Build category mapping for metrics
    category_id_to_index = {int(sheep_cat_id): 0}
    num_classes = 1

    # Initialize metrics for each method
    metrics_text = MeanIoU(num_classes=num_classes, include_background=True, per_class=True).to(args.device)
    
    all_text_preds = []

    # ========== Method 1: Text Prompting ==========
    print("\n" + "-" * 50)
    print("Method 1: Text Prompting ('sheep')")
    print("-" * 50)

    model_text = SAM3(device=args.device, precision=args.precision, confidence_threshold=args.confidence)

    for i, sample in enumerate(target_batch.samples):
        target_with_text = Sample(
            image=sample.image,
            categories=["sheep"],
            category_ids=[sheep_cat_id],
        )
        pred = model_text.predict(Batch.collate([target_with_text]))[0]
        all_text_preds.append(pred)
        n_masks = pred["pred_masks"].shape[0] if pred["pred_masks"].numel() > 0 else 0
        print(f"  Image {i}: Found {n_masks} instances")

        visualize_single_image(
            sample.image,
            pred,
            f"text_{i}.png",
            str(comparison_dir),
            category_colors,
        )


if __name__ == "__main__":
    # main()

    # Uncomment to run comparison:
    # compare_text_vs_exemplar()

    # Uncomment to compare all three methods:
    compare_all_methods()
