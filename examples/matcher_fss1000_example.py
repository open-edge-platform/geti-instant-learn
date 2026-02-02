# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Simple example of using Matcher model with fss-1000 dataset.

This script demonstrates how to:
1. Load the fss-1000 dataset
2. Create a Matcher model
3. Learn from reference samples (few-shot learning)
4. Run inference on target images
5. Compute metrics (Mean IoU)
6. Visualize the results
"""

from pathlib import Path

import torch
from torchmetrics.segmentation import MeanIoU

from getiprompt.data.base import Batch
from getiprompt.data.folder import FolderDataset
from getiprompt.models import Matcher, SoftMatcher
from getiprompt.utils.benchmark import convert_masks_to_one_hot_tensor
from getiprompt.visualizer import visualize_single_image


def main():
    # Path to fss-1000 dataset (adjust this path as needed)
    fss1000_root = Path(__file__).parent.parent / "library" / "tests" / "assets" / "fss-1000"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load the dataset
    print("Loading fss-1000 dataset...")
    dataset = FolderDataset(
        root=fss1000_root,
        categories=["apple", "basketball"],  # Categories to segment
        n_shots=2,
    )

    # 2. Get reference and target samples
    ref_dataset = dataset.get_reference_dataset()
    target_dataset = dataset.get_target_dataset()

    print(f"Categories: {dataset.categories}")
    print(f"Reference samples: {len(ref_dataset)}")
    print(f"Target samples: {len(target_dataset)}")

    # Create batches
    reference_batch = Batch.collate([ref_dataset[i] for i in range(len(ref_dataset))])
    target_batch = Batch.collate([target_dataset[i] for i in range(len(target_dataset))])

    # 3. Initialize the Matcher model
    print("\nInitializing Matcher model...")
    model = Matcher(
        device=device,
        precision="bf16",
        use_mask_refinement=True,
    )

    # 4. Learn from reference samples
    print("Learning from reference samples...")
    model.fit(reference_batch)

    # 5. Run inference on target images
    print("Running inference on target images...")
    predictions = model.predict(target_batch)

    # 6. Compute metrics (Mean IoU)
    print("\nComputing metrics...")
    num_classes = len(dataset.categories)
    category_id_to_index = {
        dataset.get_category_id(cat_name): idx for idx, cat_name in enumerate(dataset.categories)
    }

    # Initialize MeanIoU metric
    metrics = MeanIoU(
        num_classes=num_classes,
        include_background=True,
        per_class=True,
    ).to(device)

    # Convert predictions and ground truth to one-hot tensors
    batch_pred_tensors, batch_gt_tensors = convert_masks_to_one_hot_tensor(
        predictions=predictions,
        ground_truths=target_batch,
        num_classes=num_classes,
        category_id_to_index=category_id_to_index,
        device=device,
    )

    # Update metrics for each image
    for pred_tensor, gt_tensor in zip(batch_pred_tensors, batch_gt_tensors, strict=True):
        metrics.update(pred_tensor, gt_tensor)

    # Compute final IoU per class
    iou_per_class = metrics.compute()

    # Print metrics
    print("\n" + "=" * 50)
    print("METRICS RESULTS")
    print("=" * 50)
    print(f"  IoU = {iou_per_class}")
    print("=" * 50)

    # 7. Display and visualize results
    print(f"\nPredictions: {len(predictions)} images")
    for i, (pred, image) in enumerate(zip(predictions, target_batch.images, strict=True)):
        # Visualize the result
        category_colors = {0: (255, 0, 0), 1: (0, 255, 0)}  # Red for apple, Green for basketball
        visualize_single_image(
            image,
            pred,
            f"prediction_{i}.png",
            str(output_dir),
            category_colors,
        )
        

    print("\nDone!")


if __name__ == "__main__":
    main()

