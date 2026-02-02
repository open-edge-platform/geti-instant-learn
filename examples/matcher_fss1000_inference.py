# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Simple example of using InferenceModel (OpenVINO) with fss-1000 dataset.

This script demonstrates how to:
1. Export a Matcher model to OpenVINO format
2. Load the exported model with InferenceModel
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
from getiprompt.models.matcher import Matcher
from getiprompt.models.matcher.inference import InferenceModel
from getiprompt.utils.benchmark import convert_masks_to_one_hot_tensor
from getiprompt.utils.constants import Backend, SAMModelName
from getiprompt.visualizer import visualize_single_image


def main():
    # Path to fss-1000 dataset (adjust this path as needed)
    fss1000_root = Path(__file__).parent.parent / "library" / "tests" / "assets" / "fss-1000"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    export_dir = Path(__file__).parent.parent / "exports" / "matcher"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = "cpu"  # OpenVINO device: "CPU", "GPU", or "AUTO"
    print(f"Using OpenVINO device: {device}")

    # 1. Load the dataset
    print("Loading fss-1000 dataset...")
    dataset = FolderDataset(
        root=fss1000_root,
        categories=["apple", "basketball"],  # Categories to segment
        # n_shots=2,
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

    # 3. Export the model to OpenVINO format (if not already exported)
    print("\nExporting Matcher model to OpenVINO format...")
    print(f"Export directory: {export_dir}")
    
    # Create a temporary Matcher model for export
    export_model = Matcher(
        device="cpu",  # Use CPU for export
        precision="fp32",
    )
    
    # Export to OpenVINO
    export_model.export(export_dir=export_dir, backend=Backend.OPENVINO)
    print("Export complete!")

    # 4. Initialize the InferenceModel (OpenVINO)
    print("\nInitializing InferenceModel (OpenVINO)...")
    model = InferenceModel(
        model_folder=export_dir,
        sam=SAMModelName.SAM_HQ_TINY,
        device=device,
        precision="fp32",
    )

    # 5. Learn from reference samples
    print("Learning from reference samples...")
    model.fit(reference_batch)

    # 6. Run inference on target images
    print("Running inference on target images...")
    predictions = model.predict(target_batch)

    # 7. Compute metrics (Mean IoU)
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
    ).to("cpu")  # Metrics on CPU

    # Convert predictions and ground truth to one-hot tensors
    batch_pred_tensors, batch_gt_tensors = convert_masks_to_one_hot_tensor(
        predictions=predictions,
        ground_truths=target_batch,
        num_classes=num_classes,
        category_id_to_index=category_id_to_index,
        device="cpu",
    )

    # Update metrics for each image
    for pred_tensor, gt_tensor in zip(batch_pred_tensors, batch_gt_tensors, strict=True):
        metrics.update(pred_tensor, gt_tensor)

    # Compute final IoU per class
    iou_per_class = metrics.compute()

    # Print metrics
    print(iou_per_class)


if __name__ == "__main__":
    main()

