# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Example script demonstrating how to export and use OVMatcher."""

from pathlib import Path

from getiprompt.data.base import Batch
from getiprompt.data.folder import FolderDataset
from getiprompt.models import Matcher, OVMatcher
from getiprompt.utils.constants import Backend, SAMModelName

# Configuration
EXPORT_DIR = Path("./exports/matcher")
SAM_MODEL = SAMModelName.SAM_HQ_TINY
DATASET_ROOT = (
    Path(__file__).parent.parent / "library" / "tests" / "assets" / "fss-1000"
)


def export_matcher_to_openvino():
    """Export a Matcher model to OpenVINO format."""
    print("Step 1: Exporting Matcher model to OpenVINO...")

    # Create and export Matcher model
    matcher = Matcher(sam=SAM_MODEL)
    matcher.export(export_dir=EXPORT_DIR, backend=Backend.OPENVINO)

    print(f"✓ Model exported to {EXPORT_DIR}")
    print(f"  - Image encoder: {EXPORT_DIR / 'image_encoder.xml'}")
    print(f"  - SAM model: {EXPORT_DIR / 'exported_sam.xml'}")


def run_ov_matcher_inference():
    """Run inference using OVMatcher with exported models."""
    print("\nStep 2: Running inference with OVMatcher...")

    # Check if dataset exists
    if not DATASET_ROOT.exists():
        msg = f"Dataset not found at {DATASET_ROOT}. Please ensure the fss-1000 dataset is available."
        raise FileNotFoundError(msg)

    # Create dataset
    print(f"Loading dataset from {DATASET_ROOT}...")
    dataset = FolderDataset(
        root=DATASET_ROOT,
        categories=["apple", "basketball"],  # Use 2 categories for demonstration
        n_shots=1,
    )
    print(f"✓ Dataset loaded with categories: {dataset.categories}")

    # Get reference and target batches
    ref_dataset = dataset.get_reference_dataset()
    target_dataset = dataset.get_target_dataset()

    # Use first 2 reference samples
    ref_samples = [ref_dataset[i] for i in range(min(2, len(ref_dataset)))]
    ref_batch = Batch.collate(ref_samples)

    # Use first 2 target samples
    target_samples = [target_dataset[i] for i in range(min(2, len(target_dataset)))]
    target_batch = Batch.collate(target_samples)

    print(f"  - Reference samples: {len(ref_samples)}")
    print(f"  - Target samples: {len(target_samples)}")

    # Load OVMatcher from exported models
    ov_matcher = OVMatcher(
        model_folder=EXPORT_DIR,
        sam=SAM_MODEL,
        num_foreground_points=40,
        num_background_points=2,
        mask_similarity_threshold=0.38,
        device="CPU",  # Use "GPU" for Intel GPUs
    )
    print("✓ OVMatcher loaded successfully")

    # Run inference
    print("Running learn step...")
    ov_matcher.learn(ref_batch)
    print("✓ Learn step completed")

    print("Running infer step...")
    results = ov_matcher.infer(target_batch)
    print("✓ Infer step completed")

    # Print results
    print("\nInference Results:")
    print(f"  - Number of predictions: {len(results)}")
    for i, pred in enumerate(results):
        print(f"  - Prediction {i}:")
        print(f"    - Masks shape: {pred['pred_masks'].shape}")
        print(f"    - Labels shape: {pred['pred_labels'].shape}")
        if "pred_points" in pred:
            print(f"    - Points shape: {pred['pred_points'].shape}")
        if "pred_boxes" in pred:
            print(f"    - Boxes shape: {pred['pred_boxes'].shape}")


def main():
    """Main function to demonstrate OVMatcher usage."""
    print("=" * 60)
    print("OVMatcher Example: Export and Inference")
    print("=" * 60)

    # Step 1: Export model (only needed once)
    if not EXPORT_DIR.exists():
        export_matcher_to_openvino()
    else:
        print(f"Using existing exported models in {EXPORT_DIR}")

    # Step 2: Run inference with OVMatcher
    run_ov_matcher_inference()

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
