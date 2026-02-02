# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 Comparison: HuggingFace vs Geti Implementation.

Compares outputs and accuracy of:
1. HuggingFace's Sam3Model (transformers library)
2. Geti's SAM3 wrapper (getiprompt library)

Both should produce identical results since Geti's implementation
uses the same underlying model architecture.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# HuggingFace imports
from transformers import Sam3Model, Sam3Processor

# Geti imports
from getiprompt.data.base import Batch, Sample
from getiprompt.data.lvis import LVISDataset
from getiprompt.models import SAM3


# Configuration
LVIS_ROOT = Path("~/datasets/lvis").expanduser()
OUTPUT_DIR = Path("./output/sam3_comparison")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE = 0.3
MAX_TARGETS = 10
CATEGORY = "sheep"


def compute_mask_iou(masks1: torch.Tensor, masks2: torch.Tensor) -> torch.Tensor:
    """Compute IoU between two sets of masks."""
    if masks1.numel() == 0 or masks2.numel() == 0:
        return torch.zeros((masks1.shape[0], masks2.shape[0]))

    masks1 = masks1.float().flatten(1)
    masks2 = masks2.float().flatten(1)

    intersection = torch.mm(masks1, masks2.t())
    area1 = masks1.sum(dim=1, keepdim=True)
    area2 = masks2.sum(dim=1, keepdim=True)
    union = area1 + area2.t() - intersection

    return intersection / (union + 1e-8)


def compute_binary_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Compute IoU between two binary masks."""
    pred = pred.bool()
    gt = gt.bool()
    intersection = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return (intersection / (union + 1e-8)).item()


def compare_on_lvis():
    """Compare HF and Geti SAM3 on LVIS dataset with ground truth evaluation."""
    print("=" * 60)
    print("SAM3 Comparison: HuggingFace vs Geti on LVIS")
    print("=" * 60)
    print(f"Category: {CATEGORY}")
    print(f"Confidence: {CONFIDENCE}")
    print(f"Device: {DEVICE}")
    print(f"Max targets: {MAX_TARGETS}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load LVIS dataset (only need target images, SAM3 is zero-shot)
    dataset = LVISDataset(
        root=LVIS_ROOT,
        split="val",
        categories=[CATEGORY],
        n_shots=1,  # Minimal, we only use targets
    )
    target_dataset = dataset.get_target_dataset()
    n_targets = min(MAX_TARGETS, len(target_dataset))

    print(f"Evaluating on {n_targets} images\n")

    # Initialize models
    print("Loading HuggingFace SAM3...")
    hf_model = Sam3Model.from_pretrained(
        "facebook/sam3",
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
    ).to(DEVICE).eval()
    hf_processor = Sam3Processor.from_pretrained("facebook/sam3")

    print("Loading Geti SAM3...")
    geti_model = SAM3(device=DEVICE, precision="fp32", confidence_threshold=CONFIDENCE)

    # Metrics storage
    hf_ious = []
    geti_ious = []
    hf_vs_geti_ious = []

    print("\n" + "-" * 60)
    for i in range(n_targets):
        sample = target_dataset[i]

        # Get ground truth mask
        gt_masks = sample.masks
        if gt_masks.numel() == 0:
            print(f"Image {i + 1}: Skipping - no ground truth")
            continue

        gt_combined = gt_masks.any(dim=0).long().to(DEVICE)

        # Prepare image for HF
        img_np = sample.image.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * 255).astype(np.uint8) if img_np.max() <= 1 else img_np.astype(np.uint8)
        pil_image = Image.fromarray(img_np)

        # Run HuggingFace SAM3
        img_inputs = hf_processor(images=pil_image, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            vision_embeds = hf_model.get_vision_features(img_inputs.pixel_values)
            text_inputs = hf_processor(text=CATEGORY, return_tensors="pt").to(DEVICE)
            outputs = hf_model(
                vision_embeds=vision_embeds,
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )

        hf_results = hf_processor.post_process_instance_segmentation(
            outputs,
            threshold=CONFIDENCE,
            mask_threshold=0.5,
            target_sizes=img_inputs.get("original_sizes").tolist(),
        )[0]

        hf_masks = hf_results["masks"]
        if hf_masks.shape[0] > 0:
            hf_combined = hf_masks.any(dim=0).float().to(DEVICE)
            # Resize to match GT if needed
            if hf_combined.shape != gt_combined.shape:
                hf_combined = F.interpolate(
                    hf_combined.unsqueeze(0).unsqueeze(0),
                    size=gt_combined.shape,
                    mode="nearest"
                ).squeeze() > 0.5
            hf_combined = hf_combined.long()
        else:
            hf_combined = torch.zeros_like(gt_combined)

        # Run Geti SAM3
        target_sample = Sample(
            image=sample.image,
            categories=[CATEGORY],
            category_ids=[0],
        )
        geti_pred = geti_model.predict(Batch.collate([target_sample]))[0]
        geti_masks = geti_pred["pred_masks"]
        if geti_masks.shape[0] > 0:
            geti_combined = geti_masks.any(dim=0).float().to(DEVICE)
            # Resize to match GT if needed
            if geti_combined.shape != gt_combined.shape:
                geti_combined = F.interpolate(
                    geti_combined.unsqueeze(0).unsqueeze(0),
                    size=gt_combined.shape,
                    mode="nearest"
                ).squeeze() > 0.5
            geti_combined = geti_combined.long()
        else:
            geti_combined = torch.zeros_like(gt_combined)

        # Compute IoU vs ground truth
        hf_iou = compute_binary_iou(hf_combined, gt_combined)
        geti_iou = compute_binary_iou(geti_combined, gt_combined)
        hf_ious.append(hf_iou)
        geti_ious.append(geti_iou)

        # Compare HF vs Geti directly
        if hf_masks.shape[0] > 0 and geti_masks.shape[0] > 0:
            # Resize masks to same size for comparison
            h, w = gt_combined.shape
            hf_resized = F.interpolate(hf_masks.float().unsqueeze(0), size=(h, w), mode="nearest").squeeze(0) > 0.5
            geti_resized = F.interpolate(geti_masks.float().unsqueeze(0).to(DEVICE), size=(h, w), mode="nearest").squeeze(0) > 0.5
            iou_matrix = compute_mask_iou(hf_resized.to(DEVICE), geti_resized)
            hf_geti_iou = iou_matrix.max(dim=1)[0].mean().item()
        else:
            hf_geti_iou = 1.0 if (hf_masks.shape[0] == 0 and geti_masks.shape[0] == 0) else 0.0

        hf_vs_geti_ious.append(hf_geti_iou)

        print(f"Image {i + 1}: HF={hf_masks.shape[0]} masks (IoU={hf_iou:.3f}), Geti={geti_masks.shape[0]} masks (IoU={geti_iou:.3f}), HF-Geti={hf_geti_iou:.4f}")

    # Final results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    hf_score = np.mean(hf_ious)
    geti_score = np.mean(geti_ious)
    avg_hf_geti = np.mean(hf_vs_geti_ious)

    print(f"\nAccuracy vs Ground Truth (mIoU):")
    print(f"  HuggingFace SAM3: {hf_score:.4f}")
    print(f"  Geti SAM3:        {geti_score:.4f}")

    print(f"\nHF vs Geti Agreement:")
    print(f"  Average IoU: {avg_hf_geti:.4f}")

    if avg_hf_geti > 0.99:
        print("\n✓ SUCCESS: Outputs are identical!")
    elif avg_hf_geti > 0.95:
        print("\n✓ GOOD: Outputs are very similar.")
    else:
        print(f"\n⚠ WARNING: Outputs differ (IoU: {avg_hf_geti:.4f})")

    return {
        "hf_miou": hf_score,
        "geti_miou": geti_score,
        "hf_vs_geti_iou": avg_hf_geti,
    }


if __name__ == "__main__":
    compare_on_lvis()
