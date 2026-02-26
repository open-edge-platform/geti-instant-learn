# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-Processing Demo.

Demonstrates every post-processor in ``instantlearn.components.postprocessing``.

Setup
-----
Matcher is initialised with ``use_nms=False, merge_masks_per_class=False`` so
that ``predict()`` returns **raw** predictions (overlapping, per-instance masks
with no suppression).  Each post-processor is then applied individually to the
same raw output so its isolated effect is clearly visible.

Post-processors covered
-----------------------
 #  | Class                      | Category           | ONNX-safe
----|----------------------------|--------------------|----------
 1  | BoxNMS                     | Overlap (NMS)      | yes
 2  | MaskNMS                    | Overlap (NMS)      | yes
 3  | MaskIoMNMS                 | Overlap (NMS)      | yes
 4  | BoxIoMNMS                  | Overlap (NMS)      | yes
 5  | SoftNMS                    | Overlap (NMS)      | yes
 6  | ScoreFilter                | Filtering          | yes
 7  | MinimumAreaFilter          | Filtering          | yes
 8  | MorphologicalOpening       | Mask cleaning      | yes
 9  | MorphologicalClosing       | Mask cleaning      | yes
 10 | ConnectedComponentFilter   | Mask cleaning      | no
 11 | HoleFilling                | Mask cleaning      | no
 12 | PanopticArgmaxAssignment   | Overlap resolution | yes
 13 | MergePerClassMasks         | Merging            | yes
 14 | InstanceMerge              | Merging            | no

Usage
-----
    python postprocessing_demo.py            # saves plots to /tmp/postprocessing_demo/
    python postprocessing_demo.py --show     # also calls plt.show() interactively
"""

from __future__ import annotations

import argparse
import colorsys
import copy
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from instantlearn.components.postprocessing import (
    BoxIoMNMS,
    BoxNMS,
    ConnectedComponentFilter,
    HoleFilling,
    InstanceMerge,
    MaskIoMNMS,
    MaskNMS,
    MergePerClassMasks,
    MinimumAreaFilter,
    MorphologicalClosing,
    MorphologicalOpening,
    PanopticArgmaxAssignment,
    ScoreFilter,
    SoftNMS,
    apply_postprocessing,
)
from instantlearn.data import Sample
from instantlearn.data.utils.image import read_image
from instantlearn.models import Matcher
from instantlearn.visualizer import setup_colors, visualize_single_image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path("/tmp/postprocessing_demo")
# delete the existing output directory if it exists, to avoid confusion with old plots
if OUTPUT_DIR.exists():
    # delete using os
    import shutil

    shutil.rmtree(OUTPUT_DIR)


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COLOR_MAP: dict = {}
_PLOT_COUNTER = 0


def _select_device() -> str:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _instance_color_map(n_masks: int) -> dict[int, list[int]]:
    """Generate a unique colour for each mask index using evenly spaced hues."""
    cmap: dict[int, list[int]] = {}
    for i in range(n_masks):
        rgb = colorsys.hsv_to_rgb(i / max(n_masks, 1), 0.9, 0.95)
        cmap[i] = [int(c * 255) for c in rgb]
    return cmap


def _save_original_images(
    images: list,
    *,
    interactive: bool = False,
) -> None:
    """Save the original target images (no predictions) as plot 00."""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]
    for ax, img in zip(axes, images, strict=False):
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
    fig.suptitle("Original target images", fontsize=18, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "00_original_images.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Saved: {out_path.name}")
    if interactive:
        plt.show()
    else:
        plt.close(fig)


def show_predictions(
    images: list,
    predictions: list[dict[str, torch.Tensor]],
    title: str,
    *,
    filename: str,
    interactive: bool = False,
    instance_colors: bool = False,
) -> None:
    """Display a row of images with overlaid predictions and save to a single file."""
    global _PLOT_COUNTER  # noqa: PLW0603
    _PLOT_COUNTER += 1
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8))
    if n == 1:
        axes = [axes]
    for ax, img, pred in zip(axes, images, predictions, strict=False):
        if instance_colors:
            # Re-label each mask with a unique index so the visualiser
            # draws every instance in a different colour.
            pred = {**pred}  # shallow copy
            n_masks = pred["pred_masks"].shape[0]
            pred["pred_labels"] = torch.arange(n_masks)
            cmap = _instance_color_map(n_masks)
        else:
            cmap = COLOR_MAP
        vis = visualize_single_image(
            image=img,
            prediction=pred,
            file_name="_tmp.png",
            output_folder=str(OUTPUT_DIR),
            color_map=cmap,
        )
        n_masks = pred["pred_masks"].shape[0]
        ax.imshow(vis)
        ax.set_title(f"{n_masks} masks", fontsize=14)
        ax.axis("off")
    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout()
    out_path = OUTPUT_DIR / f"{_PLOT_COUNTER:02d}_{filename}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  Saved: {out_path.name}")
    if interactive:
        plt.show()
    else:
        plt.close(fig)


def print_summary(predictions: list[dict[str, torch.Tensor]], label: str = "") -> None:
    if label:
        print(f"\n--- {label} ---")
    for i, pred in enumerate(predictions):
        n = pred["pred_masks"].shape[0]
        scores = pred["pred_scores"]
        print(f"  Image {i}: {n} masks, scores [{scores.min():.2f}, {scores.max():.2f}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(interactive: bool = False, instance_colors: bool = False) -> None:
    global COLOR_MAP  # noqa: PLW0603
    device = _select_device()
    print(f"Using device: {device}")
    if instance_colors:
        print("Instance colouring: ON (each mask gets a unique colour)")

    # -- Colour map ----------------------------------------------------------
    COLOR_MAP = setup_colors({0: "object"})

    # -----------------------------------------------------------------------
    # Common setup: reference sample & target paths
    # -----------------------------------------------------------------------
    ref_sample = Sample(
        image_path=str(SCRIPT_DIR / "assets/coco/000000286874.jpg"),
        mask_paths=str(SCRIPT_DIR / "assets/coco/000000286874_mask.png"),
    )
    target_paths = [
        str(SCRIPT_DIR / "assets/coco/000000390341.jpg"),
        str(SCRIPT_DIR / "assets/coco/000000173279.jpg"),
        str(SCRIPT_DIR / "assets/coco/000000267704.jpg"),
    ]
    target_images = [read_image(p) for p in target_paths]

    # -----------------------------------------------------------------------
    # Original images (no predictions)
    # -----------------------------------------------------------------------
    _save_original_images(target_images, interactive=interactive)

    # -----------------------------------------------------------------------
    # Matcher default (use_nms=True, merge_masks_per_class=True)
    # -----------------------------------------------------------------------
    print("\n========== Matcher default (use_nms=True, merge_masks_per_class=True) ==========")
    default_model = Matcher(device=device, use_nms=True, merge_masks_per_class=True)
    default_model.fit(ref_sample)
    default_predictions = default_model.predict(target_paths)
    print_summary(default_predictions, "Matcher default")
    show_predictions(
        target_images,
        default_predictions,
        "Matcher default (use_nms=True, merge_masks_per_class=True)",
        filename="matcher_default_BoxNMS0.1_MergePerClass",
        interactive=interactive,
        instance_colors=instance_colors,
    )

    # -----------------------------------------------------------------------
    # Matcher (raw) — no NMS, no merging
    # -----------------------------------------------------------------------
    print("\n========== Initialising Matcher (use_nms=False, merge_masks_per_class=False) ==========")
    model = Matcher(device=device, use_nms=False, merge_masks_per_class=False)
    model.fit(ref_sample)
    raw_predictions = model.predict(target_paths)
    print_summary(raw_predictions, "Raw predictions (no post-processing)")
    show_predictions(
        target_images,
        raw_predictions,
        "Raw predictions (no post-processing)",
        filename="raw_predictions",
        interactive=interactive,
        instance_colors=instance_colors,
    )

    # ===================================================================
    # 1. BoxNMS
    # ===================================================================
    print("\n========== 1. BoxNMS ==========")
    for threshold in [0.5, 0.1]:
        pp = BoxNMS(iou_threshold=threshold)
        results = apply_postprocessing(copy.deepcopy(raw_predictions), pp)
        title = f"BoxNMS (iou_threshold={threshold})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"BoxNMS_iou{threshold}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 2. MaskNMS
    # ===================================================================
    print("\n========== 2. MaskNMS ==========")
    for threshold in [0.5, 0.3]:
        pp = MaskNMS(iou_threshold=threshold)
        results = apply_postprocessing(copy.deepcopy(raw_predictions), pp)
        title = f"MaskNMS (iou_threshold={threshold})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"MaskNMS_iou{threshold}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 3. MaskIoMNMS
    # ===================================================================
    print("\n========== 3. MaskIoMNMS ==========")
    for threshold in [0.5, 0.3]:
        pp = MaskIoMNMS(iom_threshold=threshold)
        results = apply_postprocessing(copy.deepcopy(raw_predictions), pp)
        title = f"MaskIoMNMS (iom_threshold={threshold})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"MaskIoMNMS_iom{threshold}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 4. BoxIoMNMS
    # ===================================================================
    print("\n========== 4. BoxIoMNMS ==========")
    for threshold in [0.9, 0.7, 0.5]:
        pp = BoxIoMNMS(iom_threshold=threshold)
        results = apply_postprocessing(copy.deepcopy(raw_predictions), pp)
        title = f"BoxIoMNMS (iom_threshold={threshold})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"BoxIoMNMS_iom{threshold}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 5. SoftNMS
    # ===================================================================
    print("\n========== 5. SoftNMS ==========")
    for sigma, score_thresh in [(0.5, 0.1), (0.3, 0.05)]:
        pp = SoftNMS(sigma=sigma, score_threshold=score_thresh)
        results = apply_postprocessing(copy.deepcopy(raw_predictions), pp)
        title = f"SoftNMS (sigma={sigma}, score_threshold={score_thresh})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"SoftNMS_sigma{sigma}_thresh{score_thresh}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # Pre-NMS baseline for cleaning post-processors (6-12)
    # ===================================================================
    print("\n========== Pre-NMS baseline for sections 6-12 ==========")
    print("Post-processors 6-12 clean masks *after* overlap removal.")
    print("We apply MaskIoMNMS(iom_threshold=0.3) as the baseline input.\n")

    nms_predictions = apply_postprocessing(
        copy.deepcopy(raw_predictions),
        MaskIoMNMS(iom_threshold=0.3),
    )
    print_summary(nms_predictions, "Baseline: MaskIoMNMS (iom_threshold=0.3)")
    show_predictions(
        target_images,
        nms_predictions,
        "Baseline: MaskIoMNMS (iom_threshold=0.3)",
        filename="baseline_MaskIoMNMS_iom0.3",
        interactive=interactive,
        instance_colors=instance_colors,
    )

    # ===================================================================
    # 6. ScoreFilter
    # ===================================================================
    print("\n========== 6. ScoreFilter ==========")
    for min_score in [0.0, 0.1, 0.3]:
        pp = ScoreFilter(min_score=min_score)
        results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
        title = f"ScoreFilter (min_score={min_score})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"ScoreFilter_score{min_score}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 7. MinimumAreaFilter
    # ===================================================================
    print("\n========== 7. MinimumAreaFilter ==========")
    for min_area in [100, 500, 2000]:
        pp = MinimumAreaFilter(min_area=min_area)
        results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
        title = f"MinimumAreaFilter (min_area={min_area})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"MinimumAreaFilter_area{min_area}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 8. MorphologicalOpening
    # ===================================================================
    print("\n========== 8. MorphologicalOpening ==========")
    for ks in [3, 7, 15]:
        pp = MorphologicalOpening(kernel_size=ks)
        results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
        title = f"MorphologicalOpening (kernel_size={ks})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"MorphologicalOpening_kernel{ks}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 9. MorphologicalClosing
    # ===================================================================
    print("\n========== 9. MorphologicalClosing ==========")
    for ks in [3, 7, 15]:
        pp = MorphologicalClosing(kernel_size=ks)
        results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
        title = f"MorphologicalClosing (kernel_size={ks})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"MorphologicalClosing_kernel{ks}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 10. ConnectedComponentFilter
    # ===================================================================
    print("\n========== 10. ConnectedComponentFilter ==========")
    for min_comp in [100, 500]:
        pp = ConnectedComponentFilter(min_component_area=min_comp)
        results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
        title = f"ConnectedComponentFilter (min_component_area={min_comp})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"ConnectedComponentFilter_area{min_comp}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 11. HoleFilling
    # ===================================================================
    print("\n========== 11. HoleFilling ==========")
    pp = HoleFilling()
    results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
    print_summary(results, "HoleFilling")
    show_predictions(
        target_images,
        results,
        "HoleFilling",
        filename="HoleFilling",
        interactive=interactive,
        instance_colors=instance_colors,
    )

    # ===================================================================
    # 12. PanopticArgmaxAssignment
    # ===================================================================
    print("\n========== 12. PanopticArgmaxAssignment ==========")
    for min_area in [0, 500]:
        pp = PanopticArgmaxAssignment(min_area=min_area)
        results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
        title = f"PanopticArgmaxAssignment (min_area={min_area})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"PanopticArgmax_area{min_area}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    # ===================================================================
    # 13. MergePerClassMasks
    # ===================================================================
    print("\n========== 13. MergePerClassMasks ==========")
    pp = MergePerClassMasks()
    results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
    print_summary(results, "MergePerClassMasks")
    show_predictions(
        target_images,
        results,
        "MergePerClassMasks",
        filename="MergePerClassMasks",
        interactive=interactive,
        instance_colors=instance_colors,
    )

    # ===================================================================
    # 14. InstanceMerge
    # ===================================================================
    print("\n========== 14. InstanceMerge ==========")
    for gap in [0, 5, 15]:
        pp = InstanceMerge(iou_threshold=0.0, gap_pixels=gap)
        results = apply_postprocessing(copy.deepcopy(nms_predictions), pp)
        title = f"InstanceMerge (gap_pixels={gap})"
        print_summary(results, title)
        show_predictions(
            target_images,
            results,
            title,
            filename=f"InstanceMerge_gap{gap}",
            interactive=interactive,
            instance_colors=instance_colors,
        )

    print(f"\nAll plots saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-processing demo")
    parser.add_argument("--show", action="store_true", help="Show plots interactively")
    parser.add_argument(
        "--instance-colors",
        action="store_true",
        help="Draw each mask in a unique colour (instance colouring)",
    )
    args = parser.parse_args()
    main(interactive=args.show, instance_colors=args.instance_colors)
