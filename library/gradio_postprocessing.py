# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Interactive Gradio app for testing post-processing pipelines on Matcher predictions.

Launch
------
    cd library
    python gradio_postprocessing.py

Requires ``pip install gradio``.
"""

from __future__ import annotations

import colorsys
import copy
import json
import logging
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
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
    PostProcessorPipeline,
    ScoreFilter,
    SoftNMS,
    apply_postprocessing,
)
from instantlearn.data import Sample
from instantlearn.data.utils.image import read_image
from instantlearn.models import Matcher
from instantlearn.visualizer import setup_colors, visualize_single_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR / "examples" / "assets" / "coco"
TMP_DIR = Path(tempfile.mkdtemp(prefix="gradio_pp_"))

ALL_POSTPROCESSORS = [
    "BoxNMS",
    "MaskNMS",
    "MaskIoMNMS",
    "BoxIoMNMS",
    "SoftNMS",
    "MinimumAreaFilter",
    "MorphologicalOpening",
    "MorphologicalClosing",
    "ConnectedComponentFilter",
    "HoleFilling",
    "PanopticArgmaxAssignment",
    "MergePerClassMasks",
    "InstanceMerge",
]

PARAM_SCHEMA: dict[str, list[dict]] = {
    "BoxNMS": [
        {"name": "iou_threshold", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "MaskNMS": [
        {"name": "iou_threshold", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "MaskIoMNMS": [
        {"name": "iom_threshold", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
        {"name": "score_margin", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "area_ratio", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "BoxIoMNMS": [
        {"name": "iom_threshold", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
        {"name": "score_margin", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "area_ratio", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "SoftNMS": [
        {"name": "sigma", "default": 0.5, "min": 0.01, "max": 2.0, "step": 0.05},
        {"name": "score_threshold", "default": 0.1, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "MinimumAreaFilter": [
        {"name": "min_area", "default": 100, "min": 0, "max": 10000, "step": 10},
    ],
    "MorphologicalOpening": [
        {"name": "kernel_size", "default": 3, "min": 3, "max": 31, "step": 2},
    ],
    "MorphologicalClosing": [
        {"name": "kernel_size", "default": 3, "min": 3, "max": 31, "step": 2},
    ],
    "ConnectedComponentFilter": [
        {"name": "min_component_area", "default": 100, "min": 0, "max": 10000, "step": 10},
    ],
    "HoleFilling": [],
    "PanopticArgmaxAssignment": [
        {"name": "min_area", "default": 0, "min": 0, "max": 10000, "step": 10},
    ],
    "MergePerClassMasks": [],
    "InstanceMerge": [
        {"name": "iou_threshold", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05},
        {"name": "gap_pixels", "default": 0, "min": 0, "max": 50, "step": 1},
    ],
    "ScoreFilter": [
        {"name": "min_score", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    ],
}

CLASS_MAP: dict[str, type] = {
    "BoxNMS": BoxNMS,
    "MaskNMS": MaskNMS,
    "MaskIoMNMS": MaskIoMNMS,
    "BoxIoMNMS": BoxIoMNMS,
    "SoftNMS": SoftNMS,
    "MinimumAreaFilter": MinimumAreaFilter,
    "MorphologicalOpening": MorphologicalOpening,
    "MorphologicalClosing": MorphologicalClosing,
    "ConnectedComponentFilter": ConnectedComponentFilter,
    "HoleFilling": HoleFilling,
    "PanopticArgmaxAssignment": PanopticArgmaxAssignment,
    "MergePerClassMasks": MergePerClassMasks,
    "InstanceMerge": InstanceMerge,
    "ScoreFilter": ScoreFilter,
}

# Recommended pipeline (optimal ordering from analysis)
RECOMMENDED_PIPELINE_STEPS: list[list] = [
    ["ScoreFilter", {"min_score": 0.0}],
    ["MaskIoMNMS", {"iom_threshold": 0.8, "score_margin": 0.1}],
    ["MorphologicalOpening", {"kernel_size": 3}],
    ["ConnectedComponentFilter", {"min_component_area": 100}],
    ["MorphologicalClosing", {"kernel_size": 3}],
    ["HoleFilling", {}],
    ["MinimumAreaFilter", {"min_area": 50}],
    ["InstanceMerge", {"iou_threshold": 0.0, "gap_pixels": 5}],
]

# Current Matcher default pipeline
CURRENT_DEFAULT_STEPS: list[list] = [
    ["BoxNMS", {"iou_threshold": 0.1}],
    ["MergePerClassMasks", {}],
]


# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------


_state: dict = {
    "model": None,
    "raw_predictions": None,
    "target_image_tv": None,
}


def _select_device() -> str:
    """Auto-select the best available device."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _instance_color_map(n: int) -> dict[int, list[int]]:
    """Generate HSV-spaced unique colours for *n* instances."""
    cmap: dict[int, list[int]] = {}
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.85, 0.95)
        cmap[i] = [int(c * 255) for c in rgb]
    return cmap


def _visualize(
    image_tv: torch.Tensor,
    prediction: dict[str, torch.Tensor],
    instance_colors: bool = True,
) -> np.ndarray:
    """Create a visualisation overlay. Returns HWC RGB uint8 numpy array."""
    pred = {**prediction}
    n_masks = pred["pred_masks"].shape[0]
    if instance_colors and n_masks > 0:
        pred["pred_labels"] = torch.arange(n_masks)
        cmap = _instance_color_map(n_masks)
    else:
        cmap = setup_colors({0: "object"})
    return visualize_single_image(
        image=image_tv,
        prediction=pred,
        file_name="_gradio_tmp.png",
        output_folder=str(TMP_DIR),
        color_map=cmap,
    )


def _build_pipeline(steps: list[list]) -> PostProcessorPipeline | None:
    """Build a ``PostProcessorPipeline`` from ``[["Name", {params}], ...]``."""
    if not steps:
        return None
    processors = []
    for name, kwargs in steps:
        cls = CLASS_MAP.get(name)
        if cls is None:
            valid = ", ".join(CLASS_MAP.keys())
            msg = f"Unknown post-processor '{name}'. Valid names: {valid}"
            raise ValueError(msg)
        processors.append(cls(**kwargs))
    return PostProcessorPipeline(processors)


def _predictions_table(
    prediction: dict[str, torch.Tensor],
    title: str = "Masks",
) -> str:
    """Build a markdown table of per-mask results."""
    masks = prediction["pred_masks"]
    scores = prediction["pred_scores"]
    labels = prediction["pred_labels"]
    n = masks.shape[0]
    if n == 0:
        return f"**{title}:** No masks."
    rows = [
        f"**{title}** ({n} masks)\n",
        "| # | Label | Score | Area (px) |",
        "|---|-------|-------|-----------|",
    ]
    for i in range(n):
        area = int(masks[i].sum().item())
        score = scores[i].item()
        label = int(labels[i].item())
        rows.append(f"| {i + 1} | {label} | {score:.3f} | {area:,} |")
    return "\n".join(rows)


def _matched_tables(
    raw_pred: dict[str, torch.Tensor],
    proc_pred: dict[str, torch.Tensor],
) -> str:
    """Build side-by-side tables showing which raw masks survived post-processing.

    Uses IoU matching to link raw mask IDs to processed mask IDs.
    """
    raw_masks = raw_pred["pred_masks"]
    raw_scores = raw_pred["pred_scores"]
    raw_labels = raw_pred["pred_labels"]
    proc_masks = proc_pred["pred_masks"]
    proc_scores = proc_pred["pred_scores"]
    proc_labels = proc_pred["pred_labels"]
    n_raw = raw_masks.shape[0]
    n_proc = proc_masks.shape[0]

    # Match raw → processed via best IoU
    raw_to_proc: dict[int, int | None] = {}
    proc_matched: set[int] = set()
    if n_raw > 0 and n_proc > 0:
        raw_flat = raw_masks.flatten(1).float()  # (R, H*W)
        proc_flat = proc_masks.flatten(1).float()  # (P, H*W)
        intersection = raw_flat @ proc_flat.T  # (R, P)
        raw_area = raw_flat.sum(dim=1, keepdim=True)  # (R, 1)
        proc_area = proc_flat.sum(dim=1, keepdim=True).T  # (1, P)
        union = raw_area + proc_area - intersection  # (R, P)
        iou = intersection / union.clamp(min=1)  # (R, P)
        for r in range(n_raw):
            best_p = int(iou[r].argmax().item())
            best_val = iou[r, best_p].item()
            if best_val > 0.1:
                raw_to_proc[r] = best_p
                proc_matched.add(best_p)
            else:
                raw_to_proc[r] = None
    else:
        for r in range(n_raw):
            raw_to_proc[r] = None

    # Build raw table with fate column
    sections = []

    # --- Raw table ---
    raw_rows = [
        f"**Raw Predictions** ({n_raw} masks)\n",
        "| # | Label | Score | Area (px) | After PP |",
        "|---|-------|-------|-----------|----------|",
    ]
    for i in range(n_raw):
        area = int(raw_masks[i].sum().item())
        score = raw_scores[i].item()
        label = int(raw_labels[i].item())
        matched = raw_to_proc.get(i)
        if matched is not None:
            fate = f"→ #{matched + 1}"
        else:
            fate = "~~removed~~"
        raw_rows.append(f"| {i + 1} | {label} | {score:.3f} | {area:,} | {fate} |")
    sections.append("\n".join(raw_rows))

    # --- Processed table ---
    proc_rows = [
        f"**After Post-Processing** ({n_proc} masks)\n",
        "| # | Label | Score | Area (px) | Origin |",
        "|---|-------|-------|-----------|--------|",
    ]
    for j in range(n_proc):
        area = int(proc_masks[j].sum().item())
        score = proc_scores[j].item()
        label = int(proc_labels[j].item())
        origins = [str(r + 1) for r, p in raw_to_proc.items() if p == j]
        origin_str = ", ".join(origins) if origins else "new"
        proc_rows.append(f"| {j + 1} | {label} | {score:.3f} | {area:,} | ←#{origin_str} |")
    sections.append("\n".join(proc_rows))

    return "\n\n".join(sections)


def _pipeline_description(steps: list[list]) -> str:
    """Human-readable description of a pipeline."""
    if not steps:
        return "No post-processing"
    parts = []
    for name, kwargs in steps:
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        parts.append(f"{name}({params})" if params else f"{name}()")
    return " → ".join(parts)


# ---------------------------------------------------------------------------
# Core callbacks
# ---------------------------------------------------------------------------


def fit_model(ref_mask_data: dict | None) -> str:
    """Fit Matcher on a reference image with a brush-drawn mask."""
    if ref_mask_data is None:
        return "⚠ Upload a reference image first."

    bg = ref_mask_data.get("background", None)
    if bg is None:
        return "⚠ Upload a reference image into the editor."

    h, w = bg.shape[:2]

    # Save reference image (convert RGBA → RGB for JPEG)
    ref_path = TMP_DIR / "ref_image.jpg"
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(bg)
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    pil_img.save(str(ref_path))

    # Extract mask from drawn layers
    mask = None
    layers = ref_mask_data.get("layers", [])
    for layer in reversed(layers):
        if isinstance(layer, np.ndarray) and layer.ndim == 3:
            drawn = (layer[..., :3].sum(axis=-1) > 0).astype(np.uint8)
            if drawn.sum() > 0:
                mask = drawn
                break

    # Fallback: composite vs background diff
    if mask is None:
        composite = ref_mask_data.get("composite", None)
        if composite is not None and isinstance(composite, np.ndarray):
            diff = np.abs(composite.astype(float) - bg.astype(float)).sum(axis=-1)
            mask = (diff > 30).astype(np.uint8)

    if mask is None or mask.sum() == 0:
        return "⚠ Draw a mask on the reference image (paint over the object with the brush)."

    mask_path = TMP_DIR / "ref_mask.png"
    cv2.imwrite(str(mask_path), mask * 255)

    device = _select_device()
    logger.info("Fitting Matcher on device=%s  image=%dx%d  mask_area=%d", device, w, h, int(mask.sum()))

    model = Matcher(device=device, use_nms=False, merge_masks_per_class=False)
    ref_sample = Sample(image_path=str(ref_path), mask_paths=str(mask_path))
    model.fit(ref_sample)
    _state["model"] = model

    return f"✓ Model fitted — {w}×{h}, mask: {int(mask.sum()):,} px, device: {device}"


def predict_target(target_image: np.ndarray | None) -> tuple[np.ndarray | None, str]:
    """Run raw prediction (no post-processing) on the target image."""
    if _state["model"] is None:
        return None, "⚠ Fit the model first (Step 1)."
    if target_image is None:
        return None, "⚠ Upload a target image."

    target_path = TMP_DIR / "target_image.jpg"
    from PIL import Image as PILImage

    pil_img = PILImage.fromarray(target_image)
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    pil_img.save(str(target_path))

    logger.info("Running raw prediction on target …")
    raw_predictions = _state["model"].predict([str(target_path)])

    target_tv = read_image(str(target_path), as_tensor=True)
    _state["raw_predictions"] = raw_predictions
    _state["target_image_tv"] = target_tv

    n = raw_predictions[0]["pred_masks"].shape[0]
    scores = raw_predictions[0]["pred_scores"]
    lo, hi = scores.min().item(), scores.max().item()

    # Visualise raw predictions
    raw_vis = _visualize(target_tv, raw_predictions[0], instance_colors=True)
    status = f"✓ {n} raw masks — scores [{lo:.2f} .. {hi:.2f}]"
    return raw_vis, status


def apply_pipeline(
    mode: str,
    custom_pipeline_json: str,
    instance_colors: bool,
) -> tuple[np.ndarray | None, str, str]:
    """Apply the chosen pipeline and return side-by-side visualisation."""
    if _state["raw_predictions"] is None:
        return None, "⚠ Run prediction first (Step 2).", ""

    # Determine pipeline steps
    if mode == "Recommended":
        steps = RECOMMENDED_PIPELINE_STEPS
    elif mode == "Current Default":
        steps = CURRENT_DEFAULT_STEPS
    elif mode == "Custom":
        try:
            steps = json.loads(custom_pipeline_json)
            steps = [[s[0], s[1]] for s in steps]
        except Exception as exc:
            return None, f"⚠ Invalid JSON: {exc}", custom_pipeline_json
    else:
        steps = []

    try:
        pipeline = _build_pipeline(steps)
    except ValueError as exc:
        return None, f"⚠ {exc}", json.dumps(steps, indent=2)
    processed = apply_postprocessing(
        copy.deepcopy(_state["raw_predictions"]),
        pipeline,
    )

    # Visualise raw vs processed
    raw_vis = _visualize(_state["target_image_tv"], _state["raw_predictions"][0], instance_colors)
    proc_vis = _visualize(_state["target_image_tv"], processed[0], instance_colors)

    # Stack side by side with a thin separator
    h = max(raw_vis.shape[0], proc_vis.shape[0])
    sep = np.full((h, 4, 3), 200, dtype=np.uint8)

    def _pad_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] < target_h:
            pad = np.zeros((target_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
            return np.vstack([img, pad])
        return img

    combined = np.hstack([_pad_h(raw_vis, h), sep, _pad_h(proc_vis, h)])

    # Build info text
    description = _pipeline_description(steps)
    raw_pred = _state["raw_predictions"][0]
    proc_pred = processed[0]
    raw_n = raw_pred["pred_masks"].shape[0]
    proc_n = proc_pred["pred_masks"].shape[0]
    tables = _matched_tables(raw_pred, proc_pred)
    info = (
        f"**Pipeline:** {description}\n\n"
        f"**Raw:** {raw_n} masks → **After:** {proc_n} masks "
        f"({raw_n - proc_n} removed)\n\n"
        f"{tables}"
    )

    return combined, info, json.dumps(steps, indent=2)


def on_mode_change(mode: str) -> tuple:
    """Update JSON editor and toggle custom builder visibility on mode change."""
    if mode == "Recommended":
        steps = RECOMMENDED_PIPELINE_STEPS
    elif mode == "Current Default":
        steps = CURRENT_DEFAULT_STEPS
    else:
        steps = RECOMMENDED_PIPELINE_STEPS  # pre-fill custom with recommended

    json_str = json.dumps(steps, indent=2)
    custom_visible = mode == "Custom"
    return gr.update(value=json_str), gr.update(visible=custom_visible)


def add_step_to_pipeline(current_json: str, step_name: str) -> str:
    """Append a post-processor step to the custom pipeline JSON."""
    try:
        steps = json.loads(current_json)
    except Exception:
        steps = []

    params = {}
    for p in PARAM_SCHEMA.get(step_name, []):
        params[p["name"]] = p["default"]
    steps.append([step_name, params])
    return json.dumps(steps, indent=2)


def remove_last_step(current_json: str) -> str:
    """Remove the last step from the custom pipeline JSON."""
    try:
        steps = json.loads(current_json)
        if steps:
            steps.pop()
    except Exception:
        steps = []
    return json.dumps(steps, indent=2)


def clear_pipeline() -> str:
    """Clear all steps."""
    return json.dumps([], indent=2)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    # Discover example target images
    example_targets: list[str] = []
    if EXAMPLES_DIR.exists():
        for name in sorted(EXAMPLES_DIR.glob("*.jpg")):
            if "mask" not in name.stem:
                example_targets.append(str(name))

    with gr.Blocks(
        title="Post-Processing Explorer",
    ) as app:
        gr.Markdown(
            "# Post-Processing Explorer\n"
            "Visually test post-processing pipelines on Matcher predictions.\n\n"
            "**Workflow:** 1) Upload reference → draw mask → Fit  ·  "
            "2) Upload target → Predict  ·  "
            "3) Choose pipeline → Apply",
        )

        # ── Step 1: Reference ────────────────────────────────────────────
        with gr.Accordion("Step 1 — Reference Image + Mask", open=True):
            gr.Markdown(
                "Upload an image, then **paint over the object** with the red brush to define the reference mask.",
            )
            with gr.Row():
                ref_editor = gr.ImageEditor(
                    label="Reference (draw mask with brush)",
                    type="numpy",
                    brush=gr.Brush(colors=["#FF0000"], default_size=20),
                    eraser=gr.Eraser(default_size=20),
                    height=400,
                )
                fit_status = gr.Textbox(label="Status", interactive=False, lines=2)
            fit_btn = gr.Button("Fit Model", variant="primary")
            fit_btn.click(fn=fit_model, inputs=[ref_editor], outputs=[fit_status])

        # ── Step 2: Target ───────────────────────────────────────────────
        with gr.Accordion("Step 2 — Target Image", open=True):
            with gr.Row():
                target_input = gr.Image(label="Target image", type="numpy", height=400)
                predict_output = gr.Image(label="Raw predictions (masks + labels)", type="numpy", height=400)
            predict_status = gr.Textbox(label="Status", interactive=False, lines=1)
            if example_targets:
                gr.Examples(
                    examples=example_targets,
                    inputs=[target_input],
                    label="Example targets from COCO",
                )
            predict_btn = gr.Button("Predict (raw, no post-processing)", variant="primary")
            predict_btn.click(fn=predict_target, inputs=[target_input], outputs=[predict_output, predict_status])

        # ── Step 3: Post-processing ──────────────────────────────────────
        with gr.Accordion("Step 3 — Post-Processing", open=True):
            with gr.Row():
                # Left column: controls
                with gr.Column(scale=1):
                    mode_radio = gr.Radio(
                        choices=["Recommended", "Current Default", "Custom"],
                        value="Recommended",
                        label="Pipeline mode",
                    )
                    instance_colors_cb = gr.Checkbox(
                        value=True,
                        label="Instance colours (unique colour per mask)",
                    )

                    # Custom pipeline builder (initially hidden)
                    with gr.Group(visible=False) as custom_group:
                        gr.Markdown("### Custom Pipeline Builder")
                        with gr.Row():
                            step_dropdown = gr.Dropdown(
                                choices=ALL_POSTPROCESSORS,
                                value="MaskIoMNMS",
                                label="Add step",
                            )
                            add_btn = gr.Button("+ Add", size="sm")
                        with gr.Row():
                            remove_btn = gr.Button("− Remove Last", size="sm")
                            clear_btn = gr.Button("Clear All", size="sm")
                        gr.Markdown(
                            'Edit the JSON below to adjust parameters.\n\nFormat: `[["Name", {params}], ...]`',
                        )

                    pipeline_json = gr.Code(
                        language="json",
                        label="Pipeline (JSON)",
                        value=json.dumps(RECOMMENDED_PIPELINE_STEPS, indent=2),
                        lines=14,
                    )
                    apply_btn = gr.Button("Apply Pipeline", variant="primary")

                # Right column: results
                with gr.Column(scale=2):
                    result_image = gr.Image(
                        label="Raw (left)  ·  Post-processed (right)",
                        type="numpy",
                        height=500,
                    )
                    result_info = gr.Markdown(label="Results")

            # Wiring: mode change
            mode_radio.change(
                fn=on_mode_change,
                inputs=[mode_radio],
                outputs=[pipeline_json, custom_group],
            )
            # Wiring: custom builder buttons
            add_btn.click(
                fn=add_step_to_pipeline,
                inputs=[pipeline_json, step_dropdown],
                outputs=[pipeline_json],
            )
            remove_btn.click(fn=remove_last_step, inputs=[pipeline_json], outputs=[pipeline_json])
            clear_btn.click(fn=clear_pipeline, outputs=[pipeline_json])
            # Wiring: apply
            apply_btn.click(
                fn=apply_pipeline,
                inputs=[mode_radio, pipeline_json, instance_colors_cb],
                outputs=[result_image, result_info, pipeline_json],
            )

        # ── Reference card ───────────────────────────────────────────────
        with gr.Accordion("Pipeline Reference", open=False):
            gr.Markdown(
                "### Available Post-Processors\n\n"
                "| # | Name | Parameters | ONNX |\n"
                "|---|------|-----------|------|\n"
                "| 1 | **BoxNMS** | `iou_threshold` (0.5) | Yes |\n"
                "| 2 | **MaskNMS** | `iou_threshold` (0.5) | Yes |\n"
                "| 3 | **MaskIoMNMS** | `iom_threshold` (0.3) | Yes |\n"
                "| 4 | **BoxIoMNMS** | `iom_threshold` (0.3) | Yes |\n"
                "| 5 | **SoftNMS** | `sigma` (0.5), `score_threshold` (0.1) | Yes |\n"
                "| 6 | **MinimumAreaFilter** | `min_area` (100) | Yes |\n"
                "| 7 | **MorphologicalOpening** | `kernel_size` (3, odd) | Yes |\n"
                "| 8 | **MorphologicalClosing** | `kernel_size` (3, odd) | Yes |\n"
                "| 9 | **ConnectedComponentFilter** | `min_component_area` (100) | No |\n"
                "| 10 | **HoleFilling** | *(none)* | No |\n"
                "| 11 | **PanopticArgmaxAssignment** | `min_area` (0) | Yes |\n"
                "| 12 | **MergePerClassMasks** | *(none)* | Yes |\n\n"
                "### Recommended Pipeline\n\n"
                "`MaskIoMNMS(0.8) → Opening(3) → CCFilter(100) → "
                "Closing(3) → HoleFilling → MinArea(50)`\n\n"
                "### Custom JSON Format\n\n"
                "```json\n"
                '[\n  ["MaskIoMNMS", {"iom_threshold": 0.8}],\n'
                '  ["MorphologicalOpening", {"kernel_size": 3}],\n'
                '  ["MinimumAreaFilter", {"min_area": 50}]\n'
                "]\n```",
            )

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
