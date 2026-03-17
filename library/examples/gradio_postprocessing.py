# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Interactive Gradio app for testing post-processing pipelines on Matcher predictions.

Launch
------
    cd library
    python examples/gradio_postprocessing.py

Workflow: (1) Upload a reference image and paint a mask over the object,
(2) upload a target image and run raw prediction, (3) choose a post-processing
pipeline and compare raw vs processed results side-by-side.

Requires ``uv pip install gradio``.
"""

from __future__ import annotations

import atexit
import colorsys
import copy
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image as PILImage

from instantlearn.components.postprocessing import (
    BoxIoMNMS,
    BoxNMS,
    MaskIoMNMS,
    MaskNMS,
    MergePerClassMasks,
    MinimumAreaFilter,
    MorphologicalClosing,
    MorphologicalOpening,
    PostProcessorPipeline,
    ScoreFilter,
    SoftNMS,
    apply_postprocessing,
)
from instantlearn.data import Sample
from instantlearn.models import Matcher
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Constants

SCRIPT_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = SCRIPT_DIR / "assets" / "coco"
TMP_DIR = Path(tempfile.mkdtemp(prefix="gradio_pp_"))
atexit.register(shutil.rmtree, TMP_DIR, ignore_errors=True)

ALL_POSTPROCESSORS = [
    "ScoreFilter",
    "BoxNMS",
    "MaskNMS",
    "MaskIoMNMS",
    "BoxIoMNMS",
    "SoftNMS",
    "MinimumAreaFilter",
    "MorphologicalOpening",
    "MorphologicalClosing",
    "MergePerClassMasks",
]

PARAM_SCHEMA: dict[str, list[dict]] = {
    "ScoreFilter": [
        {"name": "min_score", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
    ],
    "BoxNMS": [
        {"name": "iou_threshold", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "MaskNMS": [
        {"name": "iou_threshold", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "MaskIoMNMS": [
        {"name": "iom_threshold", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
        {"name": "score_margin", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
        {"name": "area_ratio", "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05},
    ],
    "BoxIoMNMS": [
        {"name": "iom_threshold", "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
        {"name": "score_margin", "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
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
    "MergePerClassMasks": [],
}

CLASS_MAP: dict[str, type] = {
    "ScoreFilter": ScoreFilter,
    "BoxNMS": BoxNMS,
    "MaskNMS": MaskNMS,
    "MaskIoMNMS": MaskIoMNMS,
    "BoxIoMNMS": BoxIoMNMS,
    "SoftNMS": SoftNMS,
    "MinimumAreaFilter": MinimumAreaFilter,
    "MorphologicalOpening": MorphologicalOpening,
    "MorphologicalClosing": MorphologicalClosing,
    "MergePerClassMasks": MergePerClassMasks,
}

# Recommended pipeline — score filter, containment-aware NMS, morphology, area filter
RECOMMENDED_PIPELINE: list[tuple[str, dict]] = [
    ("ScoreFilter", {"min_score": 0.0}),
    ("MaskIoMNMS", {"iom_threshold": 0.8, "score_margin": 0.1}),
    ("MorphologicalOpening", {"kernel_size": 3}),
    ("MorphologicalClosing", {"kernel_size": 3}),
    ("MinimumAreaFilter", {"min_area": 50}),
]

# Current default pipeline
CURRENT_DEFAULT: list[tuple[str, dict]] = [
    ("MaskIoMNMS", {"iom_threshold": 0.5}),
    ("BoxIoMNMS", {"iom_threshold": 0.5}),
]


# App state


@dataclass
class _AppState:
    """Mutable state shared across Gradio callbacks."""

    model: Matcher | None = None
    raw_predictions: list[dict[str, torch.Tensor]] | None = None
    target_image: np.ndarray | None = None  # HWC RGB uint8
    raw_visualization: np.ndarray | None = None  # cached raw overlay
    instance_colors_used: bool = True
    show_scores_used: bool = True


_state = _AppState()


# Helpers


_VALID_DEVICES = frozenset({"auto", "cpu", "cuda", "xpu"})


def _resolve_device(preference: str = "auto") -> str:
    """Resolve a device preference to an actual PyTorch device string."""
    if preference not in _VALID_DEVICES:
        preference = "auto"
    if preference != "auto":
        return preference
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _instance_color_map(n: int) -> dict[int, list[int]]:
    """Generate HSV-spaced unique colours for *n* instances."""
    cmap: dict[int, list[int]] = {}
    for i in range(n):
        rgb = colorsys.hsv_to_rgb(i / max(n, 1), 0.9, 0.95)
        cmap[i] = [int(c * 255) for c in rgb]
    return cmap


def _visualize(
    image_rgb: np.ndarray,
    prediction: dict[str, torch.Tensor],
    instance_colors: bool = True,
    *,
    show_scores: bool = True,
) -> np.ndarray:
    """Render a prediction overlay onto an image. Returns HWC RGB uint8 array."""
    pred = {**prediction}
    n_masks = pred["pred_masks"].shape[0]
    if instance_colors and n_masks > 0:
        pred["pred_labels"] = torch.arange(n_masks)
        cmap = _instance_color_map(n_masks)
    else:
        cmap = setup_colors({0: "object"})
    return render_predictions(image_rgb, pred, cmap, show_scores=show_scores)


def _build_pipeline(steps: list) -> PostProcessorPipeline | None:
    """Build a ``PostProcessorPipeline`` from ``[("Name", {params}), ...]``.

    Returns None for an empty step list.

    Raises:
        ValueError: If a step name is not in ``CLASS_MAP``.
    """
    if not steps:
        return None
    processors = []
    for name, kwargs in steps:
        cls = CLASS_MAP.get(name)
        if cls is None:
            valid = ", ".join(CLASS_MAP)
            msg = f"Unknown post-processor '{name}'. Valid names: {valid}"
            raise ValueError(msg)
        processors.append(cls(**kwargs))
    return PostProcessorPipeline(processors)


def _steps_to_json(steps: list[tuple[str, dict]]) -> str:
    """Serialize pipeline steps to pretty JSON (tuples become lists)."""
    return json.dumps([list(s) for s in steps], indent=2)


def _matched_tables(
    raw_pred: dict[str, torch.Tensor],
    proc_pred: dict[str, torch.Tensor],
) -> str:
    """Build markdown tables showing which raw masks survived post-processing.

    Uses IoU matching to link raw mask indices to processed mask indices.
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
    if n_raw > 0 and n_proc > 0:
        raw_flat = raw_masks.flatten(1).float()
        proc_flat = proc_masks.flatten(1).float()
        intersection = raw_flat @ proc_flat.T
        raw_area = raw_flat.sum(dim=1, keepdim=True)
        proc_area = proc_flat.sum(dim=1, keepdim=True).T
        union = raw_area + proc_area - intersection
        iou = intersection / union.clamp(min=1)
        for r in range(n_raw):
            best_p = int(iou[r].argmax().item())
            best_val = iou[r, best_p].item()
            raw_to_proc[r] = best_p if best_val > 0.1 else None
    else:
        for r in range(n_raw):
            raw_to_proc[r] = None

    sections = []

    # Raw predictions table
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
        fate = f"→ #{matched + 1}" if matched is not None else "~~removed~~"
        raw_rows.append(f"| {i + 1} | {label} | {score:.3f} | {area:,} | {fate} |")
    sections.append("\n".join(raw_rows))

    # Processed predictions table
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


def _pipeline_description(steps: list) -> str:
    """Build a human-readable pipeline summary like ``A(x=1) → B(y=2)``."""
    if not steps:
        return "No post-processing"
    parts = []
    for name, kwargs in steps:
        params = ", ".join(f"{k}={v}" for k, v in kwargs.items())
        parts.append(f"{name}({params})" if params else f"{name}()")
    return " → ".join(parts)


# Core callbacks


def fit_model(ref_mask_data: dict | None, device: str) -> str:
    """Fit Matcher on a reference image with a brush-drawn mask.

    Args:
        ref_mask_data: Gradio ImageEditor output with 'background' and 'layers'.
        device: Device preference ('auto', 'cpu', 'cuda', 'xpu').

    Returns:
        Status message.
    """
    if ref_mask_data is None:
        return "⚠ Upload a reference image first."

    bg = ref_mask_data.get("background")
    if bg is None:
        return "⚠ Upload a reference image into the editor."

    h, w = bg.shape[:2]

    # Save reference image (convert RGBA → RGB for JPEG)
    ref_path = TMP_DIR / "ref_image.jpg"
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
        composite = ref_mask_data.get("composite")
        if composite is not None and isinstance(composite, np.ndarray):
            diff = np.abs(composite.astype(float) - bg.astype(float)).sum(axis=-1)
            mask = (diff > 30).astype(np.uint8)

    if mask is None or mask.sum() == 0:
        return "⚠ Draw a mask on the reference image (paint over the object with the brush)."

    mask_path = TMP_DIR / "ref_mask.png"
    cv2.imwrite(str(mask_path), mask * 255)

    resolved = _resolve_device(device)
    logger.info("Fitting Matcher on device=%s  image=%dx%d  mask_area=%d", resolved, w, h, int(mask.sum()))

    model = Matcher(device=resolved, postprocessor=None)
    ref_sample = Sample(image_path=str(ref_path), mask_paths=str(mask_path))
    model.fit(ref_sample)
    _state.model = model

    # Reset predictions from a previous session
    _state.raw_predictions = None
    _state.target_image = None
    _state.raw_visualization = None

    return f"✓ Model fitted -- {w}x{h}, mask: {int(mask.sum()):,} px, device: {resolved}"


def predict_target(
    target_image: np.ndarray | None,
    instance_colors: bool,
    show_scores: bool,
) -> tuple[np.ndarray | None, str]:
    """Run raw prediction on the target image (no post-processing).

    Args:
        target_image: Target image as HWC RGB uint8 numpy array.
        instance_colors: Whether to give each mask a unique colour.
        show_scores: Whether to show confidence scores on the overlay.

    Returns:
        Tuple of (visualization image, status message).
    """
    if _state.model is None:
        return None, "⚠ Fit the model first (Step 1)."
    if target_image is None:
        return None, "⚠ Upload a target image."

    target_path = TMP_DIR / "target_image.jpg"
    pil_img = PILImage.fromarray(target_image)
    if pil_img.mode == "RGBA":
        pil_img = pil_img.convert("RGB")
    pil_img.save(str(target_path))

    # Use the RGB numpy array for visualization
    target_rgb = np.array(pil_img)

    logger.info("Running raw prediction on target …")
    raw_predictions = _state.model.predict([str(target_path)])

    _state.raw_predictions = raw_predictions
    _state.target_image = target_rgb

    n = raw_predictions[0]["pred_masks"].shape[0]
    if n == 0:
        _state.raw_visualization = target_rgb.copy()
        return target_rgb.copy(), "✓ 0 raw masks found."

    scores = raw_predictions[0]["pred_scores"]
    lo, hi = scores.min().item(), scores.max().item()

    raw_vis = _visualize(target_rgb, raw_predictions[0], instance_colors, show_scores=show_scores)
    _state.raw_visualization = raw_vis
    _state.instance_colors_used = instance_colors
    _state.show_scores_used = show_scores

    return raw_vis, f"✓ {n} raw masks — scores [{lo:.2f} .. {hi:.2f}]"


def apply_pipeline(
    mode: str,
    custom_pipeline_json: str,
    instance_colors: bool,
    show_scores: bool,
) -> tuple[np.ndarray | None, np.ndarray | None, str, str]:
    """Apply the selected pipeline and return raw + processed visualizations.

    Args:
        mode: Pipeline mode ('Recommended', 'Current Default', or 'Custom').
        custom_pipeline_json: JSON string for custom pipeline steps.
        instance_colors: Whether to use unique per-mask colours.
        show_scores: Whether to render confidence scores on overlays.

    Returns:
        Tuple of (raw_image, processed_image, info_markdown, updated_json).
    """
    if _state.raw_predictions is None:
        return None, None, "⚠ Run prediction first (Step 2).", ""

    # Determine pipeline steps
    if mode == "Recommended":
        steps = [list(s) for s in RECOMMENDED_PIPELINE]
    elif mode == "Current Default":
        steps = [list(s) for s in CURRENT_DEFAULT]
    elif mode == "Custom":
        try:
            steps = json.loads(custom_pipeline_json)
            steps = [[s[0], s[1]] for s in steps]
        except (json.JSONDecodeError, TypeError, IndexError) as exc:
            return None, None, f"⚠ Invalid JSON: {exc}", custom_pipeline_json
    else:
        steps = []

    try:
        pipeline = _build_pipeline(steps)
    except ValueError as exc:
        return None, None, f"⚠ {exc}", json.dumps(steps, indent=2)

    processed = apply_postprocessing(
        copy.deepcopy(_state.raw_predictions),
        pipeline,
    )

    # Reuse cached raw visualization when rendering settings haven't changed
    need_rerender = (
        _state.raw_visualization is None
        or _state.instance_colors_used != instance_colors
        or _state.show_scores_used != show_scores
    )
    if need_rerender:
        raw_vis = _visualize(
            _state.target_image,
            _state.raw_predictions[0],
            instance_colors,
            show_scores=show_scores,
        )
        _state.raw_visualization = raw_vis
        _state.instance_colors_used = instance_colors
        _state.show_scores_used = show_scores
    else:
        raw_vis = _state.raw_visualization

    proc_vis = _visualize(
        _state.target_image,
        processed[0],
        instance_colors,
        show_scores=show_scores,
    )

    # Build summary
    raw_pred = _state.raw_predictions[0]
    proc_pred = processed[0]
    raw_n = raw_pred["pred_masks"].shape[0]
    proc_n = proc_pred["pred_masks"].shape[0]
    description = _pipeline_description(steps)
    tables = _matched_tables(raw_pred, proc_pred)
    info = (
        f"**Pipeline:** {description}\n\n"
        f"**Raw:** {raw_n} masks → **After:** {proc_n} masks "
        f"({raw_n - proc_n} removed)\n\n"
        f"{tables}"
    )

    return raw_vis, proc_vis, info, json.dumps(steps, indent=2)


def on_mode_change(mode: str) -> tuple:
    """Update the JSON editor and toggle the custom builder on mode change.

    Args:
        mode: Selected pipeline mode.

    Returns:
        Tuple of (json update, custom group visibility update).
    """
    if mode == "Recommended":
        steps = RECOMMENDED_PIPELINE
    elif mode == "Current Default":
        steps = CURRENT_DEFAULT
    else:
        steps = RECOMMENDED_PIPELINE  # pre-fill custom with recommended

    json_str = _steps_to_json(steps)
    custom_visible = mode == "Custom"
    return gr.update(value=json_str), gr.update(visible=custom_visible)


def add_step_to_pipeline(current_json: str, step_name: str) -> str:
    """Append a post-processor step with default params to the pipeline JSON.

    Args:
        current_json: Current pipeline JSON string.
        step_name: Name of the post-processor to add.

    Returns:
        Updated pipeline JSON string.
    """
    try:
        steps = json.loads(current_json)
    except (json.JSONDecodeError, TypeError):
        steps = []

    params = {p["name"]: p["default"] for p in PARAM_SCHEMA.get(step_name, [])}
    steps.append([step_name, params])
    return json.dumps(steps, indent=2)


def remove_last_step(current_json: str) -> str:
    """Remove the last step from the pipeline JSON.

    Args:
        current_json: Current pipeline JSON string.

    Returns:
        Updated pipeline JSON string.
    """
    try:
        steps = json.loads(current_json)
        if steps:
            steps.pop()
    except (json.JSONDecodeError, TypeError):
        steps = []
    return json.dumps(steps, indent=2)


def clear_pipeline() -> str:
    """Clear all pipeline steps, returning an empty JSON list."""
    return json.dumps([], indent=2)


# Gradio UI builders


def _build_reference_section() -> tuple[gr.ImageEditor, gr.Button, gr.Textbox]:
    """Build Step 1: reference image + mask editor."""
    gr.Markdown(
        "Upload an image, then **paint over the object** with the brush to define the reference mask.",
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
    return ref_editor, fit_btn, fit_status


def _build_target_section(
    example_targets: list[str],
) -> tuple[gr.Image, gr.Button, gr.Image, gr.Textbox]:
    """Build Step 2: target image upload and raw prediction."""
    with gr.Row():
        target_input = gr.Image(label="Target image", type="numpy", height=400)
        predict_output = gr.Image(label="Raw predictions", type="numpy", height=400)
    predict_status = gr.Textbox(label="Status", interactive=False, lines=1)
    if example_targets:
        gr.Examples(
            examples=example_targets,
            inputs=[target_input],
            label="Example targets from COCO",
        )
    predict_btn = gr.Button("Predict (raw, no post-processing)", variant="primary")
    return target_input, predict_btn, predict_output, predict_status


def _build_postprocessing_section() -> tuple:
    """Build Step 3: pipeline selection and comparison results.

    Returns:
        Tuple of all interactive components needed for event wiring.
    """
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
                    remove_btn = gr.Button("- Remove Last", size="sm")
                    clear_btn = gr.Button("Clear All", size="sm")
                gr.Markdown(
                    'Edit the JSON below to adjust parameters.\n\nFormat: `[["Name", {params}], ...]`',
                )

            pipeline_json = gr.Code(
                language="json",
                label="Pipeline (JSON)",
                value=_steps_to_json(RECOMMENDED_PIPELINE),
                lines=14,
            )
            apply_btn = gr.Button("Apply Pipeline", variant="primary")

        # Right column: results
        with gr.Column(scale=2):
            with gr.Row():
                raw_result = gr.Image(label="Raw predictions", type="numpy", height=450)
                proc_result = gr.Image(label="After post-processing", type="numpy", height=450)
            result_info = gr.Markdown(label="Results")

    return (
        mode_radio,
        instance_colors_cb,
        custom_group,
        step_dropdown,
        add_btn,
        remove_btn,
        clear_btn,
        pipeline_json,
        apply_btn,
        raw_result,
        proc_result,
        result_info,
    )


def _build_reference_card() -> None:
    """Build the collapsible pipeline reference card from PARAM_SCHEMA."""
    rows = []
    for i, name in enumerate(ALL_POSTPROCESSORS, 1):
        params = PARAM_SCHEMA.get(name, [])
        if params:
            param_str = ", ".join(f"`{p['name']}` ({p['min']}-{p['max']}, default {p['default']})" for p in params)
        else:
            param_str = "*(none)*"
        rows.append(f"| {i} | **{name}** | {param_str} |")

    table = "\n".join(rows)
    gr.Markdown(
        f"### Available Post-Processors\n\n"
        f"| # | Name | Parameters (range, default) |\n"
        f"|---|------|----------------------------|\n"
        f"{table}\n\n"
        f"### Recommended Pipeline\n\n"
        f"`ScoreFilter(0.0) → MaskIoMNMS(0.8) → Opening(3) → "
        f"Closing(3) → MinArea(50)`\n\n"
        f"### Custom JSON Format\n\n"
        f"```json\n"
        f'[\n  ["MaskIoMNMS", {{"iom_threshold": 0.8}}],\n'
        f'  ["MorphologicalOpening", {{"kernel_size": 3}}],\n'
        f'  ["MinimumAreaFilter", {{"min_area": 50}}]\n'
        f"]\n```",
    )


def build_app() -> gr.Blocks:
    """Build the Gradio Blocks app with a 3-step workflow.

    Returns:
        Configured ``gr.Blocks`` instance ready to launch.
    """
    example_targets: list[str] = []
    if EXAMPLES_DIR.exists():
        example_targets.extend(str(p) for p in sorted(EXAMPLES_DIR.glob("*.jpg")) if "mask" not in p.stem)

    with gr.Blocks(title="Post-Processing Explorer") as app:
        gr.Markdown(
            "# Post-Processing Explorer\n"
            "Visually test post-processing pipelines on Matcher predictions.\n\n"
            "**Workflow:** 1) Upload reference → draw mask → Fit  ·  "
            "2) Upload target → Predict  ·  "
            "3) Choose pipeline → Apply",
        )

        # Settings
        with gr.Accordion("Settings", open=False), gr.Row():
            device_radio = gr.Radio(
                choices=["auto", "cpu", "cuda", "xpu"],
                value="auto",
                label="Device (used when fitting the model)",
            )
            show_scores_cb = gr.Checkbox(
                value=True,
                label="Show confidence scores on overlays",
            )

        # Step 1: Reference
        with gr.Accordion("Step 1 — Reference Image + Mask", open=True):
            ref_editor, fit_btn, fit_status = _build_reference_section()

        # Step 2: Target
        with gr.Accordion("Step 2 — Target Image", open=True):
            target_input, predict_btn, predict_output, predict_status = _build_target_section(example_targets)

        # Step 3: Post-processing
        with gr.Accordion("Step 3 — Post-Processing", open=True):
            (
                mode_radio,
                instance_colors_cb,
                custom_group,
                step_dropdown,
                add_btn,
                remove_btn,
                clear_btn,
                pipeline_json,
                apply_btn,
                raw_result,
                proc_result,
                result_info,
            ) = _build_postprocessing_section()

        # Reference card
        with gr.Accordion("Pipeline Reference", open=False):
            _build_reference_card()

        # Event wiring
        fit_btn.click(
            fn=fit_model,
            inputs=[ref_editor, device_radio],
            outputs=[fit_status],
        )
        predict_btn.click(
            fn=predict_target,
            inputs=[target_input, instance_colors_cb, show_scores_cb],
            outputs=[predict_output, predict_status],
        )
        mode_radio.change(
            fn=on_mode_change,
            inputs=[mode_radio],
            outputs=[pipeline_json, custom_group],
        )
        add_btn.click(
            fn=add_step_to_pipeline,
            inputs=[pipeline_json, step_dropdown],
            outputs=[pipeline_json],
        )
        remove_btn.click(fn=remove_last_step, inputs=[pipeline_json], outputs=[pipeline_json])
        clear_btn.click(fn=clear_pipeline, outputs=[pipeline_json])
        apply_btn.click(
            fn=apply_pipeline,
            inputs=[mode_radio, pipeline_json, instance_colors_cb, show_scores_cb],
            outputs=[raw_result, proc_result, result_info, pipeline_json],
        )

    return app


# Entry point

if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
