# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Web-based detection UI using Gradio with InstantLearn SAM3.

Supports two inference modes:

1. **Visual Exemplar** — draw boxes or click points on one or more reference
   images, then detect similar objects on a target image.  Multiple reference
   images can be queued (n-shot) so the model aggregates richer exemplar
   features via cross-image concatenation.

2. **Classic (text-only)** — enter comma-separated category names (e.g.
   "sheep, car") and run open-vocabulary detection on the target image
   without any reference image.

Usage:
    python sam3_gradio.py
    python sam3_gradio.py -r ref.jpg -i target.jpg
    python sam3_gradio.py --device cpu --threshold 0.3
    python sam3_gradio.py --share  # public URL via Gradio
"""

import argparse
from time import perf_counter

import gradio as gr
import numpy as np
import torch

from instantlearn.data.base.sample import Sample
from instantlearn.data.utils.image import read_image
from instantlearn.models.sam3 import SAM3, Sam3PromptMode
from instantlearn.visualizer import render_predictions

# ---------------------------------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Web UI for SAM3 detection (InstantLearn)")
parser.add_argument("-i", "--image_path", default=None, type=str, help="Default target image path")
parser.add_argument("-r", "--ref_image_path", default=None, type=str, help="Default reference image path")
parser.add_argument("-d", "--device", default="cuda", type=str, help="Device (default: cuda)")
parser.add_argument("-t", "--threshold", default=0.3, type=float, help="Detection threshold (default: 0.3)")
parser.add_argument("--precision", default="fp32", choices=["fp32", "bf16"], help="Model precision")
parser.add_argument("--host", default="0.0.0.0", type=str, help="Host (default: 0.0.0.0)")
parser.add_argument("--port", default=7860, type=int, help="Port (default: 7860)")
parser.add_argument("--share", default=False, action="store_true", help="Create a public Gradio link")

args = parser.parse_args()

# ---------------------------------------------------------------------------------------------------------------------

print("\nLoading SAM3 model...", flush=True)
model = SAM3(
    device=args.device,
    confidence_threshold=args.threshold,
    precision=args.precision,
    prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
)
print(f"  Model ready on {args.device}/{args.precision}", flush=True)

# Default color map for the web UI (single "visual" class = green)
_WEB_COLOR_MAP: dict[int, list[int]] = {0: [0, 200, 0]}


def numpy_rgb_to_tensor(image_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 numpy (H, W, C) to CHW float tensor."""
    return torch.from_numpy(image_rgb).permute(2, 0, 1).float()


def _build_sample_from_shot(shot: dict) -> Sample:
    """Build a Sample from a shot dictionary.

    Args:
        shot: Dict with 'image' (np.ndarray RGB), 'boxes' (list), 'points' (list).

    Returns:
        Sample with prompts attached.
    """
    ref_tensor = numpy_rgb_to_tensor(shot["image"])
    boxes_px = shot["boxes"]
    points_px = shot["points"]

    sample = Sample(
        image=ref_tensor,
        categories=["visual"],
        category_ids=torch.tensor([0]),
    )

    if boxes_px:
        sample.bboxes = torch.tensor(boxes_px, dtype=torch.float32)
        sample.category_ids = torch.tensor([0] * len(boxes_px))
        sample.categories = ["visual"] * len(boxes_px)
    elif points_px:
        sample.points = torch.tensor(points_px, dtype=torch.float32)
        sample.category_ids = torch.tensor([0] * len(points_px))
        sample.categories = ["visual"] * len(points_px)

    return sample


def run_visual_exemplar(
    shots: list[dict],
    tgt_rgb: np.ndarray,
    det_threshold: float,
    drop_spatial_bias: bool,
) -> tuple[np.ndarray | None, str]:
    """Run visual exemplar detection with n-shot support.

    Each shot is a dict with 'image', 'boxes', 'points' from a different
    reference image. The model encodes all shots and concatenates their
    features for richer exemplar representation.

    Args:
        shots: List of reference shots, each containing image and prompts.
        tgt_rgb: Target image (RGB uint8).
        det_threshold: Score threshold.
        drop_spatial_bias: Whether to drop spatial bias in geometry encoder.

    Returns:
        (result_image, info_text)
    """
    model.prompt_mode = Sam3PromptMode.VISUAL_EXEMPLAR
    model.drop_spatial_bias = drop_spatial_bias
    model.postprocessor.threshold = det_threshold

    ref_samples = [_build_sample_from_shot(shot) for shot in shots]

    tgt_tensor = numpy_rgb_to_tensor(tgt_rgb)

    t1 = perf_counter()
    model.fit(ref_samples)
    tgt_sample = Sample(image=tgt_tensor)
    predictions = model.predict(tgt_sample)
    t2 = perf_counter()

    pred = predictions[0]
    num_det = pred["pred_masks"].shape[0]
    ms = round(1000 * (t2 - t1))

    result_img = render_predictions(tgt_rgb, pred, _WEB_COLOR_MAP)

    total_boxes = sum(len(s["boxes"]) for s in shots)
    total_points = sum(len(s["points"]) for s in shots)

    if num_det > 0:
        scores = pred["pred_boxes"][:, 4].cpu().numpy()
        info = (
            f"Detections: {num_det} | Scores: [{100 * scores.min():.0f}, {100 * scores.max():.0f}] | "
            f"Time: {ms} ms | {len(shots)}-shot: {total_boxes} boxes, {total_points} points"
        )
    else:
        info = f"No detections above threshold ({det_threshold:.2f}) | Time: {ms} ms | {len(shots)}-shot"

    return result_img, info


def run_text_prompt(
    tgt_rgb: np.ndarray,
    text_prompt: str,
    det_threshold: float,
) -> tuple[np.ndarray | None, str]:
    """Run classic text-prompt detection on a single image.

    Args:
        tgt_rgb: Target image (RGB uint8).
        text_prompt: Comma-separated category names.
        det_threshold: Score threshold.

    Returns:
        (result_image, info_text)
    """
    model.prompt_mode = Sam3PromptMode.CLASSIC
    model.postprocessor.threshold = det_threshold

    categories = [c.strip() for c in text_prompt.split(",") if c.strip()]
    if not categories:
        return tgt_rgb, "No text categories provided."

    tgt_tensor = numpy_rgb_to_tensor(tgt_rgb)
    sample = Sample(
        image=tgt_tensor,
        categories=categories,
        category_ids=list(range(len(categories))),
    )

    t1 = perf_counter()
    predictions = model.predict(sample)
    t2 = perf_counter()

    pred = predictions[0]
    num_det = pred["pred_masks"].shape[0]
    ms = round(1000 * (t2 - t1))

    result_img = render_predictions(tgt_rgb, pred, _WEB_COLOR_MAP)

    if num_det > 0:
        scores = pred["pred_boxes"][:, 4].cpu().numpy()
        info = (
            f"Detections: {num_det} | Scores: [{100 * scores.min():.0f}, {100 * scores.max():.0f}] | "
            f"Time: {ms} ms | Text: {categories}"
        )
    else:
        info = f"No detections above threshold ({det_threshold:.2f}) | Time: {ms} ms"

    return result_img, info


# ---------------------------------------------------------------------------------------------------------------------


def _extract_ref_rgb(ref_image) -> np.ndarray | None:
    """Extract RGB numpy array from a Gradio image component value."""
    if ref_image is None:
        return None
    if isinstance(ref_image, dict):
        return ref_image.get("composite", ref_image.get("background"))
    return ref_image


def _parse_boxes(text: str, ref_rgb: np.ndarray | None) -> list[list[int]]:
    """Parse box coordinates: 'x1,y1,x2,y2; x1,y1,x2,y2; ...'"""
    boxes: list[list[int]] = []
    if not text or not text.strip():
        return boxes
    for box_str in text.strip().split(";"):
        box_str = box_str.strip()
        if not box_str:
            continue
        coords = [float(x.strip()) for x in box_str.split(",")]
        if len(coords) != 4:
            continue
        x1, y1, x2, y2 = coords
        if ref_rgb is not None and all(0 <= c <= 1.0 for c in [x1, y1, x2, y2]):
            rh, rw = ref_rgb.shape[:2]
            x1, x2 = x1 * rw, x2 * rw
            y1, y2 = y1 * rh, y2 * rh
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def _parse_points(text: str, ref_rgb: np.ndarray | None) -> list[list[int]]:
    """Parse point coordinates: 'x,y; x,y; ...'"""
    points: list[list[int]] = []
    if not text or not text.strip():
        return points
    for pt_str in text.strip().split(";"):
        pt_str = pt_str.strip()
        if not pt_str:
            continue
        coords = [float(x.strip()) for x in pt_str.split(",")]
        if len(coords) != 2:
            continue
        x, y = coords
        if ref_rgb is not None and all(0 <= c <= 1.0 for c in [x, y]):
            rh, rw = ref_rgb.shape[:2]
            x, y = x * rw, y * rh
        points.append([int(x), int(y)])
    return points


def _format_shots_summary(shots: list[dict]) -> str:
    """Format a human-readable summary of queued shots."""
    if not shots:
        return "No shots queued. Add reference images with prompts."
    lines = [f"**{len(shots)} shot(s) queued:**"]
    for i, shot in enumerate(shots, 1):
        nb = len(shot["boxes"])
        np_ = len(shot["points"])
        h, w = shot["image"].shape[:2]
        prompts = []
        if nb:
            prompts.append(f"{nb} box{'es' if nb > 1 else ''}")
        if np_:
            prompts.append(f"{np_} point{'s' if np_ > 1 else ''} {shot['points']}")
        lines.append(f"  {i}. {w}x{h} image — {', '.join(prompts)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------


def on_add_shot(shots, ref_image, box_input_text, point_input_text):
    """Add the current reference image + prompts as a new shot."""
    ref_rgb = _extract_ref_rgb(ref_image)
    if ref_rgb is None:
        return shots, _format_shots_summary(shots), "Upload a reference image first."

    boxes = _parse_boxes(box_input_text, ref_rgb)
    points = _parse_points(point_input_text, ref_rgb)

    if not boxes and not points:
        return shots, _format_shots_summary(shots), "Add box or point prompts before adding a shot."

    shot = {"image": ref_rgb.copy(), "boxes": boxes, "points": points}
    shots = shots + [shot]  # new list to trigger Gradio state update
    return shots, _format_shots_summary(shots), f"Shot {len(shots)} added ({len(boxes)} boxes, {len(points)} points)."


def on_clear_shots(shots):
    """Clear all queued shots."""
    return [], _format_shots_summary([]), "Shots cleared."


def on_detect(
    shots,
    ref_image,
    tgt_image,
    box_input_text,
    point_input_text,
    text_prompt,
    det_threshold,
    drop_spatial_bias,
):
    """Main detection callback.

    If shots are queued, uses them all. Otherwise falls back to the current
    reference image + prompts as a single shot (1-shot convenience).
    """
    if tgt_image is None:
        return shots, _format_shots_summary(shots), None, "Please upload a target image."

    ref_rgb = _extract_ref_rgb(ref_image)
    have_text = text_prompt and text_prompt.strip()

    # Build the effective shot list
    effective_shots = list(shots)  # copy queued shots

    # If current ref image has prompts, include it as an additional shot
    if ref_rgb is not None:
        cur_boxes = _parse_boxes(box_input_text, ref_rgb)
        cur_points = _parse_points(point_input_text, ref_rgb)
        if cur_boxes or cur_points:
            effective_shots.append({"image": ref_rgb.copy(), "boxes": cur_boxes, "points": cur_points})

    have_visual = len(effective_shots) > 0

    if not have_text and not have_visual:
        return shots, _format_shots_summary(shots), tgt_image, "No prompts provided. Draw boxes/points or enter text."

    # Text-only → classic mode
    if have_text and not have_visual:
        result_img, info = run_text_prompt(tgt_image, text_prompt.strip(), det_threshold)
        return shots, _format_shots_summary(shots), result_img, info

    # Visual exemplar with n-shot
    result_img, info = run_visual_exemplar(effective_shots, tgt_image, det_threshold, drop_spatial_bias)
    return shots, _format_shots_summary(shots), result_img, info


def on_ref_click(ref_image, point_input_text, evt: gr.SelectData):
    """Handle click on reference image to add a point prompt."""
    if ref_image is None:
        return point_input_text

    ref_rgb = ref_image.get("composite", ref_image.get("background")) if isinstance(ref_image, dict) else ref_image
    if ref_rgb is None:
        return point_input_text

    px, py = evt.index  # pixel coords
    existing = point_input_text.strip() if point_input_text else ""
    new_point = f"{px},{py}"
    return f"{existing}; {new_point}" if existing else new_point


# ---------------------------------------------------------------------------------------------------------------------

default_ref_img = None
default_tgt_img = None
if args.ref_image_path:
    default_ref_img = read_image(args.ref_image_path, as_tensor=False)
if args.image_path:
    default_tgt_img = read_image(args.image_path, as_tensor=False)


with gr.Blocks(title="SAM3 Detection (InstantLearn)", theme=gr.themes.Soft()) as demo:
    # Hidden state for n-shot queue
    shots_state = gr.State([])

    gr.Markdown(f"## SAM3 Detection — InstantLearn  (`{args.device}/{args.precision}`)")
    gr.Markdown(
        "Draw exemplar prompts (boxes/points) on the **reference image**, "
        "and detections will be shown on the **target image**.\n\n"
        "**N-shot workflow:** upload a reference image, add prompts, click **Add Shot** to queue it. "
        "Repeat with different reference images. Then click **Run Detection**.\n"
        "For quick 1-shot use, just add prompts and click **Run Detection** directly.\n\n"
        "- Enter box coordinates as `x1,y1,x2,y2` (pixel or normalized 0-1). "
        "Separate multiple with `;`.\n"
        "- **Click** on the reference image to add point prompts.\n"
        "- **Text-only** mode: leave reference blank, enter comma-separated categories.",
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Reference Image (draw prompts here)")
            ref_image = gr.Image(
                label="Reference Image",
                value=default_ref_img,
                type="numpy",
                interactive=True,
            )
            box_input = gr.Textbox(
                label="Box Prompts (x1,y1,x2,y2; ...)",
                placeholder="e.g. 100,150,300,400  or  0.1,0.2,0.5,0.6",
                lines=1,
            )
            point_input = gr.Textbox(
                label="Point Prompts (x,y; ...)",
                placeholder="Click on reference image or type: 150,200; 300,400",
                lines=1,
            )
            with gr.Row():
                add_shot_btn = gr.Button("Add Shot", variant="secondary")
                clear_shots_btn = gr.Button("Clear Shots", variant="secondary")
            shots_display = gr.Markdown(value="No shots queued. Add reference images with prompts.")

            text_input = gr.Textbox(
                label="Text Prompt (classic mode, no reference needed)",
                placeholder="e.g. sheep, person, car",
                lines=1,
            )

        with gr.Column():
            gr.Markdown("### Target Image (detections shown here)")
            tgt_image = gr.Image(
                label="Target Image",
                value=default_tgt_img,
                type="numpy",
                interactive=True,
            )
            result_image = gr.Image(label="Detection Result", type="numpy", interactive=False)

    with gr.Row():
        det_threshold = gr.Slider(0, 1, value=args.threshold, step=0.01, label="Detection Threshold")
        drop_spatial_bias = gr.Checkbox(value=True, label="Drop Spatial Bias (recommended for cross-image)")

    with gr.Row():
        detect_btn = gr.Button("Run Detection", variant="primary", size="lg")
        clear_btn = gr.Button("Clear All", size="lg")

    info_text = gr.Textbox(label="Info", interactive=False, lines=1)

    # Wire up callbacks
    add_shot_btn.click(
        fn=on_add_shot,
        inputs=[shots_state, ref_image, box_input, point_input],
        outputs=[shots_state, shots_display, info_text],
    )

    clear_shots_btn.click(
        fn=on_clear_shots,
        inputs=[shots_state],
        outputs=[shots_state, shots_display, info_text],
    )

    detect_btn.click(
        fn=on_detect,
        inputs=[
            shots_state,
            ref_image,
            tgt_image,
            box_input,
            point_input,
            text_input,
            det_threshold,
            drop_spatial_bias,
        ],
        outputs=[shots_state, shots_display, result_image, info_text],
    )

    clear_btn.click(
        fn=lambda: ([], _format_shots_summary([]), "", "", "", None, "Cleared."),
        inputs=[],
        outputs=[shots_state, shots_display, box_input, point_input, text_input, result_image, info_text],
    )

    ref_image.select(
        fn=on_ref_click,
        inputs=[ref_image, point_input],
        outputs=[point_input],
    )


# ---------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nStarting web UI on http://{args.host}:{args.port}", flush=True)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
