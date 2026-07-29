# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Interactive post-processing playground.

Draw a mask on a reference image, run a model on a target image, then build a
post-processing pipeline step by step and watch how it changes the output.

Run with::

    python examples/gradio_postprocessing.py

Predictions use the numpy ``Prediction`` contract directly: ``masks`` is
``(N, H, W)``, ``scores`` and ``label_ids`` are ``(N,)``. Post-processors work
on torch tensors, so this example converts at that boundary and converts back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import gradio as gr
import numpy as np
import torch

from instantlearn.components.postprocessing import (
    BoxIoMNMS,
    BoxNMS,
    MaskIoMNMS,
    MaskNMS,
    MergePerClassMasks,
    MinimumAreaFilter,
    MorphologicalClosing,
    MorphologicalOpening,
    PostProcessor,
    PostProcessorPipeline,
    ScoreFilter,
    SoftNMS,
)
from instantlearn.components.sam.decoder import masks_to_boxes_traceable
from instantlearn.data import Sample
from instantlearn.data.base.prediction import Prediction
from instantlearn.models import Matcher
from instantlearn.visualizer import render_predictions, setup_colors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CATEGORY_ID = 0
CATEGORY_NAME = "object"

# Each entry builds a post-processor with sensible defaults. The UI adds steps
# by name, so the pipeline stays a simple ordered list of these keys.
STEP_FACTORIES: dict[str, Any] = {
    "MinimumAreaFilter": lambda: MinimumAreaFilter(min_area=100),
    "ScoreFilter": lambda: ScoreFilter(min_score=0.5),
    "MorphologicalOpening": lambda: MorphologicalOpening(kernel_size=3),
    "MorphologicalClosing": lambda: MorphologicalClosing(kernel_size=3),
    "MaskNMS": lambda: MaskNMS(iou_threshold=0.5),
    "BoxNMS": lambda: BoxNMS(iou_threshold=0.5),
    "MaskIoMNMS": lambda: MaskIoMNMS(iom_threshold=0.3),
    "BoxIoMNMS": lambda: BoxIoMNMS(iom_threshold=0.3),
    "SoftNMS": lambda: SoftNMS(sigma=0.5, score_threshold=0.1),
    "MergePerClassMasks": MergePerClassMasks,
}


@dataclass
class AppState:
    """Everything the callbacks share between interactions."""

    model: Matcher | None = None
    device: str = "cpu"
    target_image: np.ndarray | None = None
    raw_prediction: Prediction | None = None
    steps: list[str] = field(default_factory=list)


state = AppState()


def resolve_device(preference: str) -> str:
    """Pick a runtime device, falling back to CPU when the choice is unavailable."""
    if preference != "auto":
        return preference
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    return "cpu"


def extract_reference(editor_value: dict | None) -> tuple[np.ndarray, np.ndarray]:
    """Pull the image and the drawn mask out of a Gradio ImageEditor value.

    Args:
        editor_value: Value produced by ``gr.ImageEditor``.

    Returns:
        The RGB image as ``(H, W, 3)`` uint8 and the mask as ``(1, H, W)`` bool.

    Raises:
        ValueError: If no image was uploaded or nothing was drawn.
    """
    if not editor_value or editor_value.get("background") is None:
        msg = "Upload a reference image first."
        raise ValueError(msg)

    image = np.asarray(editor_value["background"])[..., :3].astype(np.uint8)

    layers = editor_value.get("layers") or []
    if not layers:
        msg = "Draw over the object you want to find."
        raise ValueError(msg)

    # Gradio layers are RGBA; anything the user painted has a non-zero alpha.
    mask = np.zeros(image.shape[:2], dtype=bool)
    for layer in layers:
        layer_array = np.asarray(layer)
        alpha = layer_array[..., 3] if layer_array.shape[-1] == 4 else layer_array.max(axis=-1)
        mask |= alpha > 0

    if not mask.any():
        msg = "The drawn mask is empty."
        raise ValueError(msg)

    return image, mask[None, ...]


def build_pipeline(steps: list[str]) -> PostProcessor | None:
    """Turn an ordered list of step names into a pipeline, or None when empty."""
    if not steps:
        return None
    return PostProcessorPipeline([STEP_FACTORIES[name]() for name in steps])


def postprocess(prediction: Prediction, pipeline: PostProcessor | None, device: str) -> Prediction:
    """Run a prediction through a pipeline and return a new Prediction.

    Post-processors operate on torch tensors, so masks, scores and labels are
    converted here and converted back afterwards. Boxes are recomputed from the
    cleaned masks, since filtering may drop or reshape instances.

    Args:
        prediction: Prediction to clean up.
        pipeline: Pipeline to apply, or ``None`` to return the input unchanged.
        device: Device to run the pipeline on.

    Returns:
        A new ``Prediction`` with post-processed masks, scores, labels and boxes.
    """
    if pipeline is None or prediction.masks.shape[0] == 0:
        return prediction

    masks = torch.as_tensor(prediction.masks, device=device).bool()
    scores = torch.as_tensor(prediction.scores, device=device).float()
    labels = torch.as_tensor(prediction.label_ids, device=device)

    new_masks, new_scores, new_labels = pipeline(masks, scores, labels)

    boxes = None
    if new_masks.shape[0] > 0:
        boxes = masks_to_boxes_traceable(new_masks).cpu().numpy()

    label_ids = new_labels.cpu().numpy()
    return Prediction(
        masks=new_masks.cpu().numpy(),
        scores=new_scores.cpu().numpy(),
        label_ids=label_ids,
        label_names=np.array([CATEGORY_NAME] * len(label_ids), dtype=object),
        boxes=boxes,
    )


def render(image: np.ndarray, prediction: Prediction) -> np.ndarray:
    """Draw a prediction over an image."""
    color_map = setup_colors({CATEGORY_ID: CATEGORY_NAME})
    return render_predictions(image, prediction, color_map)


def summarize(prediction: Prediction) -> str:
    """Describe a prediction in one line of markdown."""
    count = prediction.masks.shape[0]
    if count == 0:
        return "**0 instances**"
    areas = prediction.masks.reshape(count, -1).sum(axis=1)
    return (
        f"**{count} instances** &nbsp; "
        f"score min {prediction.scores.min():.3f} / max {prediction.scores.max():.3f} &nbsp; "
        f"area min {int(areas.min())} / max {int(areas.max())} px"
    )


def fit_model(editor_value: dict | None, device_choice: str) -> str:
    """Fit the model on the reference image and drawn mask."""
    try:
        image, masks = extract_reference(editor_value)
    except ValueError as error:
        return f"⚠️ {error}"

    state.device = resolve_device(device_choice)
    state.model = Matcher(device=state.device)
    state.model.fit(Sample(image=image, masks=masks))
    state.raw_prediction = None

    return f"✅ Fitted on {masks[0].sum()} px of reference mask, running on `{state.device}`."


def predict(target_image: np.ndarray | None) -> tuple[np.ndarray | None, str]:
    """Run the fitted model on the target image."""
    if state.model is None:
        return None, "⚠️ Fit the model first."
    if target_image is None:
        return None, "⚠️ Upload a target image."

    image = np.asarray(target_image)[..., :3].astype(np.uint8)
    prediction = state.model.predict(Sample(image=image))[0]

    state.target_image = image
    state.raw_prediction = prediction

    return render(image, prediction), summarize(prediction)


def apply_pipeline() -> tuple[np.ndarray | None, str]:
    """Apply the current pipeline to the last prediction."""
    if state.raw_prediction is None or state.target_image is None:
        return None, "⚠️ Run a prediction first."

    pipeline = build_pipeline(state.steps)
    processed = postprocess(state.raw_prediction, pipeline, state.device)

    before = state.raw_prediction.masks.shape[0]
    after = processed.masks.shape[0]
    delta = f"{before} → {after} instances"

    return render(state.target_image, processed), f"{summarize(processed)}<br>{delta}"


def add_step(step_name: str) -> str:
    """Append a step to the pipeline."""
    state.steps.append(step_name)
    return describe_pipeline()


def remove_last_step() -> str:
    """Drop the last step from the pipeline."""
    if state.steps:
        state.steps.pop()
    return describe_pipeline()


def clear_pipeline() -> str:
    """Remove every step."""
    state.steps.clear()
    return describe_pipeline()


def describe_pipeline() -> str:
    """Render the pipeline as a numbered markdown list."""
    if not state.steps:
        return "_Pipeline is empty. Predictions pass through unchanged._"
    return "\n".join(f"{index}. `{name}`" for index, name in enumerate(state.steps, start=1))


def build_ui() -> gr.Blocks:
    """Assemble the Gradio interface."""
    with gr.Blocks(title="Post-processing playground") as demo:
        gr.Markdown(
            "# Post-processing playground\n"
            "Draw a mask on a reference image, predict on a target image, "
            "then stack post-processing steps to clean up the result.",
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 1. Reference")
                reference_editor = gr.ImageEditor(
                    label="Draw over the object",
                    type="numpy",
                    brush=gr.Brush(colors=["#ff0000"], color_mode="fixed"),
                )
                device_choice = gr.Dropdown(
                    choices=["auto", "cpu", "cuda", "xpu"],
                    value="auto",
                    label="Device",
                )
                fit_button = gr.Button("Fit model", variant="primary")
                fit_status = gr.Markdown()

            with gr.Column():
                gr.Markdown("### 2. Target")
                target_input = gr.Image(label="Target image", type="numpy")
                predict_button = gr.Button("Predict", variant="primary")
                raw_output = gr.Image(label="Raw prediction")
                raw_summary = gr.Markdown()

        gr.Markdown("### 3. Post-processing")
        with gr.Row():
            with gr.Column(scale=1):
                step_choice = gr.Dropdown(
                    choices=list(STEP_FACTORIES),
                    value=next(iter(STEP_FACTORIES)),
                    label="Step",
                )
                with gr.Row():
                    add_button = gr.Button("Add")
                    undo_button = gr.Button("Undo")
                    clear_button = gr.Button("Clear")
                pipeline_view = gr.Markdown(describe_pipeline())
                apply_button = gr.Button("Apply pipeline", variant="primary")

            with gr.Column(scale=2):
                processed_output = gr.Image(label="After post-processing")
                processed_summary = gr.Markdown()

        fit_button.click(fit_model, [reference_editor, device_choice], fit_status)
        predict_button.click(predict, target_input, [raw_output, raw_summary])
        add_button.click(add_step, step_choice, pipeline_view)
        undo_button.click(remove_last_step, None, pipeline_view)
        clear_button.click(clear_pipeline, None, pipeline_view)
        apply_button.click(apply_pipeline, None, [processed_output, processed_summary])

    return demo


if __name__ == "__main__":
    build_ui().launch()
