# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Export a fitted Matcher PyTorch model to OpenVINO IR.

The Matcher OpenVINO pipeline is split into **two** reusable IR sub-models so
the exported files are independent of any particular ``fit()`` call:

* ``encoder`` — ``image -> normalized patch embeddings``. Used for both the
  reference image (during ``fit()``) and the target image (during
  ``predict()``).
* ``head`` — ``(target_image, target_embeddings, ref_embeddings,
  masked_ref_embeddings, flatten_ref_masks, category_ids) -> (masks, scores,
  labels)``. This is the former ``MatcherInferenceGraph`` with the reference
  features promoted from frozen buffers to **inputs**, so a single exported
  head serves any reference the user later fits.

Conversion goes PyTorch -> ONNX -> OpenVINO IR (ONNX has far better operator
coverage for the ops Matcher uses, e.g. ``aten::pad`` / ``aten::unbind``).

See Also:
    - ``Matcher``: PyTorch model being exported.
    - ``MatcherOpenVINO``: OV sibling that calls :func:`export_matcher` on the fly.
    - ``export_sam3``: analogous script for SAM3.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from instantlearn.components.sam import SamDecoder, load_sam_model
from instantlearn.models.matcher.matcher import EncoderForwardFeaturesWrapper
from instantlearn.utils.constants import CompressionMode, SAMModelName

if TYPE_CHECKING:
    from instantlearn.models.matcher.matcher import Matcher
    from instantlearn.models.torch_base import ExportConfig

logger = logging.getLogger(__name__)

ENCODER_MODEL_NAME = "encoder"
HEAD_MODEL_NAME = "head"

_INT4_MODES = frozenset({CompressionMode.INT4_SYM, CompressionMode.INT4_ASYM})


class MatcherHeadGraph(nn.Module):
    """Traceable Matcher head: reference features are graph **inputs**.

    Unlike ``MatcherInferenceGraph`` (which bakes reference features as frozen
    buffers), this graph accepts them as inputs so the exported IR is reusable
    across ``fit()`` calls. The DINOv3 encoder is *not* part of this graph — the
    caller passes pre-computed ``target_embeddings`` from the separate encoder
    IR — so the encoder weights are stored only once (in ``encoder.xml``).
    """

    def __init__(
        self,
        prompt_generator: nn.Module,
        sam_decoder: SamDecoder,
        postprocessor: nn.Module | None = None,
    ) -> None:
        """Initialize the head graph.

        Args:
            prompt_generator: The Matcher ``BidirectionalPromptGenerator``.
            sam_decoder: The Matcher ``SamDecoder`` (export-friendly path).
            postprocessor: Optional exportable post-processor applied to the
                ``(masks, scores, labels)`` outputs.
        """
        super().__init__()
        self.prompt_generator = prompt_generator
        self.sam_decoder = sam_decoder
        self.add_module("export_postprocessor", postprocessor)

    def forward(
        self,
        target_image: torch.Tensor,
        target_embeddings: torch.Tensor,
        ref_embeddings: torch.Tensor,
        masked_ref_embeddings: torch.Tensor,
        flatten_ref_masks: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run prompt generation + SAM decoding for a single target image.

        Args:
            target_image: ``[1, 3, H, W]`` — used for the SAM image encoder and
                to derive the original size (the DINOv3 encoding is *not* redone
                here; ``target_embeddings`` is passed in).
            target_embeddings: ``[1, num_patches, embed_dim]`` from the encoder IR.
            ref_embeddings: ``[C, num_patches_total, embed_dim]``.
            masked_ref_embeddings: ``[C, embed_dim]``.
            flatten_ref_masks: ``[C, num_patches_total]``.
            category_ids: ``[C]`` integer category IDs.

        Returns:
            ``(masks [C, H, W], scores [C], labels [C])``.
        """
        feature_device = target_embeddings.device
        ref_embeddings = ref_embeddings.to(feature_device)
        masked_ref_embeddings = masked_ref_embeddings.to(feature_device)
        flatten_ref_masks = flatten_ref_masks.to(feature_device)
        category_ids = category_ids.to(feature_device)

        height = torch.scalar_tensor(target_image.shape[2], dtype=torch.long, device=feature_device)
        width = torch.scalar_tensor(target_image.shape[3], dtype=torch.long, device=feature_device)
        original_sizes = torch.stack([height, width], dim=0).unsqueeze(0)

        point_prompts, similarities = self.prompt_generator.forward(
            ref_embeddings,
            masked_ref_embeddings,
            flatten_ref_masks,
            category_ids,
            target_embeddings,
            original_sizes,
        )

        masks, scores, labels = self.sam_decoder.forward_export(
            target_image[0],
            category_ids,
            point_prompts[0],
            similarities[0],
        )

        if self.export_postprocessor is not None:
            masks, scores, labels = self.export_postprocessor(masks, scores, labels)

        return masks, scores, labels


def _fix_onnx_output_names(onnx_path: Path, expected_names: list[str]) -> None:  # noqa: C901
    """Rename ONNX graph outputs to ``expected_names`` in-place.

    Registered buffers / constants returned as outputs often get auto-generated
    names (e.g. ``'39982'``) from the ONNX tracer. This rewires node outputs,
    initializers, and graph outputs so the names match.

    Args:
        onnx_path: Path to the ONNX file to patch.
        expected_names: Desired output names, in order.
    """
    if not onnx_path.exists():
        return
    import onnx  # noqa: PLC0415

    model = onnx.load(str(onnx_path))
    rename_map: dict[str, str] = {}
    for output, expected in zip(model.graph.output, expected_names, strict=False):
        if output.name != expected:
            rename_map[output.name] = expected
    if not rename_map:
        return
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name in rename_map:
                node.output[i] = rename_map[name]
    for initializer in model.graph.initializer:
        if initializer.name in rename_map:
            initializer.name = rename_map[initializer.name]
    for output in model.graph.output:
        if output.name in rename_map:
            output.name = rename_map[output.name]
    onnx.save(model, str(onnx_path))


def _convert_onnx_to_ir(
    onnx_path: Path,
    ir_path: Path,
    output_names: list[str],
    compression: CompressionMode,
    reshape: dict | None = None,
) -> None:
    """Convert an ONNX file to OpenVINO IR with optional weight compression.

    Args:
        onnx_path: Source ONNX file.
        ir_path: Destination ``.xml`` path.
        output_names: Expected output tensor names, in order.
        compression: Weight compression mode for the IR.
        reshape: Optional ``{input_name: shape}`` static reshape applied before
            saving (improves GPU kernel selection for static inputs).
    """
    import openvino  # noqa: PLC0415

    core = openvino.Core()
    ov_model = core.read_model(str(onnx_path))

    for output, name in zip(ov_model.outputs, output_names, strict=False):
        output.tensor.set_names({name})

    if reshape:
        ov_model.reshape(reshape)

    if compression not in {CompressionMode.FP32, CompressionMode.FP16}:
        from instantlearn.utils.compression import compress_model  # noqa: PLC0415

        ov_model = compress_model(ov_model, mode=compression)

    openvino.save_model(ov_model, str(ir_path), compress_to_fp16=compression == CompressionMode.FP16)


@torch.no_grad()
def export_matcher(
    matcher: Matcher,
    output_dir: str | Path,
    config: ExportConfig | None = None,
) -> dict[str, Path]:
    """Export a Matcher to OpenVINO IR as ``encoder.xml`` + ``head.xml``.

    The export does **not** require the model to be fitted — reference features
    are inputs to the head graph, not baked constants.

    Args:
        matcher: The PyTorch ``Matcher`` to export.
        output_dir: Directory to write the IR files into (created if missing).
        config: Export options. ``None`` uses :class:`ExportConfig` defaults
            (``INT8_SYM`` compression). ``INT4`` compression is rejected because
            it produces noisy Matcher masks.

    Returns:
        Mapping ``{"encoder": encoder.xml, "head": head.xml}``.

    Raises:
        ValueError: If ``config.compression`` is an INT4 mode.
    """
    from instantlearn.models.torch_base import ExportConfig  # noqa: PLC0415

    config = config or ExportConfig()
    if config.compression in _INT4_MODES:
        msg = (
            "INT4 compressed Matcher models produce random noisy masks and are not accurate. "
            "Use INT8 compression (default) or FP16/FP32 for Matcher exports."
        )
        raise ValueError(msg)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_device = torch.device("cpu")

    # SAM-HQ-Tiny is non-deterministic under OpenVINO — fall back to SAM-HQ-base.
    segmenter = matcher.segmenter
    if matcher.sam_predictor.sam_model_name == SAMModelName.SAM_HQ_TINY:
        logger.warning(
            "SAM-HQ-Tiny is not supported for OpenVINO export (non-deterministic layers). "
            "Falling back to SAM-HQ-base for the exported head; weights will be downloaded if needed.",
        )
        fallback_predictor = load_sam_model(SAMModelName.SAM_HQ_BASE, device="cpu", precision="fp32")
        segmenter = SamDecoder(
            sam_predictor=fallback_predictor,
            confidence_threshold=matcher.segmenter.confidence_threshold,
            use_mask_refinement=matcher.segmenter.use_mask_refinement,
        )
    else:
        matcher.sam_predictor.sync_device(export_device, dtype=torch.float32)
        segmenter.device = matcher.sam_predictor.device

    input_size = matcher.encoder.input_size
    opset = config.opset

    # ----- Encoder IR -------------------------------------------------------
    encoder_graph = (
        EncoderForwardFeaturesWrapper(
            matcher.encoder._model.model,  # noqa: SLF001
            ignore_token_length=matcher.encoder._model.ignore_token_length,  # noqa: SLF001
            input_size=input_size,
        )
        .to(export_device)
        .float()
        .eval()
    )
    example_image = torch.randn(1, 3, input_size, input_size, device=export_device)
    target_embeddings = encoder_graph(example_image)  # [1, num_patches, embed_dim]

    encoder_onnx = output_dir / f"{ENCODER_MODEL_NAME}.onnx"
    logger.info("Exporting Matcher encoder to ONNX...")
    torch.onnx.export(
        encoder_graph,
        (example_image,),
        f=str(encoder_onnx),
        input_names=["image"],
        output_names=["embeddings"],
        dynamo=False,
        opset_version=opset,
    )
    _convert_onnx_to_ir(
        encoder_onnx,
        output_dir / f"{ENCODER_MODEL_NAME}.xml",
        output_names=["embeddings"],
        compression=config.compression,
        reshape={"image": [1, 3, input_size, input_size]},
    )

    # ----- Head IR ----------------------------------------------------------
    num_patches = target_embeddings.shape[1]
    embed_dim = target_embeddings.shape[2]
    # Example single-category, single-shot reference tensors (axes are dynamic).
    ref_embeddings = torch.randn(1, num_patches, embed_dim, device=export_device)
    masked_ref_embeddings = torch.randn(1, embed_dim, device=export_device)
    flatten_ref_masks = torch.randint(0, 2, (1, num_patches), device=export_device).float()
    category_ids = torch.zeros(1, dtype=torch.long, device=export_device)

    head_graph = (
        MatcherHeadGraph(
            prompt_generator=matcher.prompt_generator,
            sam_decoder=segmenter,
            postprocessor=matcher.postprocessor,
        )
        .to(export_device)
        .float()
        .eval()
    )

    head_onnx = output_dir / f"{HEAD_MODEL_NAME}.onnx"
    head_output_names = ["masks", "scores", "labels"]
    dynamic_axes = None
    if config.dynamic_shapes:
        dynamic_axes = {
            "ref_embeddings": {0: "num_categories", 1: "num_ref_patches"},
            "masked_ref_embeddings": {0: "num_categories"},
            "flatten_ref_masks": {0: "num_categories", 1: "num_ref_patches"},
            "category_ids": {0: "num_categories"},
            "masks": {0: "num_categories", 1: "height", 2: "width"},
            "scores": {0: "num_categories"},
            "labels": {0: "num_categories"},
        }
    logger.info("Exporting Matcher head to ONNX...")
    torch.onnx.export(
        head_graph,
        (
            example_image,
            target_embeddings,
            ref_embeddings,
            masked_ref_embeddings,
            flatten_ref_masks,
            category_ids,
        ),
        f=str(head_onnx),
        input_names=[
            "target_image",
            "target_embeddings",
            "ref_embeddings",
            "masked_ref_embeddings",
            "flatten_ref_masks",
            "category_ids",
        ],
        output_names=head_output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
        opset_version=opset,
    )
    _fix_onnx_output_names(head_onnx, head_output_names)
    _convert_onnx_to_ir(
        head_onnx,
        output_dir / f"{HEAD_MODEL_NAME}.xml",
        output_names=head_output_names,
        compression=config.compression,
        # Target image stays static; ref/target embedding axes remain dynamic.
        reshape={"target_image": [1, 3, input_size, input_size]},
    )

    if not config.keep_intermediate:
        encoder_onnx.unlink(missing_ok=True)
        head_onnx.unlink(missing_ok=True)

    # Metadata needed by MatcherOpenVINO to rebuild preprocessing / feature
    # extraction without the torch model.
    metadata = {
        "input_size": int(input_size),
        "patch_size": int(matcher.encoder.patch_size),
        "feature_size": int(matcher.encoder.feature_size),
        "embed_dim": int(embed_dim),
        "num_patches": int(num_patches),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    logger.info("Matcher OpenVINO export complete: %s", output_dir)
    return {
        ENCODER_MODEL_NAME: output_dir / f"{ENCODER_MODEL_NAME}.xml",
        HEAD_MODEL_NAME: output_dir / f"{HEAD_MODEL_NAME}.xml",
    }
