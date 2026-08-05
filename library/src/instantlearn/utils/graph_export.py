# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Shared ONNX / OpenVINO export for traceable inference graphs.

Models such as Matcher, SoftMatcher and PerDino differ only in how they turn a
target image into prompts.  Once that is wrapped in a traceable ``nn.Module``
returning ``(masks, scores, labels)``, the export itself is identical, so it
lives here instead of being duplicated per model.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from instantlearn.utils.constants import Backend, CompressionMode

if TYPE_CHECKING:
    from pathlib import Path

    from torch import nn

logger = logging.getLogger(__name__)

OUTPUT_NAMES = ["masks", "scores", "labels"]


def fix_onnx_output_names(onnx_path: Path, expected_names: list[str]) -> None:  # noqa: C901
    """Ensure ONNX graph outputs have the expected names.

    Registered buffers returned as outputs often get auto-generated names
    (e.g. '39982') because the ONNX tracer treats them as graph constants.
    Renames outputs in-place using the ONNX protobuf, also updating all
    internal node references and initializers so the graph stays valid.

    Args:
        onnx_path: Path to the ONNX file to rewrite in place.
        expected_names: Desired output names, in graph output order.
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
    # Update node outputs that feed into graph outputs.
    for node in model.graph.node:
        for i, name in enumerate(node.output):
            if name in rename_map:
                node.output[i] = rename_map[name]
    # Update initializers (registered buffers appear here).
    for initializer in model.graph.initializer:
        if initializer.name in rename_map:
            initializer.name = rename_map[initializer.name]
    # Update the graph output names.
    for output in model.graph.output:
        if output.name in rename_map:
            output.name = rename_map[output.name]
    onnx.save(model, str(onnx_path))


def _export_onnx(graph: nn.Module, target_image: torch.Tensor, onnx_path: Path, *, dynamic: bool) -> None:
    """Trace ``graph`` to ONNX, retrying with external data if it exceeds 2GiB.

    Raises:
        RuntimeError: If ONNX export fails for a reason other than the size limit.
    """
    dynamic_axes = (
        {
            "target_image": {2: "height", 3: "width"},
            "masks": {0: "num_masks", 1: "height", 2: "width"},
            "scores": {0: "num_masks"},
            "labels": {0: "num_masks"},
        }
        if dynamic
        else None
    )
    kwargs = {
        "args": (target_image,),
        "input_names": ["target_image"],
        "output_names": OUTPUT_NAMES,
        "dynamic_axes": dynamic_axes,
        "dynamo": False,
    }
    try:
        torch.onnx.export(graph, f=onnx_path, **kwargs)
    except RuntimeError as onnx_err:
        if "2GiB" not in str(onnx_err) and "protobuf" not in str(onnx_err):
            raise
        # Large models (e.g. SAM-HQ ViT-H ~2.6GB) exceed the protobuf limit.
        # Re-export with a string path so ONNX writes external data files.
        logger.info("Model exceeds ONNX 2GiB limit, re-exporting with external data")
        torch.onnx.export(graph, f=str(onnx_path), **kwargs)


def export_inference_graph(
    graph: nn.Module,
    export_dir: Path,
    model_name: str,
    input_size: int,
    backend: str | Backend,
    compression: CompressionMode,
    device: torch.device,
) -> Path:
    """Export a traceable inference graph to ONNX or OpenVINO.

    Args:
        graph: Traceable module mapping ``[1, 3, H, W]`` to ``(masks, scores, labels)``.
        export_dir: Directory to write artifacts into.
        model_name: Base filename, e.g. ``"matcher"`` yields ``matcher.xml``.
        input_size: Encoder input resolution, used for the trace-time dummy input.
        backend: :class:`~instantlearn.utils.constants.Backend` to export to.
        compression: Weight compression mode, applied for OpenVINO only.
        device: Device to build the trace-time input on.

    Returns:
        Path to the exported artifact (``.onnx`` or ``.xml``), or *export_dir*
        for unrecognised backends.

    Raises:
        ImportError: If OpenVINO is selected but not installed.
    """
    target_image = torch.randn(1, 3, input_size, input_size, device=device)

    if Backend(backend) == Backend.ONNX:
        onnx_path = export_dir / f"{model_name}.onnx"
        _export_onnx(graph, target_image, onnx_path, dynamic=True)
        fix_onnx_output_names(onnx_path, OUTPUT_NAMES)
        return onnx_path

    if Backend(backend) == Backend.OPENVINO:
        try:
            import openvino  # noqa: PLC0415
        except ImportError as e:
            msg = "OpenVINO is not installed. Please install it to use OpenVINO export."
            raise ImportError(msg) from e

        # Export to ONNX first, then convert. Direct PyTorch → OpenVINO conversion
        # fails on many ops (aten::pad, aten::unbind, ...); the ONNX frontend has
        # much better operator coverage.
        #
        # The graph is traced *statically* (no dynamic_axes) because dynamic axes
        # cause infer-time broadcast mismatches during GPU shape inference. Code
        # inside the graph must therefore never read a tensor's shape into a Python
        # int, or the trace-time value gets baked in as a constant.
        onnx_path = export_dir / f"{model_name}.onnx"
        _export_onnx(graph, target_image, onnx_path, dynamic=False)

        core = openvino.Core()
        if onnx_path.exists():
            try:
                ov_model = core.read_model(str(onnx_path))
            except RuntimeError:
                ov_model = openvino.convert_model(graph, example_input=target_image)
        else:
            ov_model = openvino.convert_model(graph, example_input=target_image)

        # Registered buffers returned as outputs get auto-generated names from the
        # ONNX tracer; restore the documented ones.
        for output, name in zip(ov_model.outputs, OUTPUT_NAMES, strict=False):
            output.tensor.set_names({name})

        # Reshape to static input for optimal GPU kernel compilation.
        ov_model.reshape({ov_model.inputs[0].get_any_name(): [1, 3, input_size, input_size]})

        if compression not in {CompressionMode.FP32, CompressionMode.FP16}:
            from instantlearn.utils.compression import compress_model  # noqa: PLC0415

            ov_model = compress_model(ov_model, mode=compression)

        xml_path = export_dir / f"{model_name}.xml"
        openvino.save_model(ov_model, xml_path, compress_to_fp16=compression == CompressionMode.FP16)
        return xml_path

    return export_dir
