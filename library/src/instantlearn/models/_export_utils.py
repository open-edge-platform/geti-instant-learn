# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Shared export utilities for torch-backed models.

Provides standalone functions and constants re-used by Matcher, PerDino,
and SoftMatcher to keep the ONNX/OpenVINO export code in one place.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from instantlearn.utils.constants import CompressionMode

logger = logging.getLogger(__name__)
#: Fixed IO tensor names of the baked IR / ONNX graph.
_OUTPUT_NAMES: list[str] = ["masks", "scores", "labels"]
#: Neutral IR file stem shared by all baked models (``model.xml`` / ``model.onnx``).
IR_STEM: str = "model"
#: INT4 compression modes produce noisy masks; blocked for these models.
_INT4_MODES: frozenset[CompressionMode] = frozenset({CompressionMode.INT4_SYM, CompressionMode.INT4_ASYM})


def fix_onnx_output_names(onnx_path: Path, expected_names: list[str]) -> None:  # noqa: C901
    """Ensure ONNX graph outputs have the expected names."""
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


def resolve_export_dir(export_path: str | Path | None, family: str) -> Path:
    """Resolve *export_path* to an existing directory (temp dir when ``None``)."""
    if export_path is None:
        import tempfile  # noqa: PLC0415
        return Path(tempfile.mkdtemp(prefix=f"{family}_export_"))
    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)
    return export_dir


def write_metadata(
    export_dir: Path,
    input_size: int,
    patch_size: int,
    category_names: dict[int, str],
) -> None:
    """Write ``metadata.json`` (input/patch size + category id->name map)."""
    metadata = {
        "input_size": input_size,
        "patch_size": patch_size,
        "categories": {str(k): v for k, v in category_names.items()},
    }
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def export_onnx_graph(
    graph: torch.nn.Module,
    onnx_path: Path,
    export_device: torch.device,
    input_size: int,
    *,
    dynamic_shapes: bool,
    opset: int,
) -> None:
    """Trace *graph* to an ONNX file at *onnx_path*.

    Raises:
        RuntimeError: If ONNX export fails for a reason other than the 2GiB
            protobuf limit (which is handled by re-exporting with external data).
    """
    target_image = torch.randn(1, 3, input_size, input_size, device=export_device)
    # The encoder requires a fixed square input and the OV IR is reshaped to a
    # static ``[1, 3, S, S]``, so the spatial dims of ``target_image`` are not
    # dynamic. ``dynamic_shapes`` only governs the variable ``num_masks`` count
    # of the outputs.
    dynamic_axes = (
        {
            "masks": {0: "num_masks"},
            "scores": {0: "num_masks"},
            "labels": {0: "num_masks"},
        }
        if dynamic_shapes
        else None
    )
    try:
        torch.onnx.export(
            graph,
            args=(target_image,),
            f=onnx_path,
            input_names=["target_image"],
            output_names=_OUTPUT_NAMES,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    except RuntimeError as onnx_err:
        if "2GiB" in str(onnx_err) or "protobuf" in str(onnx_err):
            logger.info("Model exceeds ONNX 2GiB limit, re-exporting with external data")
            torch.onnx.export(
                graph,
                args=(target_image,),
                f=str(onnx_path),
                input_names=["target_image"],
                output_names=_OUTPUT_NAMES,
                dynamic_axes=dynamic_axes,
                opset_version=opset,
                dynamo=False,
            )
        else:
            raise
    fix_onnx_output_names(onnx_path, _OUTPUT_NAMES)


def convert_and_save_openvino(
    graph: torch.nn.Module,
    onnx_path: Path,
    export_device: torch.device,
    export_dir: Path,
    input_size: int,
    *,
    compression: CompressionMode,
    keep_intermediate: bool,
    output_names: list[str] | None = None,
) -> Path:
    """Convert the intermediate ONNX (or graph) to an OpenVINO IR and save it.

    Prefers the ONNX frontend for operator coverage, falling back to a direct
    Torch->OV conversion. Reshapes to a static ``[1, 3, S, S]`` input, renames
    the IR outputs, applies optional weight compression, and saves ``model.xml``.

    Args:
        graph: Traceable graph (fallback conversion source).
        onnx_path: Intermediate ONNX file from :func:`export_onnx_graph`.
        export_device: Device the example input is created on.
        export_dir: Directory to write ``model.xml`` / ``model.bin`` into.
        input_size: Square spatial input size for the static reshape.
        compression: OpenVINO weight compression mode.
        keep_intermediate: Keep the intermediate ``.onnx`` after conversion.
        output_names: Names to assign to the IR outputs in order.

    Returns:
        The *export_dir* path containing the saved IR.
    """
    import openvino  # noqa: PLC0415

    example_input = torch.randn(1, 3, input_size, input_size, device=export_device)
    output_names = output_names if output_names is not None else _OUTPUT_NAMES
    core = openvino.Core()
    if onnx_path.exists():
        try:
            ov_model = core.read_model(str(onnx_path))
        except RuntimeError:
            ov_model = openvino.convert_model(graph, example_input=example_input)
    else:
        ov_model = openvino.convert_model(graph, example_input=example_input)

    # Registered buffers returned as outputs get auto-generated names; fix them.
    for output, name in zip(ov_model.outputs, output_names, strict=False):
        output.tensor.set_names({name})

    # Reshape to static input for optimal kernel compilation.
    input_name = ov_model.inputs[0].get_any_name()
    ov_model.reshape({input_name: [1, 3, input_size, input_size]})

    if compression not in {CompressionMode.FP32, CompressionMode.FP16}:
        from instantlearn.utils.compression import compress_model  # noqa: PLC0415

        ov_model = compress_model(ov_model, mode=compression)

    openvino.save_model(
        ov_model,
        export_dir / f"{IR_STEM}.xml",
        compress_to_fp16=compression == CompressionMode.FP16,
    )

    if not keep_intermediate:
        onnx_path.unlink(missing_ok=True)

    return export_dir
