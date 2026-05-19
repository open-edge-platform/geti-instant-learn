# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO model compression via NNCF (weight compression and full PTQ)."""

import logging
import time
from collections.abc import Iterable
from typing import Any

import nncf
import openvino as ov

from instantlearn.utils.constants import CompressionMode

logger = logging.getLogger(__name__)

# Mapping from CompressionMode to nncf.CompressWeightsMode.
_NNCF_MODE_MAP: dict[CompressionMode, str] = {
    CompressionMode.INT8_SYM: "INT8_SYM",
    CompressionMode.INT8_ASYM: "INT8_ASYM",
    CompressionMode.INT4_SYM: "INT4_SYM",
    CompressionMode.INT4_ASYM: "INT4_ASYM",
}


def compress_model(
    ov_model: ov.Model,
    mode: CompressionMode = CompressionMode.INT8_SYM,
    group_size: int = 128,
) -> ov.Model:
    """Apply weight compression to an OpenVINO model.

    Args:
        ov_model: OpenVINO model to compress.
        mode: Compression mode. One of the :class:`CompressionMode` values.
        group_size: Group size for INT4 compression. Ignored for INT8/FP16/FP32.
            Lower group size usually improves accuracy at the sacrifice of inference speed.

    Returns:
        Compressed OpenVINO model. For FP32 and FP16 the model is returned
        unchanged (FP16 is handled by the caller via ``CompressionMode.FP16``
        at save time).

    Raises:
        ValueError: If *mode* is not a supported compression mode.
    """
    if mode in {CompressionMode.FP32, CompressionMode.FP16}:
        # FP32 means no compression and FP16 is handled by openvino.save_model.
        return ov_model

    nncf_mode_name = _NNCF_MODE_MAP.get(mode)
    if nncf_mode_name is None:
        msg = f"Unsupported compression mode: {mode}"
        raise ValueError(msg)

    nncf_mode = getattr(nncf.CompressWeightsMode, nncf_mode_name)

    if mode in {CompressionMode.INT4_SYM, CompressionMode.INT4_ASYM}:
        logger.warning(
            "INT4 weight compression on small vision models may degrade accuracy. "
            "Benchmark the quantized model before deploying.",
        )

    logger.info("Compressing model weights to %s ...", mode.value)
    start = time.time()

    kwargs: dict = {"mode": nncf_mode}
    if mode in {CompressionMode.INT4_SYM, CompressionMode.INT4_ASYM}:
        kwargs["group_size"] = group_size

    ov_model = nncf.compress_weights(ov_model, **kwargs)

    elapsed = time.time() - start
    logger.info("Weight compression (%s) finished in %.1fs", mode.value, elapsed)

    return ov_model


def quantize_model(
    ov_model: ov.Model,
    calibration_data: Iterable[Any],
    *,
    target_device: str = "ANY",
    preset: str = "MIXED",
    subset_size: int = 300,
    model_type: str | None = None,
    fast_bias_correction: bool = True,
    ignored_scope: Any | None = None,
    advanced_parameters: Any | None = None,
) -> ov.Model:
    """Apply full post-training quantization (PTQ) to an OpenVINO model.

    Unlike ``compress_model()`` which only compresses weights, this function
    quantizes both weights AND activations to INT8 using calibration data.
    This enables hardware INT8 acceleration on CPUs with VNNI/AMX and GPUs
    with DP4A/DPAS support.

    Args:
        ov_model: OpenVINO model to quantize.
        calibration_data: Iterable of calibration samples. Each sample should
            be a dict mapping input names to numpy arrays, or a list/tuple of
            numpy arrays matching model input order.
        target_device: Target device for quantization optimization.
            One of "ANY", "CPU", "CPU_SPR", "GPU", "NPU".
        preset: Quantization preset. "PERFORMANCE" for symmetric quantization,
            "MIXED" for asymmetric activations (better for GELU/SiLU models).
        subset_size: Number of calibration samples to use.
        model_type: Optional model type hint. Use "Transformer" for
            attention-based models (enables asymmetric quantization for
            specific attention operations).
        fast_bias_correction: Whether to use fast bias correction (default True).
            Set to False for potentially better accuracy at the cost of speed.
        ignored_scope: Optional ``nncf.IgnoredScope`` instance to exclude
            specific layers or operation types from quantization.
        advanced_parameters: Optional ``nncf.quantization.advanced_parameters.AdvancedQuantizationParameters``
            instance for fine-grained control over quantization behaviour.

    Returns:
        Quantized OpenVINO model with FakeQuantize operations inserted.
    """
    device_map = {
        "ANY": nncf.TargetDevice.ANY,
        "CPU": nncf.TargetDevice.CPU,
        "CPU_SPR": nncf.TargetDevice.CPU_SPR,
        "GPU": nncf.TargetDevice.GPU,
        "NPU": nncf.TargetDevice.NPU,
    }
    preset_map = {
        "PERFORMANCE": nncf.QuantizationPreset.PERFORMANCE,
        "MIXED": nncf.QuantizationPreset.MIXED,
    }
    model_type_map = {
        "Transformer": nncf.ModelType.TRANSFORMER,
    }

    nncf_device = device_map.get(target_device.upper())
    if nncf_device is None:
        msg = f"Unsupported target_device: {target_device}. Use one of {list(device_map.keys())}"
        raise ValueError(msg)

    nncf_preset = preset_map.get(preset.upper())
    if nncf_preset is None:
        msg = f"Unsupported preset: {preset}. Use one of {list(preset_map.keys())}"
        raise ValueError(msg)

    nncf_model_type = None
    if model_type is not None:
        nncf_model_type = model_type_map.get(model_type)
        if nncf_model_type is None:
            msg = f"Unsupported model_type: {model_type}. Use one of {list(model_type_map.keys())}"
            raise ValueError(msg)

    calibration_dataset = nncf.Dataset(calibration_data)

    logger.info(
        "Running full INT8 quantization (target=%s, preset=%s, model_type=%s, subset_size=%d)...",
        target_device,
        preset,
        model_type,
        subset_size,
    )
    start = time.time()

    kwargs: dict[str, Any] = {
        "target_device": nncf_device,
        "preset": nncf_preset,
        "subset_size": subset_size,
        "fast_bias_correction": fast_bias_correction,
    }
    if nncf_model_type is not None:
        kwargs["model_type"] = nncf_model_type
    if ignored_scope is not None:
        kwargs["ignored_scope"] = ignored_scope
    if advanced_parameters is not None:
        kwargs["advanced_parameters"] = advanced_parameters

    quantized_model = nncf.quantize(
        ov_model,
        calibration_dataset,
        **kwargs,
    )

    elapsed = time.time() - start
    logger.info("Full INT8 quantization finished in %.1fs", elapsed)

    return quantized_model
