# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO model weight compression via NNCF."""

import logging
import time

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
