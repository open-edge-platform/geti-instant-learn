# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Intermediate base class for all OpenVINO-backed instantlearn models.

No torch dependency — safe to import in environments where only OpenVINO is
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openvino as ov

from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend


class OpenVINOModel(Model):
    """Intermediate base for all OpenVINO-backed models.

    Handles ``ov.Core`` initialisation and the ``backend`` property. Concrete
    subclasses implement ``card()``, ``fit()``, and ``predict()``.

    ``from_pretrained()`` is not defined here — not all OV models load from
    HuggingFace Hub. Models that support it declare their own classmethod.

    Attributes:
        model_dir: Directory containing the ``.xml`` / ``.bin`` model files.
        device: OpenVINO device hint (e.g. ``"AUTO"``, ``"CPU"``, ``"GPU"``).
        preprocessor: Optional numpy-based preprocessor applied before inference.
        postprocessor: Optional post-processor applied after inference.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "AUTO",
        preprocessor: Any = None,  # noqa: ANN401
        postprocessor: Any = None,  # noqa: ANN401
    ) -> None:
        """Initialize the OpenVINO model base.

        Args:
            model_dir: Path to the directory containing the ``.xml`` / ``.bin``
                files.
            device: OpenVINO device hint, e.g. ``"AUTO"``, ``"CPU"``,
                ``"GPU"``.
            preprocessor: Optional numpy-based preprocessor.
            postprocessor: Optional post-processor.
        """
        super().__init__()
        self.model_dir = Path(model_dir)
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self._core: ov.Core = ov.Core()

    @property
    def backend(self) -> Backend:
        """Always ``Backend.OPENVINO``."""
        return Backend.OPENVINO
