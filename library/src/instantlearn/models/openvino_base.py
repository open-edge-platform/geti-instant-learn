# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Intermediate base class for all OpenVINO-backed instantlearn models.

This module has **no torch dependency** — safe to import in environments
where only OpenVINO (not PyTorch) is installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import openvino as ov

from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend


class OpenVINOModel(Model):
    """Intermediate base for all OpenVINO-backed models.

    Provides shared boilerplate for device selection, ``ov.Core``
    initialisation, and the ``backend`` property.  Concrete subclasses
    implement :meth:`~instantlearn.models.base.Model.card`,
    :meth:`~instantlearn.models.base.Model.fit`, and
    :meth:`~instantlearn.models.base.Model.predict`.

    ``from_pretrained`` is **not** defined here — not all OV models load from
    HuggingFace Hub.  Models that support it declare their own classmethod.

    Args:
        model_dir: Path to the directory containing the ``.xml`` / ``.bin``
            files.
        device: OpenVINO device hint, e.g. ``"AUTO"``, ``"CPU"``,
            ``"GPU"``.
        preprocessor: Optional numpy-based preprocessor applied before input.
        postprocessor: Optional post-processor applied after inference.
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "AUTO",
        preprocessor: Any = None,  # noqa: ANN401
        postprocessor: Any = None,  # noqa: ANN401
    ) -> None:
        """Initialise the OpenVINO model base."""
        super().__init__()
        self.model_dir = Path(model_dir)
        self.device = device
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self._core: ov.Core = ov.Core()

    @property
    def backend(self) -> Backend:
        """Always :attr:`~instantlearn.utils.constants.Backend.OPENVINO`."""
        return Backend.OPENVINO
