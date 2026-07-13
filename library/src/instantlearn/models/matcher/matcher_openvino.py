# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher OpenVINO inference model.

``MatcherOpenVINO`` runs the baked Matcher IR (``matcher.xml``) produced by
:meth:`~instantlearn.models.matcher.matcher.Matcher.to_openvino`. The reference
features and post-processing are baked into the graph at export time, so this
class is a thin loader: it takes the IR directory, runs the single
``target_image -> (masks, scores, labels)`` graph, and returns
:class:`~instantlearn.data.base.prediction.Prediction` objects.

Because the references are baked in, ``fit()`` is **not** supported here — call
``Matcher.fit(...)`` before ``Matcher.to_openvino(...)`` to choose the
references, then load the resulting directory with ``MatcherOpenVINO``.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from instantlearn.data.base.batch import Batch
from instantlearn.models.openvino_base import OpenVINOModel
from instantlearn.models.torch_adapter import arrays_to_prediction
from instantlearn.utils import device_to_openvino_device

if TYPE_CHECKING:
    from pathlib import Path

    from instantlearn.data.base.batch import Collatable
    from instantlearn.data.base.prediction import Prediction
    from instantlearn.data.base.sample import Sample
    from instantlearn.models.model_card import ModelCard

logger = logging.getLogger(__name__)

_INPUT = "target_image"


class MatcherOpenVINO(OpenVINOModel):
    """Matcher model running the baked OpenVINO IR for inference.

    Examples:
        >>> from instantlearn.models.matcher import Matcher, MatcherOpenVINO
        >>> from instantlearn.data.base.sample import Category, Sample
        >>> import numpy as np

        >>> # 1. Fit references and export the baked IR with a torch Matcher.
        >>> matcher = Matcher(device="cpu")
        >>> matcher.fit(Sample(image_path="ref.jpg", mask_paths=["mask.png"]))
        >>> ir_dir = matcher.to_openvino("./matcher-ov")

        >>> # 2. Load and run the baked IR (no fit needed).
        >>> ov_model = MatcherOpenVINO(model_dir=ir_dir, device="CPU")
        >>> predictions = ov_model.predict(Sample(image_path="target.jpg"))
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
    ) -> None:
        """Load the Matcher IR from *model_dir*.

        Args:
            model_dir: Directory containing ``matcher.xml`` / ``matcher.bin`` and
                ``metadata.json`` (produced by ``Matcher.to_openvino``). May be a
                local path or a remote URI (``file://``, ``hf://``, ``s3://``).
            device: OpenVINO device (``"CPU"``, ``"GPU"``, ``"AUTO"``). PyTorch-style
                names (``"cuda"``, ``"cpu"``) are also accepted.

        Raises:
            FileNotFoundError: If the IR file or ``metadata.json`` are missing.
        """
        super().__init__(model_dir=model_dir, device=device_to_openvino_device(device))

        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            msg = f"metadata.json not found in {self.model_dir}. Export with Matcher.to_openvino()."
            raise FileNotFoundError(msg)
        metadata = json.loads(metadata_path.read_text())
        self.input_size: int = metadata["input_size"]
        self.patch_size: int = metadata.get("patch_size", 0)
        # Category id -> name map baked at export time (id keys stored as strings).
        self._category_names: dict[int, str] = {int(k): v for k, v in metadata.get("categories", {}).items()}

        ir_path = self.model_dir / "matcher.xml"
        if not ir_path.exists():
            msg = f"Required IR file not found: {ir_path}"
            raise FileNotFoundError(msg)

        logger.info("Loading Matcher OpenVINO model from %s on %s...", self.model_dir, self.device)
        self._model = self._core.compile_model(str(ir_path), self.device)
        self._request = self._model.create_infer_request()

    @classmethod
    def card(cls) -> ModelCard:
        """Delegate the capability card to the PyTorch sibling."""
        from instantlearn.models.matcher.matcher import Matcher  # noqa: PLC0415

        return Matcher.card()

    def fit(self, reference: Collatable) -> None:  # noqa: ARG002
        """Not supported: references are baked into the IR at export time.

        Raises:
            NotImplementedError: Always. Fit the torch ``Matcher`` and re-export
                with :meth:`~instantlearn.models.matcher.matcher.Matcher.to_openvino`
                to change the references.
        """
        msg = (
            "MatcherOpenVINO does not support fit(): reference features are baked into the IR "
            "at export time. Call Matcher.fit(...) then Matcher.to_openvino(...) to (re)build the IR."
        )
        raise NotImplementedError(msg)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize an HWC frame to the model input and return NCHW float32 (0-255)."""
        if frame.shape[0] != self.input_size or frame.shape[1] != self.input_size:
            frame = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        chw_image = np.expand_dims(frame.transpose(2, 0, 1), axis=0)
        return np.ascontiguousarray(chw_image, dtype=np.float32)

    @staticmethod
    def _resize_masks_to_frame(masks: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
        """Resize predicted masks to the original frame spatial size.

        Uses nearest-neighbour index mapping so no external resize library is needed.

        Args:
            masks: Predicted masks, ``[1, N, H, W]`` or ``[N, H, W]``.
            frame_h: Original frame height.
            frame_w: Original frame width.

        Returns:
            Boolean mask array ``[N, frame_h, frame_w]``.
        """
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]
        if masks.ndim == 3 and (masks.shape[1] != frame_h or masks.shape[2] != frame_w):
            src_h, src_w = masks.shape[1], masks.shape[2]
            row_idx = (np.arange(frame_h) * src_h // frame_h).clip(0, src_h - 1)
            col_idx = (np.arange(frame_w) * src_w // frame_w).clip(0, src_w - 1)
            masks = masks[:, row_idx][:, :, col_idx]
        return masks > 0.5

    def predict(self, target: Collatable) -> list[Prediction]:
        """Run OpenVINO inference on target image(s).

        Post-processing is baked into the IR, so the raw graph outputs are only
        resized to the original frame and wrapped into ``Prediction`` objects.

        Args:
            target: Target data (Sample, list[Sample], Batch, or image paths).

        Returns:
            A list of ``Prediction`` objects, one per input image.
        """
        target_batch = Batch.collate(target)

        results: list[Prediction] = []
        for sample in target_batch.samples:
            frame = _to_hwc_uint8(sample)
            frame_h, frame_w = frame.shape[:2]
            nchw = self._preprocess(frame)

            self._request.infer({_INPUT: nchw})
            masks = np.array(self._request.get_tensor("masks").data)
            scores = np.array(self._request.get_tensor("scores").data)
            labels = np.array(self._request.get_tensor("labels").data)

            masks_frame = self._resize_masks_to_frame(masks, frame_h, frame_w)

            results.append(
                arrays_to_prediction(
                    masks=masks_frame,
                    scores=scores.astype(np.float32),
                    label_ids=labels.astype(np.int32),
                    categories=self._category_names,
                ),
            )

        return results


def _to_hwc_uint8(sample: Sample) -> np.ndarray:
    """Return an ``(H, W, 3)`` numpy image from a ``Sample`` (numpy HWC per contract)."""
    image = sample.image
    if image is None:
        msg = "MatcherOpenVINO.predict() requires each sample to have an image."
        raise ValueError(msg)
    return image

