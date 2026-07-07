# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher OpenVINO inference model.

``MatcherOpenVINO`` runs the Matcher pipeline on OpenVINO IR sub-models
(``encoder.xml`` + ``head.xml``) while exposing the exact same ``fit()`` /
``predict()`` API as the PyTorch :class:`~instantlearn.models.matcher.Matcher`,
returning :class:`~instantlearn.data.base.prediction.Prediction` objects.

Reference features are **not** baked into the IR — ``fit()`` encodes the
reference image(s) through the encoder IR and caches the resulting feature
arrays, which are then fed as inputs to the head IR at ``predict()`` time. This
keeps a single exported model reusable across ``fit()`` calls.

Construction:

* ``MatcherOpenVINO(model_dir=...)`` — load pre-exported IR from a directory.
* ``MatcherOpenVINO.from_torch(matcher, ...)`` — convert a PyTorch ``Matcher``
  to IR on the fly (into a temp dir when no path is given) and load it.

Note:
    Post-processing still uses the torch post-processor (kept identical to the
    PyTorch sibling for parity), so ``torch`` is imported at runtime. This is
    intentional for now — only the heavy encoder/head run in OpenVINO.
"""

from __future__ import annotations

import json
import logging
import tempfile
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from instantlearn.components.feature_extractors import MaskedFeatureExtractor
from instantlearn.components.postprocessing import PostProcessor, default_postprocessor
from instantlearn.components.postprocessing.base import apply_postprocessing
from instantlearn.data.base.batch import Batch
from instantlearn.models.openvino_base import OpenVINOModel
from instantlearn.models.torch_adapter import tensors_to_prediction
from instantlearn.utils import device_to_openvino_device

if TYPE_CHECKING:
    from pathlib import Path

    from instantlearn.data.base.batch import Collatable
    from instantlearn.data.base.prediction import Prediction
    from instantlearn.models.matcher.matcher import Matcher
    from instantlearn.models.model_card import ModelCard
    from instantlearn.models.torch_base import ExportConfig

logger = logging.getLogger(__name__)

_ENCODER = "encoder"
_HEAD = "head"


class MatcherOpenVINO(OpenVINOModel):
    """Matcher model using the OpenVINO runtime for inference.

    Examples:
        >>> from instantlearn.models.matcher import Matcher, MatcherOpenVINO
        >>> from instantlearn.data.base.sample import Category, Sample
        >>> import numpy as np

        >>> # Convert a torch Matcher on the fly (writes IR to a temp dir)
        >>> ov_model = MatcherOpenVINO.from_torch(Matcher(device="cpu"), device="CPU")
        >>> ov_model.fit(Sample(image_path="ref.jpg", mask_paths=["mask.png"]))
        >>> predictions = ov_model.predict(Sample(image_path="target.jpg"))

        >>> # Or load a previously exported directory
        >>> ov_model = MatcherOpenVINO(model_dir="./matcher-ov", device="CPU")
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
        postprocessor: PostProcessor | None = None,
    ) -> None:
        """Initialize from a directory containing ``encoder.xml`` + ``head.xml``.

        Args:
            model_dir: Directory with the exported IR files and ``metadata.json``.
                Use :meth:`from_torch` to create one from a PyTorch ``Matcher``.
            device: OpenVINO device (``"CPU"``, ``"GPU"``, ``"AUTO"``). PyTorch-style
                names (``"cuda"``, ``"cpu"``) are also accepted.
            postprocessor: Torch post-processor applied in ``predict()``. Defaults
                to :func:`~instantlearn.components.postprocessing.default_postprocessor`.

        Raises:
            FileNotFoundError: If the IR files or ``metadata.json`` are missing.
        """
        super().__init__(model_dir=model_dir, device=device_to_openvino_device(device))
        self.postprocessor = postprocessor if postprocessor is not None else default_postprocessor()

        metadata_path = self.model_dir / "metadata.json"
        if not metadata_path.exists():
            msg = f"metadata.json not found in {self.model_dir}. Export with export_matcher()."
            raise FileNotFoundError(msg)
        metadata = json.loads(metadata_path.read_text())
        self.input_size: int = metadata["input_size"]
        self.patch_size: int = metadata["patch_size"]

        encoder_path = self.model_dir / f"{_ENCODER}.xml"
        head_path = self.model_dir / f"{_HEAD}.xml"
        for path in (encoder_path, head_path):
            if not path.exists():
                msg = f"Required IR file not found: {path}"
                raise FileNotFoundError(msg)

        logger.info("Loading Matcher OpenVINO models from %s on %s...", self.model_dir, self.device)
        self._encoder_model = self._core.compile_model(str(encoder_path), self.device)
        self._head_model = self._core.compile_model(str(head_path), self.device)
        self._encoder_request = self._encoder_model.create_infer_request()
        self._head_request = self._head_model.create_infer_request()

        # Torch helper reused for reference feature extraction during fit().
        self._masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.input_size,
            patch_size=self.patch_size,
            device="cpu",
        )

        # Reference state set by fit()
        self._ref_features: dict[str, np.ndarray] | None = None
        self._category_names: dict[int, str] = {}
        # Keep a handle to a temp export dir (if any) so it is not GC'd.
        self._tempdir: tempfile.TemporaryDirectory | None = None

    @classmethod
    def from_torch(
        cls,
        matcher: Matcher,
        export_path: str | Path | None = None,
        device: str = "CPU",
        config: ExportConfig | None = None,
        postprocessor: PostProcessor | None = None,
    ) -> MatcherOpenVINO:
        """Convert a PyTorch ``Matcher`` to OpenVINO IR and load it.

        Args:
            matcher: The PyTorch ``Matcher`` to export. Does not need to be fitted.
            export_path: Directory to write the IR into. ``None`` uses a temporary
                directory kept alive for the lifetime of the returned instance.
            device: OpenVINO device for the loaded model.
            config: Export options (compression, opset, ...). ``None`` uses
                :class:`~instantlearn.models.torch_base.ExportConfig` defaults.
            postprocessor: Torch post-processor for ``predict()``. Defaults to the
                exported ``matcher``'s post-processor.

        Returns:
            A ready-to-use ``MatcherOpenVINO`` instance.
        """
        from instantlearn.scripts.matcher.export_matcher import export_matcher  # noqa: PLC0415

        tempdir: tempfile.TemporaryDirectory | None = None
        if export_path is None:
            tempdir = tempfile.TemporaryDirectory(prefix="matcher_ov_")
            export_path = tempdir.name

        export_matcher(matcher, output_dir=export_path, config=config)

        instance = cls(
            model_dir=export_path,
            device=device,
            postprocessor=postprocessor if postprocessor is not None else matcher.postprocessor,
        )
        instance._tempdir = tempdir
        # Reuse the source category names if the torch model was already fitted.
        if getattr(matcher, "_category_names", None):
            instance._category_names = dict(matcher._category_names)  # noqa: SLF001
        return instance

    @classmethod
    def card(cls) -> ModelCard:
        """Delegate the capability card to the PyTorch sibling."""
        from instantlearn.models.matcher.matcher import Matcher  # noqa: PLC0415

        return Matcher.card()

    # ------------------------------------------------------------------
    # Preprocessing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_hwc_uint8(image: np.ndarray | torch.Tensor) -> np.ndarray:
        """Return an ``(H, W, 3)`` uint8/float numpy image from numpy or torch."""
        if isinstance(image, torch.Tensor):
            arr = image.detach().cpu().numpy()
            if arr.ndim == 3 and arr.shape[0] in {1, 3}:  # CHW -> HWC
                arr = arr.transpose(1, 2, 0)
            return arr
        return image

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Resize an HWC frame to the model input and return NCHW float32 (0-255)."""
        if frame.shape[0] != self.input_size or frame.shape[1] != self.input_size:
            frame = cv2.resize(frame, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR)
        chw = np.expand_dims(frame.transpose(2, 0, 1), axis=0)
        return np.ascontiguousarray(chw, dtype=np.float32)

    @staticmethod
    def _resize_masks_to_frame(masks: np.ndarray, frame_h: int, frame_w: int) -> np.ndarray:
        """Nearest-neighbour resize ``[C, H, W]`` masks to the frame size, as bool."""
        if masks.ndim == 4 and masks.shape[0] == 1:
            masks = masks[0]
        if masks.ndim == 3 and (masks.shape[1] != frame_h or masks.shape[2] != frame_w):
            src_h, src_w = masks.shape[1], masks.shape[2]
            row_idx = (np.arange(frame_h) * src_h // frame_h).clip(0, src_h - 1)
            col_idx = (np.arange(frame_w) * src_w // frame_w).clip(0, src_w - 1)
            masks = masks[:, row_idx][:, :, col_idx]
        return masks > 0.5

    def _encode(self, nchw: np.ndarray) -> np.ndarray:
        """Run the encoder IR and return ``[1, num_patches, embed_dim]`` embeddings."""
        self._encoder_request.infer({"image": nchw})
        return np.array(self._encoder_request.get_tensor("embeddings").data)

    # ------------------------------------------------------------------
    # Model contract
    # ------------------------------------------------------------------

    def fit(self, reference: Collatable) -> None:
        """Encode reference image(s) and cache reference features for predict().

        Args:
            reference: Reference data (samples with images + masks + categories).

        Raises:
            ValueError: If no reference image is available.
        """
        reference_batch = Batch.collate(reference)

        embeddings_list = []
        for sample in reference_batch.samples:
            if sample.image is None:
                continue
            frame = self._to_hwc_uint8(sample.image)
            nchw = self._preprocess(frame)
            embeddings_list.append(torch.from_numpy(self._encode(nchw)))

        if not embeddings_list:
            msg = "MatcherOpenVINO.fit() requires at least one reference sample with an image."
            raise ValueError(msg)

        embeddings = torch.cat(embeddings_list, dim=0)  # [B, num_patches, embed_dim]
        ref_features = self._masked_feature_extractor(
            embeddings,
            reference_batch.masks,
            reference_batch.label_ids,
        )

        self._ref_features = {
            "ref_embeddings": ref_features.ref_embeddings.detach().cpu().numpy().astype(np.float32),
            "masked_ref_embeddings": ref_features.masked_ref_embeddings.detach().cpu().numpy().astype(np.float32),
            "flatten_ref_masks": ref_features.flatten_ref_masks.detach().cpu().numpy().astype(np.float32),
            "category_ids": np.array(ref_features.category_ids, dtype=np.int64),
        }

        self._category_names = {}
        for sample in reference_batch.samples:
            if not sample.label_ids or not sample.category_labels:
                continue
            for cat_id, label in zip(sample.label_ids, sample.category_labels, strict=False):
                self._category_names.setdefault(int(cat_id), label)

    def predict(self, target: Collatable) -> list[Prediction]:
        """Run OpenVINO inference on target image(s).

        Args:
            target: Target data (Sample, list[Sample], Batch, or image paths).

        Returns:
            A list of ``Prediction`` objects with post-processing applied.

        Raises:
            ModelNotFittedError: If ``fit()`` has not been called.
        """
        from instantlearn.utils.errors import ModelNotFittedError  # noqa: PLC0415

        if self._ref_features is None:
            msg = "MatcherOpenVINO requires fit() before predict(). Call model.fit(reference_sample) first."
            raise ModelNotFittedError(msg)

        target_batch = Batch.collate(target)
        max_id = max(self._category_names) if self._category_names else -1
        categories = [self._category_names.get(i, str(i)) for i in range(max_id + 1)]

        results: list[Prediction] = []
        for sample in target_batch.samples:
            frame = self._to_hwc_uint8(sample.image)
            frame_h, frame_w = frame.shape[:2]
            nchw = self._preprocess(frame)
            embeddings = self._encode(nchw)

            self._head_request.infer({
                "target_image": nchw,
                "target_embeddings": embeddings,
                "ref_embeddings": self._ref_features["ref_embeddings"],
                "masked_ref_embeddings": self._ref_features["masked_ref_embeddings"],
                "flatten_ref_masks": self._ref_features["flatten_ref_masks"],
                "category_ids": self._ref_features["category_ids"],
            })
            masks = np.array(self._head_request.get_tensor("masks").data)
            scores = np.array(self._head_request.get_tensor("scores").data)
            labels = np.array(self._head_request.get_tensor("labels").data)

            masks_frame = self._resize_masks_to_frame(masks, frame_h, frame_w)

            # Torch post-processing (identical to the PyTorch sibling for parity).
            pred_dict = {
                "pred_masks": torch.from_numpy(np.ascontiguousarray(masks_frame)),
                "pred_scores": torch.from_numpy(scores.astype(np.float32)),
                "pred_labels": torch.from_numpy(labels.astype(np.int64)),
            }
            processed = apply_postprocessing([pred_dict], self.postprocessor)[0]

            boxes = None
            if "pred_boxes" in processed and processed["pred_boxes"].numel() > 0:
                boxes = processed["pred_boxes"][:, :4]

            results.append(
                tensors_to_prediction(
                    masks=processed["pred_masks"],
                    scores=processed["pred_scores"],
                    label_ids=processed["pred_labels"].to(torch.int32),
                    categories=categories,
                    boxes=boxes,
                ),
            )

        return results
