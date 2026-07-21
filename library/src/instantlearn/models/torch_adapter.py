# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Torch adapter: the single bridge between backend-neutral data and torch.

This module is the *only* place where ``instantlearn.data.base`` numpy
containers are converted into torch tensors (and back). Keeping the conversion
here (instead of on ``Sample`` / ``Prediction``) preserves dependency
inversion: the backend-neutral abstractions never import torch, and adding a
new backend never forces a change to the core data classes (Open/Closed).

Torch-backed models are the consumers: they call :func:`samples_to_tensors`
to convert inputs and :func:`tensors_to_prediction` (from torch tensors) or
:func:`arrays_to_prediction` (from numpy arrays) to build the numpy
``Prediction`` at the return boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from instantlearn.data.base.batch import Batch
from instantlearn.data.base.prediction import Prediction
from instantlearn.data.base.sample import Sample

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass
class TensorSample:
    """Torch-native counterpart of :class:`~instantlearn.data.base.sample.Sample`.

    Produced by :func:`sample_to_tensors` and consumed internally by
    ``TorchModel`` subclasses. All array fields are tensors; ``category_labels``
    stays as a plain list of strings.

    Attributes:
        image: Image tensor of shape ``(C, H, W)`` float32.
        masks: Instance masks of shape ``(N, H, W)``.
        bboxes: Bounding boxes of shape ``(N, 4)`` float32 in xyxy format.
        points: Prompt points of shape ``(N, K, 2)`` float32.
        scores: Per-instance scores of shape ``(N,)`` float32.
        category_labels: List of category name strings.
        label_ids: Category IDs of shape ``(N,)`` int32.
    """

    image: torch.Tensor | None = None
    masks: torch.Tensor | None = None
    bboxes: torch.Tensor | None = None
    points: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    category_labels: list[str] | None = None
    label_ids: torch.Tensor | None = None


@dataclass
class TensorBatch:
    """Torch-native counterpart of :class:`~instantlearn.data.base.batch.Batch`.

    Produced by :func:`batch_to_tensors`, it holds the per-sample
    :class:`TensorSample` list and exposes the same batch-level accessors as
    ``Batch`` (``images`` / ``masks`` / ``label_ids`` / ...) but as torch
    tensors instead of numpy arrays. This keeps ``TorchModel`` subclasses on a
    single torch boundary while mirroring the numpy ``Batch`` API.

    Attributes:
        samples: Ordered list of :class:`TensorSample` objects.
    """

    samples: list[TensorSample] = field(default_factory=list)

    def __len__(self) -> int:
        """Return the number of samples in the batch."""
        return len(self.samples)

    def __iter__(self):  # noqa: ANN204
        """Iterate over the underlying :class:`TensorSample` objects."""
        return iter(self.samples)

    @property
    def images(self) -> list[torch.Tensor | None]:
        """Per-sample image tensors of shape ``(C, H, W)``."""
        return [s.image for s in self.samples]

    @property
    def masks(self) -> list[torch.Tensor | None]:
        """Per-sample instance masks of shape ``(N, H, W)``."""
        return [s.masks for s in self.samples]

    @property
    def label_ids(self) -> list[torch.Tensor | None]:
        """Per-sample integer category ids of shape ``(N,)``."""
        return [s.label_ids for s in self.samples]

    @property
    def category_labels(self) -> list[list[str] | None]:
        """Per-sample category label strings."""
        return [s.category_labels for s in self.samples]


def label_ids_as_ints(sample: Sample | TensorSample) -> list[int]:
    """Return a sample's category ids as plain Python integers.

    Normalizes the three shapes ``label_ids`` can take at the torch boundary:
    ``None`` (no categories), a ``torch.Tensor`` (``TensorSample``), or a plain
    list of ints (``Sample``).

    Args:
        sample: A backend-neutral ``Sample`` or torch-native ``TensorSample``.

    Returns:
        Category ids as a list of ``int`` (empty when ``label_ids`` is ``None``).
    """
    label_ids = sample.label_ids
    if label_ids is None:
        return []
    if isinstance(label_ids, torch.Tensor):
        return [int(label_id) for label_id in label_ids.detach().cpu().tolist()]
    return [int(label_id) for label_id in label_ids]


@dataclass(frozen=True)
class CategoryRegistry(Mapping):
    """Bidirectional category id <-> name identity, built once from references.

    This is the single home for category identity across every model.

    Because it implements :class:`collections.abc.Mapping` over
    ``id_to_name``, an instance can be passed directly as the ``categories``
    argument of :func:`arrays_to_prediction`, :func:`tensors_to_prediction`, and
    :func:`dict_to_prediction`.

    Attributes:
        id_to_name: Mapping from integer category id to category name.
        name_to_id: Mapping from category name to integer category id.
    """

    id_to_name: dict[int, str] = field(default_factory=dict)
    name_to_id: dict[str, int] = field(default_factory=dict)

    def __getitem__(self, key: int) -> str:
        """Return the category name for *key* (an integer id)."""
        return self.id_to_name[key]

    def __iter__(self):  # noqa: ANN204
        """Iterate over the integer category ids."""
        return iter(self.id_to_name)

    def __len__(self) -> int:
        """Return the number of registered categories."""
        return len(self.id_to_name)

    @classmethod
    def from_samples(
        cls,
        samples: Sample | list[Sample] | Batch | list[TensorSample],
    ) -> CategoryRegistry:
        """Build a registry from reference samples, keeping first occurrences.

        Iterates samples in order and records each ``(id, name)`` pair the first
        time its name is seen, so earlier references win on duplicates.

        Args:
            samples: A single ``Sample``, a list of ``Sample`` / ``TensorSample``,
                or a ``Batch``.

        Returns:
            A populated :class:`CategoryRegistry`.
        """
        if isinstance(samples, Sample):
            sample_list: list[Sample | TensorSample] = [samples]
        elif isinstance(samples, Batch):
            sample_list = list(samples.samples)
        else:
            sample_list = list(samples)

        id_to_name: dict[int, str] = {}
        name_to_id: dict[str, int] = {}
        for sample in sample_list:
            labels = sample.category_labels
            if not labels:
                continue
            for cat_id, name in zip(label_ids_as_ints(sample), labels, strict=False):
                if name not in name_to_id:
                    name_to_id[name] = cat_id
                    id_to_name.setdefault(cat_id, name)
        return cls(id_to_name=id_to_name, name_to_id=name_to_id)

    @classmethod
    def from_metadata(cls, categories: Mapping) -> CategoryRegistry:
        """Build a registry from a serialized ``{id: name}`` mapping.

        Used by OpenVINO siblings that load category identity from
        ``metadata.json`` (ids are JSON object keys, hence strings).

        Args:
            categories: Mapping from category id (``int`` or ``str``) to name.

        Returns:
            A populated :class:`CategoryRegistry`.
        """
        id_to_name = {int(key): value for key, value in categories.items()}
        name_to_id = {name: cat_id for cat_id, name in id_to_name.items()}
        return cls(id_to_name=id_to_name, name_to_id=name_to_id)

    def names_indexed(self) -> list[str]:
        """Return category names as a dense list indexed by id.

        Ids missing from the registry fall back to their string form. Useful for
        callers that want a ``Sequence[str]`` rather than a mapping.

        Returns:
            ``[name_for(0), name_for(1), ...]`` up to the largest known id.
        """
        if not self.id_to_name:
            return []
        max_id = max(self.id_to_name)
        return [self.id_to_name.get(cat_id, str(cat_id)) for cat_id in range(max_id + 1)]


def sample_to_tensors(sample: Sample, device: str = "cpu") -> TensorSample:
    """Convert a :class:`Sample` to a torch :class:`TensorSample`.

    ``image`` is permuted from HWC to CHW and cast to float32. This is the
    torch boundary — ``Sample`` itself never imports torch.

    Args:
        sample: Backend-neutral sample. Numpy arrays are converted to tensors;
            tensor-valued fields are moved to *device* for compatibility with
            existing torch examples.
        device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        A ``TensorSample`` with all non-``None`` fields moved to *device*.
    """
    def _to_tensor(value: np.ndarray | torch.Tensor | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.ascontiguousarray(value))
        return tensor.to(device=device, dtype=dtype) if dtype is not None else tensor.to(device=device)

    image_t = None
    if sample.image is not None:
        arr = sample.image
        if isinstance(arr, torch.Tensor):
            image_t = arr.float().to(device)
        else:
            if arr.ndim == 3:
                arr = arr.transpose(2, 0, 1)  # HWC -> CHW
            image_t = torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)

    label_ids = sample.label_ids
    return TensorSample(
        image=image_t,
        masks=_to_tensor(sample.masks),
        bboxes=_to_tensor(sample.bboxes, dtype=torch.float32),
        points=_to_tensor(sample.points, dtype=torch.float32),
        scores=_to_tensor(sample.scores, dtype=torch.float32),
        category_labels=sample.category_labels,
        label_ids=torch.tensor(label_ids, dtype=torch.int32, device=device) if label_ids else None,
    )


def samples_to_tensors(target: Sample | list[Sample] | Batch, device: str = "cpu") -> list[TensorSample]:
    """Convert ``Sample`` / ``list[Sample]`` / ``Batch`` inputs to ``TensorSample``.

    This is the single numpy->torch entry point for model inputs. Single
    samples, lists, and ``Batch`` objects are handled uniformly.

    Args:
        target: One or more samples, or a ``Batch``.
        device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        A list of ``TensorSample`` objects on *device*.
    """
    if isinstance(target, Sample):
        target = [target]
    elif isinstance(target, Batch):
        target = target.samples
    return [sample_to_tensors(s, device) for s in target]


def batch_to_tensors(target: Sample | list[Sample] | Batch, device: str = "cpu") -> TensorBatch:
    """Convert ``Sample`` / ``list[Sample]`` / ``Batch`` inputs to a ``TensorBatch``.

    Torch-native sibling of :func:`samples_to_tensors` that keeps the batch
    container (rather than a bare list), giving models ``batch.images`` /
    ``batch.masks`` / ``batch.label_ids`` as tensors.

    Args:
        target: One or more samples, or a ``Batch``.
        device: Target device string, e.g. ``"cpu"`` or ``"cuda"``.

    Returns:
        A ``TensorBatch`` whose samples live on *device*.
    """
    return TensorBatch(samples=samples_to_tensors(target, device))


def dict_to_prediction(
    pred: dict[str, torch.Tensor],
    categories: Mapping[int, str] | Sequence[str],
) -> Prediction:
    """Convert a single torch prediction dict to a numpy ``Prediction``.

    Shared by every torch-backed model so the ``pred_masks`` / ``pred_scores`` /
    ``pred_labels`` / ``pred_boxes`` dict dialect is unpacked in exactly one
    place. Missing scores default to ones; ``pred_boxes`` is sliced to the first
    four columns (xyxy), dropping any trailing score column.

    Args:
        pred: Dict with ``pred_masks`` ``(N, H, W)``, ``pred_labels`` ``(N,)``
            and optionally ``pred_scores`` ``(N,)`` / ``pred_boxes`` ``(N, 4+)``.
        categories: Mapping (or sequence) resolving a label id to its name.

    Returns:
        A numpy ``Prediction`` with contract dtypes enforced.
    """
    masks = pred["pred_masks"]
    label_ids = pred["pred_labels"].to(torch.int32)
    scores = pred.get("pred_scores")
    if scores is None:
        scores = torch.ones(masks.shape[0], device=masks.device)

    boxes = None
    if "pred_boxes" in pred and pred["pred_boxes"].numel() > 0:
        boxes = pred["pred_boxes"][:, :4]

    return tensors_to_prediction(
        masks=masks,
        scores=scores,
        label_ids=label_ids,
        categories=categories,
        boxes=boxes,
    )


def prediction_to_dict(
    prediction: Prediction,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert a numpy ``Prediction`` back to the torch prediction dict dialect.

    Inverse of :func:`dict_to_prediction`. This is the boundary used by torch
    consumers that still speak the ``pred_masks`` / ``pred_scores`` /
    ``pred_labels`` / ``pred_boxes`` dict form — most notably the
    post-processing subsystem, which stays torch-based while model I/O uses the
    backend-neutral ``Prediction``.

    ``pred_boxes`` is only emitted when ``prediction.boxes`` is present; it is
    built as ``[x1, y1, x2, y2, score]`` (5 columns) to match the convention
    consumed by :func:`~instantlearn.components.postprocessing.base.apply_postprocessing`
    and re-sliced back to xyxy by :func:`dict_to_prediction`.

    Args:
        prediction: Backend-neutral numpy ``Prediction``.
        device: Target device string for the produced tensors.

    Returns:
        A dict with ``pred_masks`` ``(N, H, W)``, ``pred_scores`` ``(N,)``,
        ``pred_labels`` ``(N,)`` and optionally ``pred_boxes`` ``(N, 5)``.
    """
    masks = torch.as_tensor(np.ascontiguousarray(prediction.masks), device=device)
    scores = torch.as_tensor(np.ascontiguousarray(prediction.scores), device=device).float()
    labels = torch.as_tensor(np.ascontiguousarray(prediction.label_ids), device=device)

    result: dict[str, torch.Tensor] = {
        "pred_masks": masks,
        "pred_scores": scores,
        "pred_labels": labels,
    }
    if prediction.boxes is not None and len(prediction.boxes):
        boxes = torch.as_tensor(np.ascontiguousarray(prediction.boxes), device=device).float()
        result["pred_boxes"] = torch.cat([boxes, scores.unsqueeze(1)], dim=1)
    return result


def tensors_to_prediction(
    masks: torch.Tensor,
    scores: torch.Tensor,
    label_ids: torch.Tensor,
    categories: Mapping[int, str] | Sequence[str],
    boxes: torch.Tensor | None = None,
    points: torch.Tensor | None = None,
    metadata: dict | None = None,
) -> Prediction:
    """Convert torch model outputs to a numpy ``Prediction``.

    This is the single torch->numpy boundary for all PyTorch-backed models.
    Every tensor is moved to host memory via ``detach().cpu().numpy()``, then
    dtype normalization and ``label_ids -> label_names`` mapping are applied by
    :func:`arrays_to_prediction`. The resulting ``Prediction`` is a pure numpy,
    backend-neutral data container.

    Args:
        masks: Instance masks tensor of shape ``(N, H, W)``.
        scores: Confidence scores tensor of shape ``(N,)``.
        label_ids: Integer category IDs tensor of shape ``(N,)``.
        categories: Mapping (or sequence) resolving a label ID to its name.
        boxes: Optional bounding boxes tensor of shape ``(N, 4)``.
        points: Optional point predictions tensor of shape ``(N, K, 2)``.
        metadata: Optional free-form per-prediction metadata.

    Returns:
        A numpy ``Prediction`` with contract dtypes enforced.
    """

    def _np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    def _np_opt(t: torch.Tensor | None) -> np.ndarray | None:
        return t.detach().cpu().numpy() if t is not None else None

    return arrays_to_prediction(
        masks=_np(masks),
        scores=_np(scores),
        label_ids=_np(label_ids),
        categories=categories,
        boxes=_np_opt(boxes),
        points=_np_opt(points),
        metadata=metadata,
    )


def arrays_to_prediction(
    masks: np.ndarray,
    scores: np.ndarray,
    label_ids: np.ndarray,
    categories: Mapping[int, str] | Sequence[str],
    boxes: np.ndarray | None = None,
    points: np.ndarray | None = None,
    metadata: dict | None = None,
) -> Prediction:
    """Assemble a normalized numpy ``Prediction`` from raw numpy arrays.

    Enforces the contract dtypes:

    - ``masks``: ``bool`` if already boolean, otherwise ``uint8``.
    - ``scores``: ``float32``.
    - ``label_ids``: ``int32``.
    - ``boxes`` / ``points``: ``float32`` when present.

    ``label_names`` is derived by resolving each entry of ``label_ids`` against
    ``categories`` (a mapping keyed by id, or a sequence indexed by id); ids
    outside the range / not present fall back to ``str(id)``.

    Args:
        masks: Instance masks of shape ``(N, H, W)``.
        scores: Per-instance confidence scores of shape ``(N,)``.
        label_ids: Per-instance integer category IDs of shape ``(N,)``.
        categories: Mapping (or sequence) resolving a label ID to its name.
        boxes: Optional bounding boxes of shape ``(N, 4)`` in xyxy format.
        points: Optional point predictions of shape ``(N, K, 2)``.
        metadata: Optional free-form per-prediction metadata.

    Returns:
        A ``Prediction`` with all arrays cast to the contract dtypes.
    """
    masks = np.ascontiguousarray(masks)
    if masks.dtype != np.bool_:
        masks = masks.astype(np.uint8, copy=False)

    scores = np.ascontiguousarray(scores, dtype=np.float32)
    label_ids = np.ascontiguousarray(label_ids, dtype=np.int32)

    if isinstance(categories, Mapping):
        label_names = np.array(
            [categories.get(int(i), str(i)) for i in label_ids.tolist()],
            dtype=object,
        )
    else:
        n_categories = len(categories)
        label_names = np.array(
            [categories[i] if 0 <= i < n_categories else str(i) for i in label_ids.tolist()],
            dtype=object,
        )

    if boxes is not None:
        boxes = np.ascontiguousarray(boxes, dtype=np.float32)
    if points is not None:
        points = np.ascontiguousarray(points, dtype=np.float32)

    return Prediction(
        masks=masks,
        scores=scores,
        label_ids=label_ids,
        label_names=label_names,
        boxes=boxes,
        points=points,
        metadata=metadata if metadata is not None else {},
    )
