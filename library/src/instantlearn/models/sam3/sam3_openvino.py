# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 OpenVINO inference model for text, box, point, and visual-exemplar prompting.

This module provides ``SAM3OpenVINO``, which loads pre-exported SAM3 OpenVINO IR
(or ONNX) models and provides the same inference API as the PyTorch ``SAM3`` model.

Supported prompt types (matching PyTorch ``SAM3`` parity):

* **Text prompts** — category names via ``fit()`` or per-sample ``categories``
* **Box prompts** — bounding boxes via the ``bboxes`` field
* **Point prompts** — click points via the ``points`` field
* **Combined text + box/point** — both at the same time
* **Visual exemplar mode** — encode reference image prompts at ``fit()``
  time, reuse cached features to detect similar objects on any target image

The model expects 5 sub-models (custom-exported from ``Sam3Model``):

* ``vision-encoder``   — ViT + FPN backbone
* ``text-encoder``     — CLIP text encoder + projection
* ``geometry-encoder`` — Geometry encoder (classic, ``drop_spatial_bias=False``)
* ``geometry-encoder-exemplar`` — Geometry encoder (exemplar, ``drop_spatial_bias=True``)
* ``prompt-decoder``   — DETR encoder/decoder + box refinement + scoring + mask decoder

See Also:
    - ``SAM3``: PyTorch-based SAM3 model
    - ``export_openvino``: ONNX export wrappers and conversion utilities
    - ``export_sam3_openvino.py``: CLI script for exporting models
"""

import logging
from collections import defaultdict
from enum import Enum
from itertools import zip_longest
from pathlib import Path

import numpy as np
import openvino as ov
import torch
from transformers import CLIPTokenizerFast

from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils import device_to_openvino_device

from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor
from .sam3 import Sam3PromptMode

logger = logging.getLogger(__name__)

# Default HuggingFace repo for SAM3 OpenVINO models and tokenizer
_DEFAULT_HF_REPO = "rajeshgangireddy/exported_sam3"


class SAM3OVVariant(str, Enum):
    """Available SAM3 OpenVINO model variants.

    Each variant maps to a subdirectory name on HuggingFace Hub.
    """

    FP16 = "openvino-fp16"
    FP32 = "openvino-fp32"
    INT8 = "openvino-nncf-int8"
    INT4 = "openvino-nncf-int4"


# Sub-model file names
_VISION_ENCODER = "vision-encoder"
_TEXT_ENCODER = "text-encoder"
_GEOMETRY_ENCODER = "geometry-encoder"
_GEOMETRY_ENCODER_EXEMPLAR = "geometry-encoder-exemplar"
_PROMPT_DECODER = "prompt-decoder"


def _find_model_file(model_dir: Path, name: str) -> Path | None:
    """Find a model file in a directory, supporting OV IR (.xml) and ONNX (.onnx).

    Search order:
      1. ``{name}.xml`` — OpenVINO IR (preferred)
      2. ``{name}.onnx`` — canonical ONNX name
      3. ``{name}-fp16.onnx`` — FP16 ONNX variant
      4. Any remaining ``{name}*.onnx`` — other quantised variants

    Args:
        model_dir: Directory to search.
        name: Base name of the model (without extension).

    Returns:
        Path to the found model file, or ``None`` if not found.
    """
    candidates = [
        model_dir / f"{name}.xml",
        model_dir / f"{name}.onnx",
        model_dir / f"{name}-fp16.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    onnx_variants = sorted(model_dir.glob(f"{name}*.onnx"))
    if onnx_variants:
        return onnx_variants[0]
    return None


def _require_model_file(model_dir: Path, name: str) -> Path:
    """Find a model file or raise.

    Args:
        model_dir: Directory to search.
        name: Base name of the model (without extension).

    Returns:
        Path to the found model file.

    Raises:
        FileNotFoundError: If no matching model file is found.
    """
    path = _find_model_file(model_dir, name)
    if path is None:
        msg = f"Model '{name}' not found in {model_dir}."
        raise FileNotFoundError(msg)
    return path


class SAM3OpenVINO(Model):
    """SAM3 model using OpenVINO runtime for inference.

    Provides the same capabilities as the PyTorch ``SAM3`` model:

    * **CLASSIC** mode — text, box, point, or combined prompts per target image.
    * **VISUAL_EXEMPLAR** mode — encode reference-image prompts during ``fit()``
      and reuse cached geometry features for every target image in ``predict()``.

    The model loads 4 (or 5) pre-exported sub-models from *model_dir*:

    * ``vision-encoder``
    * ``text-encoder``
    * ``geometry-encoder``  (classic mode)
    * ``geometry-encoder-exemplar``  (exemplar fit)
    * ``prompt-decoder``

    Examples:
        >>> from instantlearn.models.sam3 import SAM3OpenVINO, Sam3PromptMode
        >>> from instantlearn.data.base.sample import Sample
        >>> import numpy as np

        >>> # Auto-download default variant (FP16) from HuggingFace
        >>> model = SAM3OpenVINO(device="CPU")
        >>> model.fit(Sample(categories=["elephant"], category_ids=[0]))
        >>> results = model.predict(
        ...     Sample(image_path="examples/assets/coco/000000286874.jpg", categories=["elephant"]),
        ... )

        >>> # Auto-download INT8 quantised variant
        >>> model = SAM3OpenVINO(variant=SAM3OVVariant.INT8, device="CPU")

        >>> # Use a local model directory (no download)
        >>> model = SAM3OpenVINO(model_dir="./sam3-openvino/openvino-fp16", device="CPU")

        >>> # Box prompting (elephant bounding box)
        >>> target = Sample(
        ...     image_path="examples/assets/coco/000000286874.jpg",
        ...     bboxes=np.array([[180, 105, 490, 370]]),
        ... )
        >>> results = model.predict(target)

        >>> # Point prompting (click on elephant)
        >>> target = Sample(
        ...     image_path="examples/assets/coco/000000286874.jpg",
        ...     points=np.array([[335, 240]]),
        ... )
        >>> results = model.predict(target)

        >>> # Visual exemplar mode — fit on one image, predict on another
        >>> model_ve = SAM3OpenVINO(
        ...     variant=SAM3OVVariant.FP16,
        ...     prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR,
        ... )
        >>> ref = Sample(
        ...     image_path="examples/assets/coco/000000286874.jpg",
        ...     bboxes=np.array([[180, 105, 490, 370]]),
        ...     categories=["elephant"],
        ...     category_ids=[0],
        ... )
        >>> model_ve.fit(ref)
        >>> results = model_ve.predict(
        ...     Sample(image_path="examples/assets/coco/000000390341.jpg"),
        ... )
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "CPU",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        prompt_mode: Sam3PromptMode = Sam3PromptMode.CLASSIC,
        drop_spatial_bias: bool = True,
        tokenizer_path: str | Path | None = None,
        variant: SAM3OVVariant = SAM3OVVariant.FP16,
        repo_id: str = _DEFAULT_HF_REPO,
    ) -> None:
        """Initialise SAM3 OpenVINO model.

        When *model_dir* is provided, models are loaded from the local directory.
        Otherwise, the *variant* is automatically downloaded from HuggingFace Hub.

        Args:
            model_dir: Directory containing OpenVINO IR or ONNX sub-models.
                When ``None``, models are auto-downloaded from HuggingFace Hub.
            device: OpenVINO device (``"CPU"``, ``"GPU"``, ``"AUTO"``).
                PyTorch-style names (``"cuda"``, ``"cpu"``) are also accepted.
            confidence_threshold: Minimum confidence score for predictions.
            resolution: Input image resolution (must match exported model).
            prompt_mode: ``Sam3PromptMode.CLASSIC`` or
                ``Sam3PromptMode.VISUAL_EXEMPLAR``.
            drop_spatial_bias: When True the exemplar geometry encoder drops
                coordinate projections/position encodings and keeps only pooled
                visual features (better for cross-image transfer).
            tokenizer_path: Explicit tokenizer path or HuggingFace model ID.
            variant: Model variant to download when *model_dir* is ``None``.
            repo_id: HuggingFace repository ID for auto-download.
        """
        super().__init__()

        self.model_dir = self._resolve_model_dir(model_dir, variant=variant, repo_id=repo_id)
        self.ov_device = device_to_openvino_device(device)
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.prompt_mode = prompt_mode
        self.drop_spatial_bias = drop_spatial_bias

        # Category mapping from fit()
        self.category_mapping: dict[str, int] | None = None

        # Exemplar cache (populated in _fit_visual_exemplar)
        self.exemplar_geometry_features: list[np.ndarray] | None = None
        self.exemplar_geometry_mask: list[np.ndarray] | None = None
        self.exemplar_text_features: list[np.ndarray] | None = None
        self.exemplar_text_mask: list[np.ndarray] | None = None
        self.exemplar_category_ids: list[int] | None = None

        # -- Load sub-models --
        core = ov.Core()

        # Vision encoder + text encoder (always required)
        vision_path = _require_model_file(self.model_dir, _VISION_ENCODER)
        text_path = _require_model_file(self.model_dir, _TEXT_ENCODER)

        logger.info("Loading SAM3 OpenVINO models from %s on %s...", self.model_dir, self.ov_device)
        self.vision_model = core.compile_model(vision_path, self.ov_device)
        self.text_model = core.compile_model(text_path, self.ov_device)
        logger.info("  Vision encoder: %s", vision_path.name)
        logger.info("  Text encoder: %s", text_path.name)

        # Load prompt decoder (required)
        prompt_decoder_path = _require_model_file(self.model_dir, _PROMPT_DECODER)
        self.decoder_model = core.compile_model(prompt_decoder_path, self.ov_device)
        logger.info("  Prompt decoder: %s", prompt_decoder_path.name)

        # Load geometry encoders (optional — needed for box/point/exemplar)
        geo_path = _find_model_file(self.model_dir, _GEOMETRY_ENCODER)
        if geo_path is not None:
            self.geometry_model = core.compile_model(geo_path, self.ov_device)
            logger.info("  Geometry encoder (classic): %s", geo_path.name)
        else:
            self.geometry_model = None

        geo_ex_path = _find_model_file(self.model_dir, _GEOMETRY_ENCODER_EXEMPLAR)
        if geo_ex_path is not None:
            self.geometry_exemplar_model = core.compile_model(geo_ex_path, self.ov_device)
            logger.info("  Geometry encoder (exemplar): %s", geo_ex_path.name)
        else:
            self.geometry_exemplar_model = None

        # Pre-create infer requests to avoid per-call allocation overhead (GPU)
        self._vision_request = self.vision_model.create_infer_request()
        self._text_request = self.text_model.create_infer_request()
        self._decoder_request = self.decoder_model.create_infer_request()
        self._geometry_request = self.geometry_model.create_infer_request() if self.geometry_model is not None else None
        self._geometry_exemplar_request = (
            self.geometry_exemplar_model.create_infer_request() if self.geometry_exemplar_model is not None else None
        )

        self.image_preprocessor = Sam3Preprocessor(target_size=resolution)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=resolution)
        self.postprocessor = Sam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
        )

        # Tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        logger.info("SAM3 OpenVINO model loaded successfully (mode=%s).", prompt_mode.value)

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------

    def _load_tokenizer(self, tokenizer_path: str | Path | None) -> CLIPTokenizerFast:
        """Load CLIP tokenizer from local path or HuggingFace.

        Args:
            tokenizer_path: Explicit path/repo, or ``None`` for auto-detection.

        Returns:
            Loaded ``CLIPTokenizerFast`` instance.
        """
        if tokenizer_path is not None:
            return CLIPTokenizerFast.from_pretrained(str(tokenizer_path))
        if (self.model_dir / "tokenizer.json").exists():
            return CLIPTokenizerFast.from_pretrained(str(self.model_dir))
        return CLIPTokenizerFast.from_pretrained(_DEFAULT_HF_REPO)

    # ------------------------------------------------------------------
    # Model resolution / download
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_model_dir(
        model_dir: str | Path | None,
        *,
        variant: SAM3OVVariant = SAM3OVVariant.FP16,
        repo_id: str = _DEFAULT_HF_REPO,
    ) -> Path:
        """Resolve the model directory, downloading from HuggingFace if needed.

        Args:
            model_dir: Explicit local directory, or ``None`` for auto-download.
            variant: Model variant to download.
            repo_id: HuggingFace repository ID.

        Returns:
            Local ``Path`` to the directory containing the sub-model files.

        Raises:
            FileNotFoundError: If *model_dir* is given but does not exist.
            ImportError: If ``huggingface_hub`` is not installed.
        """
        if model_dir is not None:
            path = Path(model_dir)
            if not path.is_dir():
                msg = f"Model directory not found: {path}"
                raise FileNotFoundError(msg)
            return path

        variant = SAM3OVVariant(variant)
        subdir = variant.value

        try:
            from huggingface_hub import snapshot_download  # noqa: PLC0415
        except ImportError:
            msg = "huggingface_hub is required for auto-download. Install it with: uv pip install huggingface-hub"
            raise ImportError(msg)  # noqa: B904

        logger.info("Downloading SAM3 '%s' from HuggingFace: %s", variant.name, repo_id)
        cache_dir = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[f"{subdir}/*", "tokenizer*", "special_tokens_map*"],
        )
        return Path(cache_dir) / subdir

    # ------------------------------------------------------------------
    # Sub-model runners
    # ------------------------------------------------------------------

    def _run_vision_encoder(self, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
        """Run vision encoder.

        Args:
            pixel_values: ``[1, 3, H, W]`` float32.

        Returns:
            Dict with ``fpn_feat_0``, ``fpn_feat_1``, ``fpn_feat_2``, ``fpn_pos_2``.
        """
        self._vision_request.infer([pixel_values])
        return {
            "fpn_feat_0": np.array(self._vision_request.get_tensor("fpn_feat_0").data),
            "fpn_feat_1": np.array(self._vision_request.get_tensor("fpn_feat_1").data),
            "fpn_feat_2": np.array(self._vision_request.get_tensor("fpn_feat_2").data),
            "fpn_pos_2": np.array(self._vision_request.get_tensor("fpn_pos_2").data),
        }

    def _run_text_encoder(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run text encoder.

        Args:
            input_ids: ``[1, 32]`` int64.
            attention_mask: ``[1, 32]`` int64.

        Returns:
            Dict with ``text_features`` and ``text_mask``.
        """
        self._text_request.infer([input_ids, attention_mask])
        return {
            "text_features": np.array(self._text_request.get_tensor("text_features").data),
            "text_mask": np.array(self._text_request.get_tensor("text_mask").data),
        }

    def _run_geometry_encoder(
        self,
        fpn_feat_2: np.ndarray,
        fpn_pos_2: np.ndarray,
        input_boxes: np.ndarray,
        input_boxes_labels: np.ndarray,
        input_points: np.ndarray,
        input_points_labels: np.ndarray,
        *,
        exemplar: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run geometry encoder.

        Args:
            fpn_feat_2: ``[1, 256, H, W]`` float32!
            fpn_pos_2: ``[1, 256, H, W]`` float32.
            input_boxes: ``[1, N, 4]`` cxcywh normalised.
            input_boxes_labels: ``[1, N]`` int64.
            input_points: ``[1, M, 2]`` xy normalised.
            input_points_labels: ``[1, M]`` int64.
            exemplar: Use the exemplar geometry encoder (``drop_spatial_bias=True``).

        Returns:
            Dict with ``geometry_features`` ``[1, K, 256]`` and
            ``geometry_mask`` ``[1, K]``.

        Raises:
            RuntimeError: If the required geometry encoder model is not loaded.
        """
        model = self.geometry_exemplar_model if exemplar else self.geometry_model
        if model is None:
            variant = "exemplar" if exemplar else "classic"
            msg = f"Geometry encoder ({variant}) is not loaded. Re-export models."
            raise RuntimeError(msg)
        request = self._geometry_exemplar_request if exemplar else self._geometry_request
        request.infer([
            fpn_feat_2,
            fpn_pos_2,
            input_boxes,
            input_boxes_labels,
            input_points,
            input_points_labels,
        ])
        return {
            "geometry_features": np.array(request.get_tensor("geometry_features").data),
            "geometry_mask": np.array(request.get_tensor("geometry_mask").data),
        }

    def _run_prompt_decoder(
        self,
        vision_features: dict[str, np.ndarray],
        prompt_features: np.ndarray,
        prompt_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run prompt decoder (DETR pipeline + mask decoder).

        Args:
            vision_features: FPN features from vision encoder.
            prompt_features: ``[1, T, 256]`` pre-concatenated prompt features.
            prompt_mask: ``[1, T]`` bool.

        Returns:
            Dict with ``pred_masks``, ``pred_boxes``, ``pred_logits``,
            ``presence_logits``.
        """
        self._decoder_request.infer([
            vision_features["fpn_feat_0"],
            vision_features["fpn_feat_1"],
            vision_features["fpn_feat_2"],
            vision_features["fpn_pos_2"],
            prompt_features,
            prompt_mask,
        ])
        return {
            "pred_masks": np.array(self._decoder_request.get_tensor("pred_masks").data),
            "pred_boxes": np.array(self._decoder_request.get_tensor("pred_boxes").data),
            "pred_logits": np.array(self._decoder_request.get_tensor("pred_logits").data),
            "presence_logits": np.array(self._decoder_request.get_tensor("presence_logits").data),
        }

    # ------------------------------------------------------------------
    # Tokenisation helpers
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """Tokenise a single text prompt and pad/truncate to length 32.

        Args:
            text: Text prompt string.

        Returns:
            Tuple of ``(input_ids, attention_mask)`` as ``[1, 32]`` int64 arrays.
        """
        text_inputs = self.tokenizer([text], return_tensors="np", padding=True)
        input_ids = self._pad_or_truncate(text_inputs.input_ids.astype(np.int64), 32)
        attention_mask = self._pad_or_truncate(text_inputs.attention_mask.astype(np.int64), 32)
        return input_ids, attention_mask

    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or truncate a 2-D array to the target sequence length.

        Args:
            arr: ``[batch, seq_len]``.
            target_len: Target sequence length.

        Returns:
            ``[batch, target_len]``.
        """
        current_len = arr.shape[1]
        if current_len == target_len:
            return arr
        if current_len > target_len:
            return arr[:, :target_len]
        padding = np.zeros((arr.shape[0], target_len - current_len), dtype=arr.dtype)
        return np.concatenate([arr, padding], axis=1)

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference data.

        * **CLASSIC** mode: stores category mapping only.
        * **VISUAL_EXEMPLAR** mode: encodes reference image prompts into cached
          geometry features for reuse during ``predict()``.

        Args:
            reference: Reference data containing category information and,
                for exemplar mode, images with bboxes/points.
        """
        reference_batch = Batch.collate(reference)
        if self.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR:
            self._fit_visual_exemplar(reference_batch)
        else:
            self._fit_classic(reference_batch)

    def _fit_classic(self, reference_batch: Batch) -> None:
        """Store category mapping (classic mode).

        Args:
            reference_batch: Batch of reference samples.
        """
        self.category_mapping = self._build_category_mapping(reference_batch)

    def _fit_visual_exemplar(self, reference_batch: Batch) -> None:
        """Encode exemplar geometry features from reference images.

        Mirrors the PyTorch ``SAM3._fit_visual_exemplar()`` flow:

        1. For each reference sample, run vision encoder.
        2. Convert all box prompts to point-only (box centre) — point encoding
           transfers better across images than ROI pooling.
        3. Group prompts by category, encode with geometry encoder (exemplar).
        4. Cross-image: concatenate features for the same category.
        5. Encode text prompts and cache everything.

        Args:
            reference_batch: Batch of reference samples with images and prompts.

        Raises:
            ValueError: If no reference samples contain bboxes or points.
        """
        encoded_by_category: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        category_text_map: dict[int, str] = {}

        for sample in reference_batch.samples:
            self._encode_sample_prompts(sample, encoded_by_category, category_text_map)

        if not encoded_by_category:
            msg = "VISUAL_EXEMPLAR mode requires at least one reference sample with bboxes or points."
            raise ValueError(msg)

        # Aggregate per-category features
        geo_features_list: list[np.ndarray] = []
        geo_masks_list: list[np.ndarray] = []
        category_ids: list[int] = []
        text_prompts: list[str] = []

        for cat_id in sorted(encoded_by_category.keys()):
            features_list = encoded_by_category[cat_id]
            if len(features_list) == 1:
                geo_feats, geo_mask = features_list[0]
            else:
                geo_feats = np.concatenate([f[0] for f in features_list], axis=1)
                geo_mask = np.concatenate([f[1] for f in features_list], axis=1)

            geo_features_list.append(geo_feats)
            geo_masks_list.append(geo_mask)
            category_ids.append(cat_id)
            text_prompts.append(category_text_map[cat_id])

        # Encode text prompts
        text_features_list: list[np.ndarray] = []
        text_masks_list: list[np.ndarray] = []
        text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        for prompt in text_prompts:
            if prompt not in text_cache:
                input_ids, attention_mask = self._tokenize(prompt)
                text_out = self._run_text_encoder(input_ids, attention_mask)
                text_cache[prompt] = (text_out["text_features"], text_out["text_mask"])
            text_features_list.append(text_cache[prompt][0])
            text_masks_list.append(text_cache[prompt][1])

        # Store cached features
        self.exemplar_geometry_features = geo_features_list
        self.exemplar_geometry_mask = geo_masks_list
        self.exemplar_text_features = text_features_list
        self.exemplar_text_mask = text_masks_list
        self.exemplar_category_ids = category_ids
        self.category_mapping = self._build_category_mapping(reference_batch)

        # Log shot counts
        shot_info = {
            category_text_map[cat_id]: sum(f[0].shape[1] for f in encoded_by_category[cat_id])
            for cat_id in category_ids
        }
        logger.info(
            "Cached %d category exemplar(s): %s, category_ids=%s",
            len(category_ids),
            shot_info,
            category_ids,
        )

    def _encode_sample_prompts(
        self,
        sample: Sample,
        encoded_by_category: dict[int, list[tuple[np.ndarray, np.ndarray]]],
        category_text_map: dict[int, str],
    ) -> None:
        """Encode one sample's box/point prompts into per-category geometry features.

        All box prompts are converted to point-only (box centre ``[cx, cy]``)
        because point encoding transfers better across images.

        Args:
            sample: Reference sample with image and bboxes/points.
            encoded_by_category: Accumulator mapping cat_id to encoded features.
            category_text_map: Accumulator mapping cat_id to text name.

        Raises:
            ValueError: If the sample has prompts but no image.
        """
        bboxes = sample.bboxes
        points = sample.points
        has_bboxes = bboxes is not None and not (isinstance(bboxes, (np.ndarray, torch.Tensor)) and bboxes.size == 0)
        has_points = points is not None and not (isinstance(points, (np.ndarray, torch.Tensor)) and points.size == 0)

        if not has_bboxes and not has_points:
            return
        if sample.image is None:
            msg = "VISUAL_EXEMPLAR mode requires images in reference samples."
            raise ValueError(msg)

        # Run vision encoder
        image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
        with torch.no_grad():
            pixel_values, original_sizes = self.image_preprocessor(image_tensor)
        vision_features = self._run_vision_encoder(pixel_values.numpy())

        # Build metadata
        num_prompts = max(len(bboxes) if has_bboxes else 0, len(points) if has_points else 0)
        categories = sample.categories if sample.categories is not None else ["visual"] * num_prompts
        category_ids = sample.category_ids if sample.category_ids is not None else [0] * num_prompts

        # Convert all prompts to normalised point coords grouped by category
        category_coords: dict[int, list[np.ndarray]] = defaultdict(list)
        prompts = bboxes if has_bboxes else points

        for prompt, category, raw_cat_id in zip(prompts, categories, category_ids, strict=True):
            if has_bboxes:
                input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=prompt)
                coord = input_boxes[..., :2].numpy()  # box centre [1, 1, 2]
            else:
                _, coord_tensor = self.prompt_preprocessor(original_sizes, input_points=prompt)
                coord = coord_tensor.numpy()
            int_cat_id = int(raw_cat_id)
            category_coords[int_cat_id].append(coord)
            category_text_map[int_cat_id] = category

        # Encode each category's points together (n-shot batching)
        for cat_id, coords_list in category_coords.items():
            all_coords = np.concatenate(coords_list, axis=1)  # [1, N, 2]
            num_pts = all_coords.shape[1]

            # No boxes for exemplar — pass ignore sentinels
            ignore_boxes = np.zeros((1, 1, 4), dtype=np.float32)
            ignore_box_labels = np.full((1, 1), -10, dtype=np.int64)
            point_labels = np.ones((1, num_pts), dtype=np.int64)

            geo_out = self._run_geometry_encoder(
                fpn_feat_2=vision_features["fpn_feat_2"],
                fpn_pos_2=vision_features["fpn_pos_2"],
                input_boxes=ignore_boxes,
                input_boxes_labels=ignore_box_labels,
                input_points=all_coords.astype(np.float32),
                input_points_labels=point_labels,
                exemplar=True,
            )
            encoded_by_category[cat_id].append((
                np.array(geo_out["geometry_features"]),
                np.array(geo_out["geometry_mask"]),
            ))

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images.

        Supports all prompt types: text, box, point, and combined. In
        visual-exemplar mode, uses cached geometry features from ``fit()``.

        Args:
            target: Target data to infer. Accepts Sample, list[Sample], Batch,
                or file paths.

        Returns:
            List of prediction dicts per image with ``pred_masks``,
            ``pred_boxes``, ``pred_labels``.
        """
        if self.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR:
            return self._predict_visual_exemplar(target)
        return self._predict_classic(target)

    def _predict_classic(self, target: Collatable) -> list[dict[str, torch.Tensor]]:  # noqa: PLR0915
        """Classic prediction with per-image text/box/point prompts.

        Args:
            target: Target data.

        Returns:
            List of prediction dicts per image.
        """
        target_batch = Batch.collate(target)
        results = []
        use_fitted_categories = self.category_mapping is not None

        for sample in target_batch.samples:
            img_size = sample.image.shape[-2:]
            bboxes = sample.bboxes if sample.bboxes is not None else []
            points = sample.points if sample.points is not None else []

            # Preprocess image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad():
                pixel_values, original_sizes = self.image_preprocessor(image_tensor)
            vision_features = self._run_vision_encoder(pixel_values.numpy())

            # Determine prompts
            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories or []
                category_ids = sample.category_ids
                num_visual = max(len(bboxes), len(points))
                if num_visual and len(texts) != num_visual:
                    texts = ["visual"] * num_visual

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for text, bbox, point, cat_id in zip_longest(texts, bboxes, points, category_ids, fillvalue=None):
                # Tokenise and encode text
                input_ids, attention_mask = self._tokenize(text or "visual")
                text_out = self._run_text_encoder(input_ids, attention_mask)
                text_features = text_out["text_features"]  # [1, 32, 256]
                text_mask = text_out["text_mask"]  # [1, 32]

                # Prepare geometry prompts (if any)
                has_box = bbox is not None and len(bbox)
                has_point = point is not None and len(point)

                if has_box or has_point:
                    # Encode geometry prompts with separate geometry encoder
                    with torch.no_grad():
                        norm_boxes, norm_points = self.prompt_preprocessor(
                            original_sizes,
                            input_boxes=bbox if has_box else None,
                            input_points=point if has_point else None,
                        )

                    if norm_boxes is not None:
                        ov_boxes = norm_boxes.numpy().astype(np.float32)
                        ov_box_labels = np.ones((1, ov_boxes.shape[1]), dtype=np.int64)
                    else:
                        ov_boxes = np.zeros((1, 1, 4), dtype=np.float32)
                        ov_box_labels = np.full((1, 1), -10, dtype=np.int64)

                    if norm_points is not None:
                        ov_points = norm_points.numpy().astype(np.float32)
                        ov_point_labels = np.ones((1, ov_points.shape[1]), dtype=np.int64)
                    else:
                        ov_points = np.zeros((1, 1, 2), dtype=np.float32)
                        ov_point_labels = np.full((1, 1), -10, dtype=np.int64)

                    geo_out = self._run_geometry_encoder(
                        fpn_feat_2=vision_features["fpn_feat_2"],
                        fpn_pos_2=vision_features["fpn_pos_2"],
                        input_boxes=ov_boxes,
                        input_boxes_labels=ov_box_labels,
                        input_points=ov_points,
                        input_points_labels=ov_point_labels,
                    )

                    # Concatenate text + geometry
                    prompt_features = np.concatenate(
                        [text_features, geo_out["geometry_features"]],
                        axis=1,
                    ).astype(np.float32)
                    prompt_mask = np.concatenate(
                        [text_mask.astype(bool), geo_out["geometry_mask"].astype(bool)],
                        axis=1,
                    )

                    decoder_out = self._run_prompt_decoder(
                        vision_features,
                        prompt_features,
                        prompt_mask,
                    )

                else:
                    # Text-only as prompt features
                    decoder_out = self._run_prompt_decoder(
                        vision_features,
                        text_features.astype(np.float32),
                        text_mask.astype(bool),
                    )

                # Convert outputs and postprocess
                outputs_torch = {
                    "pred_masks": torch.from_numpy(np.array(decoder_out["pred_masks"])),
                    "pred_boxes": torch.from_numpy(np.array(decoder_out["pred_boxes"])),
                    "pred_logits": torch.from_numpy(np.array(decoder_out["pred_logits"])),
                    "presence_logits": torch.from_numpy(np.array(decoder_out["presence_logits"])),
                }

                with torch.no_grad():
                    result = self.postprocessor(outputs_torch, target_sizes=[img_size])

                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"])
                all_boxes.append(boxes_with_scores)
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results

    def _predict_visual_exemplar(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Visual-exemplar prediction using cached geometry features from ``fit()``.

        For each target image, reuses the cached exemplar geometry features
        (extracted from reference images) as prompt conditioning together with
        the cached text features.

        Args:
            target: Target data.

        Returns:
            List of prediction dicts per image.

        Raises:
            RuntimeError: If ``fit()`` has not been called.
        """
        if self.exemplar_geometry_features is None:
            msg = "No cached exemplar features. Call fit() with reference images and bboxes/points first."
            raise RuntimeError(msg)

        target_batch = Batch.collate(target)
        results = []

        for sample in target_batch.samples:
            img_size = sample.image.shape[-2:]

            # Preprocess target image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad():
                pixel_values, _ = self.image_preprocessor(image_tensor)
            vision_features = self._run_vision_encoder(pixel_values.numpy())

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            # Run detection for each cached exemplar
            for geo_feats, geo_mask, text_feats, text_mask, cat_id in zip(
                self.exemplar_geometry_features,
                self.exemplar_geometry_mask,
                self.exemplar_text_features,
                self.exemplar_text_mask,
                self.exemplar_category_ids,
                strict=True,
            ):
                # Concatenate text + geometry features
                prompt_features = np.concatenate(
                    [text_feats, geo_feats],
                    axis=1,
                ).astype(np.float32)
                prompt_mask = np.concatenate(
                    [text_mask.astype(bool), geo_mask.astype(bool)],
                    axis=1,
                )

                decoder_out = self._run_prompt_decoder(
                    vision_features,
                    prompt_features,
                    prompt_mask,
                )

                outputs_torch = {
                    "pred_masks": torch.from_numpy(np.array(decoder_out["pred_masks"])),
                    "pred_boxes": torch.from_numpy(np.array(decoder_out["pred_boxes"])),
                    "pred_logits": torch.from_numpy(np.array(decoder_out["pred_logits"])),
                    "presence_logits": torch.from_numpy(np.array(decoder_out["presence_logits"])),
                }

                with torch.no_grad():
                    result = self.postprocessor(outputs_torch, target_sizes=[img_size])

                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"])
                all_boxes.append(boxes_with_scores)
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _build_category_mapping(reference_batch: Batch) -> dict[str, int]:
        """Build category name → id mapping from reference samples.

        Args:
            reference_batch: Batch of reference samples.

        Returns:
            Mapping from category name to category id.
        """
        mapping: dict[str, int] = {}
        for sample in reference_batch.samples:
            if sample.categories is None or sample.category_ids is None:
                continue
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in mapping:
                    mapping[category] = int(category_id)
        return mapping

    @staticmethod
    def _aggregate_results(
        all_masks: list[torch.Tensor],
        all_boxes: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Aggregate results from multiple prompt predictions for one image.

        Args:
            all_masks: List of mask tensors.
            all_boxes: List of box tensors.
            all_labels: List of label tensors.
            img_size: Original image size ``(height, width)``.

        Returns:
            Aggregated predictions dict.
        """
        non_empty_masks = [m for m in all_masks if m.numel() > 0]
        non_empty_boxes = [b for b in all_boxes if b.numel() > 0]
        non_empty_labels = [lb for lb in all_labels if lb.numel() > 0]

        if non_empty_masks:
            return {
                "pred_masks": torch.cat(non_empty_masks, dim=0),
                "pred_boxes": torch.cat(non_empty_boxes, dim=0),
                "pred_labels": torch.cat(non_empty_labels, dim=0),
            }
        return {
            "pred_masks": torch.empty(0, *img_size),
            "pred_boxes": torch.empty(0, 5),
            "pred_labels": torch.empty(0, dtype=torch.long),
        }

    def export(
        self,
        export_dir: str | Path = Path("./exports/sam3"),  # noqa: ARG002
        backend: str = "openvino",  # noqa: ARG002
    ) -> Path:
        """Export is not applicable — this model already uses exported models.

        For exporting from PyTorch, use::

            python scripts/export_sam3_openvino.py

        Args:
            export_dir: Not used.
            backend: Not used.

        Returns:
            Path to the model directory.
        """
        msg = (
            "SAM3OpenVINO already uses pre-exported models. "
            "To export from PyTorch, use: python scripts/export_sam3_openvino.py"
        )
        logger.info(msg)
        return self.model_dir
