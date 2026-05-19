# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 OpenVINO inference model for text, box, point, and visual-exemplar prompting.

This module provides ``SAM3OpenVINO``, which loads pre-exported SAM3 OpenVINO IR
(or ONNX) models and provides the same inference API as the PyTorch ``SAM3`` model.

The model expects 5 sub-models (custom-exported from ``Sam3Model``):

* ``vision-encoder``   — ViT + FPN backbone
* ``text-encoder``     — CLIP text encoder + projection
* ``geometry-encoder`` — Geometry encoder (classic, ``drop_spatial_bias=False``)
* ``geometry-encoder-exemplar`` — Geometry encoder (exemplar, ``drop_spatial_bias=True``)
* ``prompt-decoder``   — DETR encoder/decoder + box refinement + scoring + mask decoder

See Also:
    - ``SAM3``: PyTorch-based SAM3 model
    - ``onnx_wrappers``: ONNX export wrappers for SAM3 sub-components
    - ``export_sam3``: CLI script for exporting and quantizing models
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

from .canvas_helpers import (
    build_canvas_multishot,
    build_canvas_shared_grouped,
    build_canvas_vertical,
    extract_target_predictions,
    group_references_by_category,
    merge_cross_category,
)
from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor
from .sam3 import CanvasConfig, Sam3PromptMode

logger = logging.getLogger(__name__)

# Default HuggingFace repo for SAM3 OpenVINO models 
SAM3_OV_REPO = "rajeshgangireddy/SAM3_OpenVINO"


class SAM3OVVariant(str, Enum):
    """Available SAM3 OpenVINO model variants.

    Each variant maps to a subdirectory name on HuggingFace Hub
    (``rajeshgangireddy/SAM3_OpenVINO``).

    Recommended variants:

    * **FP16** — Best GPU performance (Arc/Xe). Default for GPU inference.
    * **INT8_SYM** — W8A16 weight-only compression via ``compress_weights(INT8_SYM)``.
      ~50% smaller than FP16, 1.6x CPU speedup, no accuracy loss.
    * **INT8_PTQ** — W8A8 post-training quantization via ``nncf.quantize()``.
      Quantizes both weights and activations. 2.1x CPU speedup (VNNI),
      no accuracy loss. Best CPU variant. Slower on GPU (0.7x) due to
      INT8 Q/DQ dispatch overhead on FP16 DPAS units.
    * **ONNX** — Original ONNX exports. Can be loaded directly by OpenVINO
      runtime (auto-converted to IR at load time).

    Other variants (available on HuggingFace but no significant advantage
    over the recommended set):

    * **FP32** — No accuracy or speed benefit over FP16 on GPU.
    * **INT8_ASYM** — Similar to INT8_SYM, no measurable difference.
    * **INT4_SYM / INT4_ASYM** — Smaller models but accuracy degradation
      on text-mode prompting.
    * **INT8_W8A16** — Identical to INT8_SYM (same ``compress_weights``
      with ``INT8_SYM`` mode), kept for backward compatibility.
    """

    # Recommended variants
    FP16 = "openvino-fp16"
    INT8_SYM = "openvino-int8_sym"
    INT8_PTQ = "openvino-int8_ptq_gpu"
    ONNX = "onnx"

    # Other variants (no significant advantage over recommended set)
    FP32 = "openvino-fp32"
    INT8_ASYM = "openvino-int8_asym"
    INT4_SYM = "openvino-int4_sym"
    INT4_ASYM = "openvino-int4_asym"
    INT8_W8A16 = "openvino-int8_w8a16"


# Sub-model file names
_VISION_ENCODER = "vision-encoder"
_TEXT_ENCODER = "text-encoder"
_GEOMETRY_ENCODER = "geometry-encoder"
_GEOMETRY_ENCODER_EXEMPLAR = "geometry-encoder-exemplar"
_PROMPT_DECODER = "prompt-decoder"


def _get_model_file(model_dir: Path, name: str, *, required: bool = True) -> Path | None:
    """Locate a sub-model file in *model_dir*.

    Search order: ``{name}.xml`` (OpenVINO IR), ``{name}.onnx``,
    ``{name}-fp16.onnx``, then any ``{name}*.onnx`` glob match.

    Args:
        model_dir: Directory to search.
        name: Base name of the model (without extension).
        required: If ``True``, raise when the file is missing.

    Returns:
        Path to the model file, or ``None`` when *required* is ``False``
        and no file is found.

    Raises:
        FileNotFoundError: If *required* is ``True`` and no file is found.
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
    if required:
        msg = f"Model '{name}' not found in {model_dir}."
        raise FileNotFoundError(msg)
    return None


class SAM3OpenVINO(Model):
    """SAM3 model using OpenVINO runtime for inference.

    Provides the same capabilities as the PyTorch ``SAM3`` model:

    * **CLASSIC** mode — text, box, point, or combined prompts per target image.
    * **CANVAS** mode — stitch reference crop + target into a single image and
      detect by visual similarity together with text. Default mode.
    * **VISUAL_EXEMPLAR** mode — encode reference-image prompts during ``fit()``
      and reuse cached geometry features for every target image in ``predict()``.

    The model loads pre-exported sub-models from *model_dir*:

    * ``vision-encoder`` (always)
    * ``text-encoder`` (always)
    * ``prompt-decoder`` (always)
    * ``geometry-encoder`` (loaded when present, needed for box/point/canvas)
    * ``geometry-encoder-exemplar`` (loaded on demand for VISUAL_EXEMPLAR mode)

    Examples:
        >>> from instantlearn.models.sam3 import SAM3OpenVINO, Sam3PromptMode
        >>> from instantlearn.models.sam3.sam3 import CanvasConfig
        >>> from instantlearn.data.base.sample import Sample
        >>> import numpy as np

        >>> # Auto-download default variant (INT8_SYM) — canvas mode
        >>> model = SAM3OpenVINO(device="CPU")
        >>> ref = Sample(
        ...     image_path="examples/assets/coco/000000286874.jpg",
        ...     bboxes=np.array([[180, 105, 490, 370]]),
        ...     categories=["elephant"],
        ...     category_ids=[0],
        ... )
        >>> model.fit(ref)
        >>> results = model.predict(
        ...     Sample(image_path="examples/assets/coco/000000390341.jpg"),
        ... )

        >>> # Text-only prompting (classic mode)
        >>> model = SAM3OpenVINO(
        ...     variant=SAM3OVVariant.INT8_SYM,
        ...     prompt_mode=Sam3PromptMode.CLASSIC,
        ...     device="CPU",
        ... )
        >>> model.fit(Sample(categories=["elephant"], category_ids=[0]))
        >>> results = model.predict(
        ...     Sample(image_path="examples/assets/coco/000000286874.jpg", categories=["elephant"]),
        ... )

        >>> # Use a local model directory (no download)
        >>> model = SAM3OpenVINO(model_dir="./sam3-openvino/openvino-int8_sym", device="CPU")
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        device: str = "AUTO",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        prompt_mode: Sam3PromptMode = Sam3PromptMode.CANVAS,
        drop_spatial_bias: bool = True,
        tokenizer_path: str | Path | None = None,
        variant: SAM3OVVariant = SAM3OVVariant.INT8_SYM,
        repo_id: str = SAM3_OV_REPO,
        canvas_config: CanvasConfig | None = None,
        cache_dir: str | Path | None = None,
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
            prompt_mode: ``Sam3PromptMode.CANVAS`` (default),
                ``Sam3PromptMode.CLASSIC``, or
                ``Sam3PromptMode.VISUAL_EXEMPLAR``.
            drop_spatial_bias: Stored for API compatibility but has no
                runtime effect.  The exemplar geometry encoder already has
                ``drop_spatial_bias=True`` baked in at export time.
            tokenizer_path: Explicit tokenizer path or HuggingFace model ID.
            variant: Model variant to download when *model_dir* is ``None``.
            repo_id: HuggingFace repository ID for auto-download.
            canvas_config: Configuration for canvas mode (split ratio, crop
                padding, text caching). See :class:`CanvasConfig`.
                Default: ``None`` (uses ``CanvasConfig()`` defaults).
            cache_dir: Directory for caching compiled OpenVINO models. When set
                (and not ``"CPU"`` device), the OV runtime stores compiled GPU
                kernels here so subsequent loads of the same model are ~7x
                faster. ``None`` (default) uses ``model_dir / ".ov_cache"``.
                Pass an empty string ``""`` to disable caching.
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

        # Canvas mode config and cache (populated in _fit_canvas)
        self.canvas_config = canvas_config or CanvasConfig()
        self._canvas_refs_by_category: dict[int, dict] | None = None
        self._canvas_text_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        core = ov.Core()

        # Compile properties: optimise for single-request latency (default GPU
        # hint is THROUGHPUT which adds overhead for real-time single-image use).
        # From openvino doc  : Models may load slower and consume more memory when 
        # using ov::hint::PerformanceMode::THROUGHPUT compared 
        # to ov::hint::PerformanceMode::LATENCY.
        _compile_props = {"PERFORMANCE_HINT": "LATENCY"} if self.ov_device != "CPU" else {}

        # Model caching: huge startup speedup on GPU (~7x faster recompile).
        # No latency change. Disabled on CPU (CPU compile is already fast).
        # Pass ``cache_dir=""`` to disable explicitly.
        if self.ov_device != "CPU" and cache_dir != "":
            _cache_path = Path(cache_dir) if cache_dir else self.model_dir / ".ov_cache"
            _cache_path.mkdir(parents=True, exist_ok=True)
            _compile_props["CACHE_DIR"] = str(_cache_path)

        # Always required: vision encoder, text encoder, prompt decoder
        vision_path = _get_model_file(self.model_dir, _VISION_ENCODER)
        text_path = _get_model_file(self.model_dir, _TEXT_ENCODER)
        prompt_decoder_path = _get_model_file(self.model_dir, _PROMPT_DECODER)

        logger.info("Loading SAM3 OpenVINO models from %s on %s...", self.model_dir, self.ov_device)
        self.vision_model = core.compile_model(vision_path, self.ov_device, _compile_props)
        self.text_model = core.compile_model(text_path, self.ov_device, _compile_props)
        self.decoder_model = core.compile_model(prompt_decoder_path, self.ov_device, _compile_props)
        logger.info("  Vision encoder: %s", vision_path.name)
        logger.info("  Text encoder: %s", text_path.name)
        logger.info("  Prompt decoder: %s", prompt_decoder_path.name)

        # Geometry encoder (classic) — needed for box/point/canvas prompts
        geo_path = _get_model_file(self.model_dir, _GEOMETRY_ENCODER, required=False)
        if geo_path is not None:
            self.geometry_model = core.compile_model(geo_path, self.ov_device, _compile_props)
            logger.info("  Geometry encoder (classic): %s", geo_path.name)
        else:
            self.geometry_model = None

        # Geometry encoder (exemplar) — loaded on demand for VISUAL_EXEMPLAR mode
        self.geometry_exemplar_model = None
        self._ov_core = core
        self._compile_props = _compile_props

        # Pre-create infer requests to avoid per-call allocation overhead (GPU)
        self._vision_request = self.vision_model.create_infer_request()
        self._text_request = self.text_model.create_infer_request()
        self._decoder_request = self.decoder_model.create_infer_request()
        self._geometry_request = self.geometry_model.create_infer_request() if self.geometry_model is not None else None
        self._geometry_exemplar_request = None

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
        return CLIPTokenizerFast.from_pretrained(SAM3_OV_REPO)

    @staticmethod
    def _resolve_model_dir(
        model_dir: str | Path | None,
        *,
        variant: SAM3OVVariant = SAM3OVVariant.INT8_SYM,
        repo_id: str = SAM3_OV_REPO,
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

    def _ensure_geometry_exemplar_loaded(self) -> None:
        """Lazy-load the geometry-encoder-exemplar model on first use."""
        if self.geometry_exemplar_model is not None:
            return
        geo_ex_path = _get_model_file(self.model_dir, _GEOMETRY_ENCODER_EXEMPLAR)
        self.geometry_exemplar_model = self._ov_core.compile_model(
            geo_ex_path, self.ov_device, self._compile_props
        )
        self._geometry_exemplar_request = self.geometry_exemplar_model.create_infer_request()
        logger.info("  Geometry encoder (exemplar): %s [lazy]", geo_ex_path.name)

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
        if exemplar:
            self._ensure_geometry_exemplar_loaded()
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
        * **CANVAS** mode: stores reference images/bboxes and pre-caches text
          features for canvas-based prediction.

        Args:
            reference: Reference data containing category information and,
                for exemplar/canvas mode, images with bboxes/points.
        """
        reference_batch = Batch.collate(reference)
        if self.prompt_mode == Sam3PromptMode.CANVAS:
            self._fit_canvas(reference_batch)
        elif self.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR:
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
        In canvas mode, stitches reference and target into a single canvas.

        Args:
            target: Target data to infer. Accepts Sample, list[Sample], Batch,
                or file paths.

        Returns:
            List of prediction dicts per image with ``pred_masks``,
            ``pred_boxes``, ``pred_labels``.
        """
        if self.prompt_mode == Sam3PromptMode.CANVAS:
            return self._predict_canvas(target)
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
                category_ids = list(sample.category_ids or [])

            # Keep prompt text and category ids aligned with the effective number of prompts.
            num_prompts = max(len(texts), len(bboxes), len(points))
            if num_prompts:
                if len(texts) == 0:
                    texts = ["visual"] * num_prompts
                elif len(texts) != num_prompts:
                    texts = [texts[0]] * num_prompts

                if len(category_ids) == 0:
                    category_ids = [0] * num_prompts
                elif len(category_ids) != num_prompts:
                    category_ids = [category_ids[0]] * num_prompts

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
                label_id = cat_id if cat_id is not None else 0
                all_labels.append(torch.full((len(result[0]["boxes"]),), label_id, dtype=torch.int64))

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
                label_id = cat_id if cat_id is not None else 0
                all_labels.append(torch.full((len(result[0]["boxes"]),), label_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results

    def _fit_canvas(self, reference_batch: Batch) -> None:
        """Store reference images and bboxes for canvas-based prediction.

        References are grouped by category so that each category gets its own
        canvas at prediction time, enabling multi-category detection.

        Args:
            reference_batch: Batch of reference samples with images and bboxes.

        Raises:
            ValueError: If no reference samples contain bboxes.
        """
        refs_by_category = group_references_by_category(reference_batch.samples)

        self._canvas_refs_by_category = refs_by_category
        self._canvas_text_cache = {}
        self.category_mapping = self._build_category_mapping(reference_batch)

        # Pre-cache text features
        for cat_refs in refs_by_category.values():
            text = cat_refs["text"]
            if text not in self._canvas_text_cache:
                input_ids, attention_mask = self._tokenize(text)
                text_out = self._run_text_encoder(input_ids, attention_mask)
                self._canvas_text_cache[text] = (
                    text_out["text_features"],
                    text_out["text_mask"],
                )

        logger.info(
            "Canvas mode: stored %d reference(s) across %d category(ies), "
            "ratio=%.2f, cached %d text embeddings",
            sum(len(g["images"]) for g in refs_by_category.values()),
            len(refs_by_category),
            self.canvas_config.split_ratio,
            len(self._canvas_text_cache),
        )

    def _predict_canvas(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Canvas prediction: stitch reference + target, run classic pipeline.

        Args:
            target: Target data to infer.

        Returns:
            List of prediction dicts per image.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self._canvas_refs_by_category is None:
            msg = "Canvas mode requires fit() to be called first."
            raise RuntimeError(msg)

        target_batch = Batch.collate(target)
        results = []
        n_categories = len(self._canvas_refs_by_category)

        for sample in target_batch.samples:
            tgt_image = sample.image
            tgt_h, tgt_w = tgt_image.shape[-2:]

            if n_categories == 1:
                result = self._predict_canvas_single_category(tgt_image, tgt_h, tgt_w)
            elif isinstance(self.canvas_config.share_vision, str):
                result = self._predict_canvas_shared(tgt_image, tgt_h, tgt_w)
            else:
                result = self._predict_canvas_sequential(tgt_image, tgt_h, tgt_w)
            results.append(result)

        return results

    def _predict_canvas_single_category(
        self,
        tgt_image: torch.Tensor,
        tgt_h: int,
        tgt_w: int,
    ) -> dict[str, torch.Tensor]:
        """Canvas prediction for single category.

        Args:
            tgt_image: Target image tensor (C, H, W).
            tgt_h: Original target height.
            tgt_w: Original target width.

        Returns:
            Prediction dict.
        """
        cat_id, cat_refs = next(iter(self._canvas_refs_by_category.items()))
        cat_images = cat_refs["images"]
        cat_bboxes = cat_refs["bboxes"]
        cat_text = cat_refs["text"]

        canvas, canvas_bboxes, tgt_region = self._build_category_canvas(
            cat_images, tgt_image, cat_bboxes,
        )

        pred = self._run_canvas_forward(canvas, canvas_bboxes, cat_text)
        remapped = extract_target_predictions(pred, tgt_region, tgt_h, tgt_w)

        boxes = remapped.get("pred_boxes", torch.empty(0, 5))
        if boxes.shape[0] > 0:
            remapped["pred_labels"] = torch.full(
                (boxes.shape[0],), cat_id, dtype=torch.int64,
            )
        else:
            remapped["pred_labels"] = torch.empty(0, dtype=torch.int64)

        return remapped

    def _predict_canvas_shared(
        self,
        tgt_image: torch.Tensor,
        tgt_h: int,
        tgt_w: int,
    ) -> dict[str, torch.Tensor]:
        """Multi-category canvas with shared vision encoder pass.

        Builds one canvas with all categories, runs one vision encoder pass,
        then per-category geometry + decoder passes.

        Args:
            tgt_image: Target image tensor (C, H, W).
            tgt_h: Original target height.
            tgt_w: Original target width.

        Returns:
            Merged prediction dict with cross-category NMS.
        """
        canvas, per_cat_bboxes, tgt_region = self._build_canvas_shared_layout(tgt_image)

        # Single vision encoder pass
        image_tensor = canvas.unsqueeze(0) if canvas.ndim == 3 else canvas
        with torch.no_grad():
            pixel_values, original_sizes = self.image_preprocessor(image_tensor)
        vision_features = self._run_vision_encoder(pixel_values.numpy())

        canvas_size = canvas.shape[-2:]
        all_boxes_list: list[torch.Tensor] = []
        all_masks_list: list[torch.Tensor] = []
        all_labels_list: list[torch.Tensor] = []

        for cat_id, cat_refs in self._canvas_refs_by_category.items():
            cat_bboxes = per_cat_bboxes[cat_id]
            cat_text = cat_refs["text"]

            pred = self._run_canvas_decoder_with_vision(
                vision_features, original_sizes, canvas_size, cat_bboxes, cat_text,
            )
            remapped = extract_target_predictions(pred, tgt_region, tgt_h, tgt_w)
            boxes = remapped.get("pred_boxes", torch.empty(0, 5))
            if boxes.shape[0] > 0:
                all_boxes_list.append(boxes)
                all_masks_list.append(
                    remapped.get("pred_masks", torch.empty(0, tgt_h, tgt_w)),
                )
                all_labels_list.append(
                    torch.full((boxes.shape[0],), cat_id, dtype=torch.int64),
                )

        if all_boxes_list:
            return merge_cross_category(
                all_boxes_list, all_masks_list, all_labels_list, (tgt_h, tgt_w),
            )
        return {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, tgt_h, tgt_w),
            "pred_labels": torch.empty(0, dtype=torch.int64),
        }

    def _predict_canvas_sequential(
        self,
        tgt_image: torch.Tensor,
        tgt_h: int,
        tgt_w: int,
    ) -> dict[str, torch.Tensor]:
        """Multi-category canvas: sequential per-category canvases.

        Args:
            tgt_image: Target image tensor (C, H, W).
            tgt_h: Original target height.
            tgt_w: Original target width.

        Returns:
            Merged prediction dict with cross-category NMS.
        """
        all_boxes_list: list[torch.Tensor] = []
        all_masks_list: list[torch.Tensor] = []
        all_labels_list: list[torch.Tensor] = []

        for cat_id, cat_refs in self._canvas_refs_by_category.items():
            canvas, canvas_bboxes, tgt_region = self._build_category_canvas(
                cat_refs["images"], tgt_image, cat_refs["bboxes"],
            )
            pred = self._run_canvas_forward(canvas, canvas_bboxes, cat_refs["text"])
            remapped = extract_target_predictions(pred, tgt_region, tgt_h, tgt_w)
            boxes = remapped.get("pred_boxes", torch.empty(0, 5))
            if boxes.shape[0] > 0:
                all_boxes_list.append(boxes)
                all_masks_list.append(
                    remapped.get("pred_masks", torch.empty(0, tgt_h, tgt_w)),
                )
                all_labels_list.append(
                    torch.full((boxes.shape[0],), cat_id, dtype=torch.int64),
                )

        if all_boxes_list:
            return merge_cross_category(
                all_boxes_list, all_masks_list, all_labels_list, (tgt_h, tgt_w),
            )
        return {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, tgt_h, tgt_w),
            "pred_labels": torch.empty(0, dtype=torch.int64),
        }

    def _build_category_canvas(
        self,
        cat_images: list[torch.Tensor],
        tgt_image: torch.Tensor,
        cat_bboxes: list[np.ndarray],
    ) -> tuple[torch.Tensor, list[np.ndarray], tuple[int, int, int, int]]:
        """Build canvas for a single category's references.

        Args:
            cat_images: Reference images for this category.
            tgt_image: Target image tensor.
            cat_bboxes: Reference bounding boxes.

        Returns:
            (canvas, canvas_bboxes, tgt_region).
        """
        if len(cat_images) == 1:
            canvas, cbbox, tgt_region = build_canvas_vertical(
                cat_images[0], tgt_image, cat_bboxes[0],
                self.canvas_config.split_ratio,
            )
            return canvas, [cbbox], tgt_region
        return build_canvas_multishot(
            cat_images, tgt_image, cat_bboxes,
            self.canvas_config.split_ratio,
            self.canvas_config.crop_padding,
        )

    def _build_canvas_shared_layout(
        self,
        tgt_image: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[int, list[np.ndarray]], tuple[int, int, int, int]]:
        """Build a single canvas with all categories' refs in the reference strip.

        Uses grouped layout: same-category refs packed side-by-side.

        Args:
            tgt_image: Target image tensor (C, H, W).

        Returns:
            (canvas, per_cat_bboxes, tgt_region).
        """
        return build_canvas_shared_grouped(
            self._canvas_refs_by_category,
            tgt_image,
            self.canvas_config.split_ratio,
        )

    def _run_canvas_forward(
        self,
        canvas: torch.Tensor,
        canvas_bboxes: list[np.ndarray],
        text: str,
    ) -> dict[str, torch.Tensor]:
        """Run full OV pipeline on a canvas image with box prompts.

        Args:
            canvas: Canvas image tensor (C, H, W).
            canvas_bboxes: Bounding boxes on the canvas.
            text: Text prompt for this category.

        Returns:
            Prediction dict with pred_boxes and pred_masks.
        """
        canvas_size = canvas.shape[-2:]
        image_tensor = canvas.unsqueeze(0) if canvas.ndim == 3 else canvas

        with torch.no_grad():
            pixel_values, original_sizes = self.image_preprocessor(image_tensor)
        vision_features = self._run_vision_encoder(pixel_values.numpy())

        return self._run_canvas_decoder_with_vision(
            vision_features, original_sizes, canvas_size, canvas_bboxes, text,
        )

    def _run_canvas_decoder_with_vision(
        self,
        vision_features: dict[str, np.ndarray],
        original_sizes: torch.Tensor,
        canvas_size: tuple[int, int],
        canvas_bboxes: list[np.ndarray],
        text: str,
    ) -> dict[str, torch.Tensor]:
        """Run geometry encoder + prompt decoder using pre-computed vision features.

        Args:
            vision_features: FPN features from vision encoder.
            original_sizes: Original sizes from preprocessor.
            canvas_size: (H, W) of the canvas.
            canvas_bboxes: Bounding boxes on the canvas for this category.
            text: Text prompt for this category.

        Returns:
            Prediction dict with pred_boxes and pred_masks.
        """
        # Get text features (cached or compute)
        if self.canvas_config.cache_text and text in self._canvas_text_cache:
            text_features, text_mask = self._canvas_text_cache[text]
        else:
            input_ids, attention_mask = self._tokenize(text)
            text_out = self._run_text_encoder(input_ids, attention_mask)
            text_features = text_out["text_features"]
            text_mask = text_out["text_mask"]

        all_masks: list[torch.Tensor] = []
        all_boxes: list[torch.Tensor] = []

        for bbox in canvas_bboxes:
            # Normalize bbox for geometry encoder
            with torch.no_grad():
                norm_boxes, _ = self.prompt_preprocessor(
                    original_sizes, input_boxes=bbox,
                )

            ov_boxes = norm_boxes.numpy().astype(np.float32)
            ov_box_labels = np.ones((1, ov_boxes.shape[1]), dtype=np.int64)
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

            # Concatenate text + geometry features
            prompt_features = np.concatenate(
                [text_features, geo_out["geometry_features"]],
                axis=1,
            ).astype(np.float32)
            prompt_mask = np.concatenate(
                [text_mask.astype(bool), geo_out["geometry_mask"].astype(bool)],
                axis=1,
            )

            decoder_out = self._run_prompt_decoder(
                vision_features, prompt_features, prompt_mask,
            )

            outputs_torch = {
                "pred_masks": torch.from_numpy(np.array(decoder_out["pred_masks"])),
                "pred_boxes": torch.from_numpy(np.array(decoder_out["pred_boxes"])),
                "pred_logits": torch.from_numpy(np.array(decoder_out["pred_logits"])),
                "presence_logits": torch.from_numpy(np.array(decoder_out["presence_logits"])),
            }

            with torch.no_grad():
                result = self.postprocessor(outputs_torch, target_sizes=[canvas_size])

            boxes_with_scores = torch.cat(
                [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                dim=1,
            )
            all_masks.append(result[0]["masks"].cpu())
            all_boxes.append(boxes_with_scores.cpu())

        if all_boxes:
            return {
                "pred_boxes": torch.cat(all_boxes, dim=0),
                "pred_masks": torch.cat(all_masks, dim=0),
            }
        return {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, *canvas_size),
        }

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

