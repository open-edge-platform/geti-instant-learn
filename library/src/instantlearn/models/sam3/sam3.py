# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

import logging
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum
from itertools import zip_longest

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from transformers import CLIPTokenizerFast

from instantlearn.components.postprocessing import PostProcessor, default_postprocessor
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils import precision_to_torch_dtype

from .model import Sam3Model
from .post_processing import PostProcessingConfig
from .processing import Sam3Postprocessor, Sam3Preprocessor, Sam3PromptPreprocessor

logger = logging.getLogger(__name__)

SAM3_LIBRARY_MODEL_ID = "facebook/sam3.1"
SAM3_APPLICATION_MODEL_ID = "research21/sam3.1"


@dataclass
class CanvasConfig:
    """Configuration for SAM3 canvas mode.

    Canvas mode stitches reference and target images into a single canvas,
    runs detection, and extracts predictions from the target region.

    Args:
        split_ratio: Fraction of the canvas allocated to the reference strip.
            Lower values give more space to the target image. Must be in
            (0, 1). Default: 0.3.
        crop_padding: Padding factor around the reference bounding box when
            cropping. A factor of 2.0 means the crop region is 2x the bbox
            size. Must be positive. Default: 2.0.
        cache_text: Cache text embeddings across canvas forward passes to
            avoid redundant CLIP encoding. Default: True.
        share_vision: Vision sharing strategy for multi-category canvas mode.
            - ``"auto"``: Groups same-category refs together with gaps between
              categories (equivalent to ``"grouped"``).
            - ``"grouped"``: Same-category refs packed side-by-side, gaps only
              between category groups.
            - ``"spaced"``: Each ref in its own slot with gaps between all refs.
            - ``False``: Sequential per-category canvases (no sharing).

    Examples:
        Use defaults:

        >>> config = CanvasConfig()

        Tune split ratio for small reference objects:

        >>> config = CanvasConfig(split_ratio=0.25, crop_padding=3.0)
    """

    split_ratio: float = 0.3
    crop_padding: float = 2.0
    cache_text: bool = True
    share_vision: bool | str = "auto"

    def __post_init__(self) -> None:
        if not 0 < self.split_ratio < 1:
            msg = f"split_ratio must be in (0, 1), got {self.split_ratio}"
            raise ValueError(msg)
        if self.crop_padding <= 0:
            msg = f"crop_padding must be positive, got {self.crop_padding}"
            raise ValueError(msg)


class Sam3PromptMode(str, Enum):
    """Prompt mode for SAM3 inference.

    Attributes:
        CLASSIC: Original SAM3 behavior. Text/box prompts are provided per target
            image. Boxes are encoded against the target image's own features.
        VISUAL_EXEMPLAR: Cross-image visual query detection. Box prompts on a
            reference image are encoded during fit() and reused for all target
            images. Enables "draw box on image A → detect similar on images B, C, D".
        CANVAS: FSS-SAM3 unified canvas approach. Stitches reference and target
            images into a single canvas, runs CLASSIC mode with the reference bbox
            mapped to canvas coordinates. Best visual-only performance.
    """

    CLASSIC = "classic"
    VISUAL_EXEMPLAR = "visual_exemplar"
    CANVAS = "canvas"


class SAM3(Model):
    """SAM3 model for text and visual prompting.

    This model uses SAM3 (Segment Anything Model 3) for zero-shot segmentation
    using either text prompts or visual prompts (bounding boxes).

    **Important: SAM3 differs from other prompt-based models** in that it does NOT
    require a separate learning phase. Instead, it performs zero-shot segmentation
    directly during inference using:
    - Text prompts (category names) provided via ``fit()`` or per-sample ``categories``, OR
    - Visual prompts (bounding boxes) provided in the ``bboxes`` field of each sample

    At least one of these prompt types must be provided for each sample during inference.

    NOTE: Currently, SAM3 does not work well with torch.bfloat16 precision.

    Prompt Modes:
        **CLASSIC** (default): Original SAM3 behavior. Text/box prompts are
        provided per target image. ``fit()`` only stores category names.

        **VISUAL_EXEMPLAR**: Cross-image visual query detection. During ``fit()``,
        box/point prompts on reference images are encoded into geometry features and
        cached. During ``predict()``, these cached features are reused for each
        target image — no boxes/points needed on targets.

    Examples:
        >>> from instantlearn.models import SAM3
        >>> from instantlearn.models.sam3.sam3 import Sam3PromptMode
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> # Classic mode (default)
        >>> sam3 = SAM3()
        >>> ref_sample = Sample(categories=["shoe", "person"], category_ids=[0, 1])
        >>> sam3.fit(ref_sample)
        >>> results = sam3.predict(Sample(image=torch.zeros((3, 1024, 1024))))

        >>> # Visual exemplar mode with boxes
        >>> sam3_ve = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     bboxes=np.array([[100, 100, 200, 200]]),  # [x1, y1, x2, y2] on reference
        ...     category_ids=np.array([0]),
        ... )
        >>> sam3_ve.fit(ref_sample)
        >>> results = sam3_ve.predict(Sample(image=torch.zeros((3, 1024, 1024))))

        >>> # Visual exemplar mode with points
        >>> sam3_pt = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     points=np.array([[150, 150]]),  # [x, y] on reference
        ...     category_ids=np.array([0]),
        ... )
        >>> sam3_pt.fit(ref_sample)
        >>> results = sam3_pt.predict(Sample(image=torch.zeros((3, 1024, 1024))))

        >>> # N-shot: multiple point prompts for the same category (same image)
        >>> sam3_nshot = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     points=np.array([[100, 100], [200, 300], [400, 500]]),  # 3 shots
        ...     categories=["shoe", "shoe", "shoe"],
        ...     category_ids=np.array([0, 0, 0]),  # same category
        ... )
        >>> sam3_nshot.fit(ref_sample)  # encodes 3 points together
        >>> results = sam3_nshot.predict(Sample(image=torch.zeros((3, 1024, 1024))))

        >>> # N-shot across multiple reference images
        >>> sam3_cross = SAM3(prompt_mode=Sam3PromptMode.VISUAL_EXEMPLAR)
        >>> refs = [
        ...     Sample(
        ...         image=torch.zeros((3, 1024, 1024)),
        ...         points=np.array([[100, 100]]),
        ...         categories=["shoe"],
        ...         category_ids=np.array([0]),
        ...     ),
        ...     Sample(
        ...         image=torch.zeros((3, 1024, 1024)),
        ...         points=np.array([[200, 200]]),
        ...         categories=["shoe"],
        ...         category_ids=np.array([0]),  # same category, different image
        ...     ),
        ... ]
        >>> sam3_cross.fit(refs)  # features concatenated across images
        >>> results = sam3_cross.predict(Sample(image=torch.zeros((3, 1024, 1024))))
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        precision: str = "fp32",
        compile_models: bool = False,
        model_id: str = SAM3_LIBRARY_MODEL_ID,
        post_processing: PostProcessingConfig | None = None,
        prompt_mode: Sam3PromptMode | str = Sam3PromptMode.CLASSIC,
        drop_spatial_bias: bool = False,
        postprocessor: PostProcessor | None = None,
        canvas_config: CanvasConfig | None = None,
    ) -> None:
        """Initialize the SAM3 model.

        Args:
            device: The device to use ('cuda', 'xpu', or 'cpu').
            confidence_threshold: The confidence threshold for filtering predictions.
            resolution: The input image resolution.
            precision: The precision to use for the model ('bf16' or 'fp32').
            compile_models: Whether to compile the models.
            model_id: HuggingFace model ID or local path to load the SAM3 model
                and tokenizer from. Default: SAM3_LIBRARY_MODEL_ID.
            post_processing: Optional post-processing configuration for NMS,
                mask overlap removal, and non-overlapping pixel constraints.
            prompt_mode: Prompt mode for inference. 'classic' for original SAM3
                behavior, 'visual_exemplar' for cross-image visual query detection,
                'canvas' for FSS-SAM3 unified canvas approach.
            drop_spatial_bias: When True and in VISUAL_EXEMPLAR mode, skip
                coordinate projection and position encoding in the geometry
                encoder, keeping only ROI-pooled visual features. This removes
                spatial bias from the reference image position. Default: False.
            postprocessor: Post-processor applied after predict().
                Defaults to :func:`~instantlearn.components.postprocessing.default_postprocessor`
                (MaskIoMNMS + BoxIoMNMS).
            canvas_config: Configuration for canvas mode (split ratio, crop
                padding, text caching, vision sharing). See :class:`CanvasConfig`.
                Default: ``None`` (uses ``CanvasConfig()`` defaults).
        """
        if postprocessor is None:
            postprocessor = default_postprocessor()
        super().__init__(postprocessor=postprocessor)

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.precision = precision
        self.compile_models = compile_models
        self.model_id = model_id
        self.prompt_mode = Sam3PromptMode(prompt_mode)
        self.drop_spatial_bias = drop_spatial_bias
        self.canvas_config = canvas_config or CanvasConfig()

        # Category mapping from fit() - optional for consistency with GroundedSAM
        self.category_mapping: dict[str, int] | None = None

        # Visual exemplar cached features (set during fit in VISUAL_EXEMPLAR mode)
        self.exemplar_geometry_features: list[torch.Tensor] | None = None
        self.exemplar_geometry_mask: list[torch.Tensor] | None = None
        self.exemplar_text_features: list[torch.Tensor] | None = None
        self.exemplar_text_mask: list[torch.Tensor] | None = None
        self.exemplar_category_ids: list[int] | None = None

        # Canvas mode cached reference data (set during fit in CANVAS mode)
        self._canvas_ref_images: list[torch.Tensor] | None = None
        self._canvas_ref_bboxes: list[np.ndarray] | None = None
        self._canvas_ref_text: str | None = None
        self._canvas_refs_by_category: dict[int, dict] | None = None
        self._canvas_text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        # Preprocessors and postprocessor
        self.image_preprocessor = Sam3Preprocessor(target_size=resolution).to(device)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=resolution).to(device)
        self.sam3_postprocessor = Sam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
            post_processing=post_processing,
        ).to(device)

        # Tokenizer for text prompts (still from transformers, but not used in ONNX path)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id)

        self.model = (
            Sam3Model.from_pretrained(
                model_id,
                torch_dtype=precision_to_torch_dtype(precision),
            )
            .to(device)
            .eval()
        )

    # Hook methods for subclass customization

    def _get_autocast_context(self) -> torch.autocast | nullcontext:  # noqa: PLR6301
        """Return the autocast context manager for model inference.

        SAM3 uses no autocast (nullcontext). Subclasses like EfficientSAM3
        override this to enable torch.autocast for mixed-precision inference.

        Returns:
            A context manager (nullcontext or torch.autocast).
        """
        return nullcontext()

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        """Tokenize text prompts with model-specific settings.

        SAM3 uses max_length=32 with default padding. Subclasses can override
        for different tokenizer configurations (e.g. different max_length,
        truncation).

        Args:
            texts: List of text strings to tokenize.

        Returns:
            Tokenizer output dict with input_ids and attention_mask on device.
        """
        text_inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", max_length=32)
        return {
            "input_ids": text_inputs.input_ids.to(self.device),
            "attention_mask": text_inputs.attention_mask.to(self.device),
        }

    # -- Public API --

    @torch.inference_mode()
    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Learn from reference samples.

        In CLASSIC mode, stores category mapping only (no image processing).
        In VISUAL_EXEMPLAR mode, encodes box/point prompts on reference images into
        geometry features and caches them for reuse during predict().

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)

        if self.prompt_mode == Sam3PromptMode.CLASSIC:
            self._fit_classic(reference_batch)
        elif self.prompt_mode == Sam3PromptMode.CANVAS:
            self._fit_canvas(reference_batch)
        else:
            self._fit_visual_exemplar(reference_batch)

    @torch.inference_mode()
    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Perform inference on target images.

        In CLASSIC mode, processes text/box prompts per target image.
        In VISUAL_EXEMPLAR mode, reuses cached exemplar features from fit().

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            List of prediction dicts per image with 'pred_masks', 'pred_boxes',
            'pred_labels'.
        """
        if self.prompt_mode == Sam3PromptMode.VISUAL_EXEMPLAR:
            return self.apply_postprocessing(self._predict_visual_exemplar(target))
        if self.prompt_mode == Sam3PromptMode.CANVAS:
            return self.apply_postprocessing(self._predict_canvas(target))
        return self.apply_postprocessing(self._predict_classic(target))

    @torch.no_grad()
    def export(
        self,
        export_dir: str = "./exports/sam3_canvas",
        text: str = "visual",
    ) -> str:
        """Export the SAM3 canvas inference graph to ONNX and OpenVINO IR.

        Creates a traceable inference graph with frozen text features and
        exports via PyTorch → ONNX → OpenVINO. Canvas construction and
        coordinate remapping stay in Python (not part of the exported model).

        The exported model accepts:
        - canvas_image: [1, 3, H, W] raw canvas image
        - input_boxes: [1, N, 4] normalized bboxes (xyxy, [0,1] range)

        And outputs: (scores, boxes, masks) for detections on the canvas.

        Args:
            export_dir: Directory to save exported model files.
            text: Text prompt to bake into the model. Default: "visual".

        Returns:
            Path to the exported OpenVINO XML file.

        Raises:
            ImportError: If OpenVINO is not installed.
        """
        from pathlib import Path

        from .inference_graph import Sam3CanvasInferenceGraph

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        # Pre-compute and freeze text features
        text_inputs = self._tokenize([text])
        text_input_ids = text_inputs["input_ids"].cpu()
        text_attention_mask = text_inputs["attention_mask"].cpu()

        # Build traceable inference graph on CPU for export
        graph = Sam3CanvasInferenceGraph(
            sam3_model=self.model.cpu(),
            preprocessor=self.image_preprocessor.cpu(),
            postprocessor=self.sam3_postprocessor.cpu(),
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
        ).eval()

        # Dummy inputs for tracing
        dummy_image = torch.randn(1, 3, self.resolution, self.resolution)
        dummy_boxes = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)

        # Export to ONNX first (direct PyTorch → OpenVINO fails on many ops)
        onnx_path = export_path / "sam3_canvas.onnx"
        torch.onnx.export(
            graph,
            args=(dummy_image, dummy_boxes),
            f=onnx_path,
            input_names=["canvas_image", "input_boxes"],
            output_names=["scores", "boxes", "masks"],
            dynamic_axes={
                "canvas_image": {2: "height", 3: "width"},
                "input_boxes": {1: "num_boxes"},
                "scores": {0: "num_detections"},
                "boxes": {0: "num_detections"},
                "masks": {0: "num_detections", 1: "mask_h", 2: "mask_w"},
            },
            opset_version=16,
        )
        logger.info("ONNX export saved to %s", onnx_path)

        # Convert ONNX → OpenVINO IR
        try:
            import openvino  # noqa: PLC0415
        except ImportError as e:
            msg = "OpenVINO is required for IR export. Install with: pip install openvino"
            raise ImportError(msg) from e

        core = openvino.Core()
        try:
            ov_model = core.read_model(str(onnx_path))
        except RuntimeError:
            ov_model = openvino.convert_model(graph, example_input=(dummy_image, dummy_boxes))

        xml_path = export_path / "sam3_canvas.xml"
        openvino.save_model(ov_model, str(xml_path))
        logger.info("OpenVINO IR saved to %s", xml_path)

        # Move model back to original device
        self.model.to(self.device)
        self.image_preprocessor.to(self.device)
        self.sam3_postprocessor.to(self.device)

        return str(xml_path)

    # -- Fit internals --

    def _fit_classic(self, reference_batch: Batch) -> None:
        """Store category mapping from reference batch.

        Args:
            reference_batch: Batch of reference samples.
        """
        self.category_mapping = self._build_category_mapping(reference_batch)

    def _fit_visual_exemplar(self, reference_batch: Batch) -> None:
        """Encode visual exemplar features from reference images and boxes/points.

        Supports n-shot encoding: multiple prompts for the same category are
        batched together in a single geometry encoder call, enabling self-attention
        between shots from the same image. Features from different images for the
        same category are concatenated, giving the DETR decoder richer conditioning.

        This means passing 3 point prompts for "shoe" produces a single, stronger
        exemplar rather than 3 separate weak ones.

        Note:
            Both boxes and points are encoded as point-only because point encoding
            transfers better across images (0.83 vs 0.59 mIoU): grid_sample at a
            single point captures local appearance without averaging over an ROI
            region.

        Args:
            reference_batch: Batch of reference samples with images and bboxes/points.

        Raises:
            ValueError: If no reference samples contain bboxes or points.
        """
        encoded_by_category, category_text_map = self._encode_batch_prompts(reference_batch)

        if not encoded_by_category:
            msg = "VISUAL_EXEMPLAR mode requires at least one reference sample with bboxes or points."
            raise ValueError(msg)

        geometry_features, geometry_masks, category_ids, text_prompts = self._aggregate_category_features(
            encoded_by_category,
            category_text_map,
        )
        self.exemplar_geometry_features = geometry_features
        self.exemplar_geometry_mask = geometry_masks
        self.exemplar_category_ids = category_ids
        self.exemplar_text_features, self.exemplar_text_mask = self._cache_text_features(text_prompts)
        self.category_mapping = self._build_category_mapping(reference_batch)

        # Log shot counts per category
        shot_info = {
            category_text_map[cat_id]: sum(f[0].shape[1] for f in encoded_by_category[cat_id])
            for cat_id in self.exemplar_category_ids
        }
        logger.info(
            "Cached %d category exemplar(s) with n-shot encoding: %s, category_ids=%s",
            len(self.exemplar_category_ids),
            shot_info,
            self.exemplar_category_ids,
        )

    def _encode_batch_prompts(
        self,
        reference_batch: Batch,
    ) -> tuple[dict[int, list[tuple[torch.Tensor, torch.Tensor]]], dict[int, str]]:
        """Encode all samples' box/point prompts into per-category geometry features.

        Iterates over every sample in the batch so that the same category can
        accumulate geometry features from different reference images (cross-image
        n-shot).  For example, two samples each containing a "shoe" box will both
        append to ``encoded_by_category[shoe_id]``; the downstream aggregation step
        concatenates those features to give the DETR decoder richer conditioning.

        Args:
            reference_batch: Batch of reference samples with images and bboxes/points.

        Returns:
            Tuple of (encoded_by_category, category_text_map) where
            encoded_by_category maps cat_id to list of (features, mask) tuples
            and category_text_map maps cat_id to text name.
        """
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        category_text_map: dict[int, str] = {}

        for sample in reference_batch.samples:
            self._encode_sample_prompts(sample, encoded_by_category, category_text_map)

        return encoded_by_category, category_text_map

    def _encode_sample_prompts(
        self,
        sample: Sample,
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        category_text_map: dict[int, str],
    ) -> None:
        """Encode one sample's box/point prompts into per-category geometry features.

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

        # Extract vision features
        image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
        with self._get_autocast_context():
            pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
            vision_embeds = self.model.get_vision_features(pixel_values)
        fpn_hidden_states = vision_embeds["fpn_hidden_states"][:-1]
        fpn_position_encoding = vision_embeds["fpn_position_encoding"][:-1]

        # Build aligned metadata lists
        num_prompts = max(len(bboxes) if has_bboxes else 0, len(points) if has_points else 0)
        categories = sample.categories if sample.categories is not None else ["visual"] * num_prompts
        category_ids = sample.category_ids if sample.category_ids is not None else [0] * num_prompts

        # Convert prompts to point coords grouped by category
        category_coords: dict[int, list[torch.Tensor]] = defaultdict(list)
        prompts = bboxes if has_bboxes else points

        for prompt, category, cat_id in zip(prompts, categories, category_ids, strict=True):
            if has_bboxes:
                input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=prompt)
                coord = input_boxes[..., :2]  # box center (1, 1, 2)
            else:
                _, coord = self.prompt_preprocessor(original_sizes, input_points=prompt)
            cat_id_int = int(cat_id)
            category_coords[cat_id_int].append(coord)
            category_text_map[cat_id_int] = category

        # Encode each category's points together (same-image n-shot batching)
        for cat_id, coords_list in category_coords.items():
            all_coords = torch.cat(coords_list, dim=1)  # [1, N, 2]
            num_points = all_coords.shape[1]

            with self._get_autocast_context():
                geometry_outputs = self.model.geometry_encoder(
                    point_embeddings=all_coords.to(dtype=fpn_hidden_states[0].dtype),
                    point_mask=torch.ones(1, num_points, dtype=torch.bool, device=self.device),
                    point_labels=torch.ones((1, num_points), dtype=torch.long, device=self.device),
                    img_feats=fpn_hidden_states,
                    img_pos_embeds=fpn_position_encoding,
                    drop_spatial_bias=self.drop_spatial_bias,
                )
            encoded_by_category[cat_id].append((
                geometry_outputs["last_hidden_state"],
                geometry_outputs["attention_mask"],
            ))

    @staticmethod
    def _aggregate_category_features(
        encoded_by_category: dict[int, list[tuple[torch.Tensor, torch.Tensor]]],
        category_text_map: dict[int, str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], list[str]]:
        """Merge per-category geometry features across reference images.

        Args:
            encoded_by_category: Per-category encoded features.
            category_text_map: Mapping from cat_id to text name.

        Returns:
            Tuple of (geometry_features, geometry_masks, category_ids, text_prompts).
        """
        all_geometry_features: list[torch.Tensor] = []
        all_geometry_masks: list[torch.Tensor] = []
        all_category_ids: list[int] = []
        all_text_prompts: list[str] = []

        for cat_id in sorted(encoded_by_category.keys()):
            features_list = encoded_by_category[cat_id]
            if len(features_list) == 1:
                geo_feats, geo_mask = features_list[0]
            else:
                geo_feats = torch.cat([f[0] for f in features_list], dim=1)
                geo_mask = torch.cat([f[1] for f in features_list], dim=1)

            all_geometry_features.append(geo_feats)
            all_geometry_masks.append(geo_mask)
            all_category_ids.append(cat_id)
            all_text_prompts.append(category_text_map[cat_id])

        return all_geometry_features, all_geometry_masks, all_category_ids, all_text_prompts

    def _cache_text_features(
        self,
        text_prompts: list[str],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Tokenize and embed text prompts.

        Args:
            text_prompts: Text prompts aligned with exemplar categories.

        Returns:
            Tuple of (text_features, text_masks) per exemplar.
        """
        unique_prompts = list(dict.fromkeys(text_prompts))
        text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for prompt in unique_prompts:
            tokenized = self._tokenize([prompt])
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            with self._get_autocast_context():
                text_outputs = self.model.get_text_features(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
            text_cache[prompt] = (text_outputs.pooler_output, attention_mask.bool())

        return (
            [text_cache[p][0] for p in text_prompts],
            [text_cache[p][1] for p in text_prompts],
        )

    # -- Predict internals --

    def _predict_classic(self, target: Collatable) -> list[dict[str, torch.Tensor]]:  # noqa: PLR0915
        """Classic SAM3 prediction with per-image text/box/point prompts.

        Args:
            target: Target data to infer.

        Returns:
            List of prediction dicts per image.
        """
        target_batch = Batch.collate(target)
        results = []
        samples = target_batch.samples

        # Use stored categories from fit() if available, otherwise use per-sample
        use_fitted_categories = self.category_mapping is not None

        # Process each image's prompts individually
        for sample in samples:
            img_size = sample.image.shape[-2:]
            bboxes = sample.bboxes if sample.bboxes is not None else []
            points = sample.points if sample.points is not None else []

            # Preprocess image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad(), self._get_autocast_context():
                pixel_values, original_sizes = self.image_preprocessor(image_tensor.to(self.device))
                vision_embeds = self.model.get_vision_features(pixel_values)

            # Determine text prompts and category IDs
            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories or []
                category_ids = sample.category_ids
                # Use "visual" placeholder when only bboxes/points are provided
                num_visual_prompts = max(len(bboxes), len(points))
                if num_visual_prompts and len(texts) != num_visual_prompts:
                    texts = ["visual"] * num_visual_prompts

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for text, bbox, point, cat_id in zip_longest(texts, bboxes, points, category_ids, fillvalue=None):
                # Tokenize text prompt (default to "visual" for visual-only prompts)
                tokenized = self._tokenize([text or "visual"])
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]

                # Prepare box inputs if bbox is provided (xyxy format)
                input_boxes = None
                input_boxes_labels = None
                if bbox is not None and len(bbox):
                    input_boxes, _ = self.prompt_preprocessor(original_sizes, input_boxes=bbox)
                    input_boxes_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                # Prepare point inputs if point is provided (xy format)
                input_points = None
                input_points_labels = None
                if point is not None and len(point):
                    _, input_points = self.prompt_preprocessor(original_sizes, input_points=point)
                    input_points_labels = torch.ones((1, 1), dtype=torch.long, device=self.device)

                with torch.no_grad(), self._get_autocast_context():
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                        input_points=input_points,
                        input_points_labels=input_points_labels,
                    )

                # Postprocess — thread pixel-space point prompts through to
                # the postprocessor so point-aware mask filtering is applied.
                pixel_points = None
                pixel_points_labels = None
                if point is not None and len(point):
                    pt = torch.as_tensor(point, dtype=torch.float32)
                    if pt.ndim == 1:
                        pt = pt.unsqueeze(0)  # (2,) -> (1, 2)
                    pixel_points = [pt.to(self.device)]
                    pixel_points_labels = [torch.ones(pt.shape[0], dtype=torch.long, device=self.device)]
                result = self.sam3_postprocessor(
                    outputs,
                    target_sizes=[img_size],
                    input_points=pixel_points,
                    input_points_labels=pixel_points_labels,
                )
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results

    def _predict_visual_exemplar(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Visual exemplar prediction using cached geometry features from fit().

        For each target image, reuses the cached exemplar geometry features
        (extracted from reference images during fit) as prompt conditioning.

        Args:
            target: Target data to infer.

        Returns:
            List of prediction dicts per image.

        Raises:
            RuntimeError: If fit() has not been called.
        """
        if self.exemplar_geometry_features is None:
            msg = "No cached exemplar features. Call fit() with reference images and bboxes first."
            raise RuntimeError(msg)

        target_batch = Batch.collate(target)
        results = []

        for sample in target_batch.samples:
            img_size = sample.image.shape[-2:]

            # Preprocess target image
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad(), self._get_autocast_context():
                pixel_values, _ = self.image_preprocessor(image_tensor.to(self.device))
                vision_embeds = self.model.get_vision_features(pixel_values)

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
                with torch.no_grad(), self._get_autocast_context():
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        text_embeds=text_feats,
                        attention_mask=text_mask.long(),
                        precomputed_geometry_features=geo_feats,
                        precomputed_geometry_mask=geo_mask,
                    )

                # Postprocess — no raw point prompts needed here because
                # geometry features were already encoded during fit().
                result = self.sam3_postprocessor(outputs, target_sizes=[img_size])
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())
                all_labels.append(torch.full((len(result[0]["boxes"]),), cat_id, dtype=torch.int64))

            results.append(self._aggregate_results(all_masks, all_boxes, all_labels, img_size))

        return results

    # -- Canvas mode --

    def _fit_canvas(self, reference_batch: Batch) -> None:
        """Store reference images and bboxes for canvas-based prediction.

        References are grouped by category so that each category gets its own
        canvas at prediction time, enabling multi-category detection.

        Args:
            reference_batch: Batch of reference samples with images and bboxes.

        Raises:
            ValueError: If no reference samples contain bboxes.
        """
        # Per-category storage: {cat_id: {"images": [...], "bboxes": [...], "text": str}}
        refs_by_category: dict[int, dict] = {}

        for sample in reference_batch.samples:
            if sample.bboxes is None or len(sample.bboxes) == 0:
                continue
            bbox = np.asarray(sample.bboxes[0][:4], dtype=np.float32)
            cat_id = int(sample.category_ids[0]) if sample.category_ids is not None else 0
            cat_text = (
                sample.categories[0]
                if sample.categories and sample.categories[0] != "visual"
                else "visual"
            )

            if cat_id not in refs_by_category:
                refs_by_category[cat_id] = {"images": [], "bboxes": [], "text": cat_text}
            refs_by_category[cat_id]["images"].append(sample.image)
            refs_by_category[cat_id]["bboxes"].append(bbox)
            # Keep the most specific (non-"visual") text for this category
            if cat_text != "visual":
                refs_by_category[cat_id]["text"] = cat_text

        if not refs_by_category:
            msg = "CANVAS mode requires at least one reference sample with bboxes."
            raise ValueError(msg)

        self._canvas_refs_by_category = refs_by_category
        # Legacy single-category attrs for backward compatibility
        all_images = [img for g in refs_by_category.values() for img in g["images"]]
        all_bboxes = [bb for g in refs_by_category.values() for bb in g["bboxes"]]
        self._canvas_ref_images = all_images
        self._canvas_ref_bboxes = all_bboxes
        self._canvas_ref_text = next(iter(refs_by_category.values()))["text"]
        self.category_mapping = self._build_category_mapping(reference_batch)

        # Pre-cache text features (T4 optimization)
        self._canvas_text_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for cat_refs in refs_by_category.values():
            text = cat_refs["text"]
            if text not in self._canvas_text_cache:
                tokenized = self._tokenize([text])
                with torch.no_grad(), self._get_autocast_context():
                    text_outputs = self.model.get_text_features(
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                    )
                self._canvas_text_cache[text] = (
                    text_outputs.pooler_output,
                    tokenized["attention_mask"],
                )

        logger.info(
            "Canvas mode: stored %d reference(s) across %d category(ies), "
            "ratio=%.2f, cached %d text embeddings",
            len(all_images), len(refs_by_category),
            self.canvas_config.split_ratio,
            len(self._canvas_text_cache),
        )

    def _predict_canvas(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Canvas prediction with shared vision encoder and cached text features.

        Routing (multi-category):
        - share_vision='auto'/'grouped'/'spaced': Shared canvas with
          1 ViT pass and per-category decoder passes. Spacing strategy
          controls reference layout in the strip.
        - share_vision=False: Sequential per-category canvases.

        For single-category, builds one canvas and runs once.

        Args:
            target: Target data to infer.

        Returns:
            List of prediction dicts per image.
        """
        if self._canvas_ref_images is None:
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
                result = self._predict_canvas_shared_spaced(
                    tgt_image, tgt_h, tgt_w,
                    spacing=self.canvas_config.share_vision,
                )
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
        """Canvas prediction for single category — direct model call with T4.

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

        pred = self._run_canvas_forward(canvas, canvas_bboxes, cat_text, (tgt_h, tgt_w))
        remapped = self._extract_target_predictions(pred, tgt_region, tgt_h, tgt_w)

        boxes = remapped.get("pred_boxes", torch.empty(0, 5))
        if boxes.shape[0] > 0:
            remapped["pred_labels"] = torch.full(
                (boxes.shape[0],), cat_id, dtype=torch.int64,
            )
        else:
            remapped["pred_labels"] = torch.empty(0, dtype=torch.int64)

        return remapped

    def _predict_canvas_shared_spaced(
        self,
        tgt_image: torch.Tensor,
        tgt_h: int,
        tgt_w: int,
        spacing: str = "auto",
    ) -> dict[str, torch.Tensor]:
        """Shared canvas with spaced-apart references (1 ViT, per-cat decoder).

        Builds ONE canvas with all category references placed in the
        reference strip according to the spacing strategy. Runs a single
        ViT pass, then per-category decoder passes — each decoder call
        only gets the bboxes for its own category's reference region.

        Args:
            tgt_image: Target image tensor (C, H, W).
            tgt_h: Original target height.
            tgt_w: Original target width.
            spacing: Layout strategy ('auto', 'grouped', or 'spaced').

        Returns:
            Merged prediction dict with cross-category NMS.
        """
        canvas, per_cat_bboxes, tgt_region = self._build_canvas_shared_spaced(
            tgt_image, spacing=spacing,
        )

        # Single ViT pass on the shared canvas
        img_tensor = canvas.unsqueeze(0) if canvas.ndim == 3 else canvas
        with torch.no_grad(), self._get_autocast_context():
            pixel_values, original_sizes = self.image_preprocessor(
                img_tensor.to(self.device),
            )
            vision_embeds = self.model.get_vision_features(pixel_values)

        # Per-category decoder pass using shared vision embeddings
        all_boxes_list: list[torch.Tensor] = []
        all_masks_list: list[torch.Tensor] = []
        all_labels_list: list[torch.Tensor] = []

        for cat_id, cat_refs in self._canvas_refs_by_category.items():
            cat_bboxes = per_cat_bboxes[cat_id]
            pred = self._run_canvas_forward_with_vision(
                vision_embeds, original_sizes,
                canvas.shape[-2:], cat_bboxes, cat_refs["text"],
            )
            remapped = self._extract_target_predictions(
                pred, tgt_region, tgt_h, tgt_w,
            )
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
            return self._merge_cross_category(
                all_boxes_list, all_masks_list, all_labels_list, (tgt_h, tgt_w),
            )
        return {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, tgt_h, tgt_w),
            "pred_labels": torch.empty(0, dtype=torch.int64),
        }

    def _build_canvas_shared_spaced(
        self,
        tgt_image: torch.Tensor,
        spacing: str = "auto",
    ) -> tuple[torch.Tensor, dict[int, list[np.ndarray]], tuple[int, int, int, int]]:
        """Build a single canvas with all categories' refs in the reference strip.

        Layout: target on top (1-split_ratio), reference strip on bottom.

        Spacing strategies:
        - "auto": Selects the best layout automatically. Uses "grouped"
          for multi-category scenarios (experimentally optimal).
        - "grouped": Same-category refs packed side-by-side in one group,
          gaps between category groups. E.g. for 2 categories, 2-shot:
          [cat1_ref1 cat1_ref2] [gap] [cat2_ref1 cat2_ref2]
        - "spaced": Every ref gets its own slot with gaps between all.
          E.g. [ref1] [gap] [ref2] [gap] [ref3] [gap] [ref4]

        Each ref uses the FULL reference image (not cropped), resized
        to fit its slot. Bboxes are remapped to canvas coordinates.

        Args:
            tgt_image: Target image tensor (C, H, W).
            spacing: Layout strategy. Default: 'auto'.

        Returns:
            (canvas, per_cat_bboxes, tgt_region) where per_cat_bboxes
            maps cat_id -> list of canvas bboxes for that category.
        """
        if spacing == "auto":
            spacing = "grouped"

        C = tgt_image.shape[0]
        canvas_w = tgt_image.shape[2]
        for cat_refs in self._canvas_refs_by_category.values():
            for img in cat_refs["images"]:
                canvas_w = max(canvas_w, img.shape[2])
        canvas_h = canvas_w

        ref_strip_h = int(canvas_h * self.canvas_config.split_ratio)
        tgt_canvas_h = canvas_h - ref_strip_h

        tgt_resized = F.interpolate(
            tgt_image.unsqueeze(0).float(), size=(tgt_canvas_h, canvas_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        cat_items = list(self._canvas_refs_by_category.items())
        n_cats = len(cat_items)

        ref_strip = torch.zeros(C, ref_strip_h, canvas_w, dtype=tgt_resized.dtype)
        per_cat_bboxes: dict[int, list[np.ndarray]] = {}

        if spacing == "grouped":
            # [cat1_refs] [gap] [cat2_refs] ... — 2K-1 slots
            n_slots = 2 * n_cats - 1
            slot_w = canvas_w // n_slots

            for cat_idx, (cat_id, cat_refs) in enumerate(cat_items):
                group_x = cat_idx * 2 * slot_w
                group_w = slot_w if cat_idx < n_cats - 1 else canvas_w - group_x

                n_refs = len(cat_refs["images"])
                cat_bboxes: list[np.ndarray] = []

                for ref_idx, (ref_img, ref_bbox) in enumerate(zip(
                    cat_refs["images"], cat_refs["bboxes"], strict=True,
                )):
                    sub_x = group_x + ref_idx * (group_w // n_refs)
                    sub_w = (
                        group_w // n_refs
                        if ref_idx < n_refs - 1
                        else group_w - ref_idx * (group_w // n_refs)
                    )

                    ref_h, ref_w = ref_img.shape[1], ref_img.shape[2]
                    ref_resized = F.interpolate(
                        ref_img.unsqueeze(0).float(),
                        size=(ref_strip_h, sub_w),
                        mode="bilinear", align_corners=False,
                    ).squeeze(0)
                    ref_strip[:, :, sub_x:sub_x + sub_w] = ref_resized

                    sx = sub_w / ref_w
                    sy = ref_strip_h / ref_h
                    x1, y1, x2, y2 = ref_bbox[:4]
                    cat_bboxes.append(np.array([
                        x1 * sx + sub_x,
                        y1 * sy + tgt_canvas_h,
                        x2 * sx + sub_x,
                        y2 * sy + tgt_canvas_h,
                    ], dtype=np.float32))

                per_cat_bboxes[cat_id] = cat_bboxes

        else:  # "spaced" — every ref individually separated
            all_refs: list[tuple[int, torch.Tensor, np.ndarray]] = []
            for cat_id, cat_refs in cat_items:
                for ref_img, ref_bbox in zip(
                    cat_refs["images"], cat_refs["bboxes"], strict=True,
                ):
                    all_refs.append((cat_id, ref_img, ref_bbox))

            n_total = len(all_refs)
            n_slots = max(2 * n_total - 1, 1)
            slot_w = canvas_w // n_slots

            for cat_id, _ in cat_items:
                per_cat_bboxes[cat_id] = []

            for ref_idx, (cat_id, ref_img, ref_bbox) in enumerate(all_refs):
                slot_x = ref_idx * 2 * slot_w
                this_w = slot_w if ref_idx < n_total - 1 else canvas_w - slot_x

                ref_h, ref_w = ref_img.shape[1], ref_img.shape[2]
                ref_resized = F.interpolate(
                    ref_img.unsqueeze(0).float(),
                    size=(ref_strip_h, this_w),
                    mode="bilinear", align_corners=False,
                ).squeeze(0)
                ref_strip[:, :, slot_x:slot_x + this_w] = ref_resized

                sx = this_w / ref_w
                sy = ref_strip_h / ref_h
                x1, y1, x2, y2 = ref_bbox[:4]
                per_cat_bboxes[cat_id].append(np.array([
                    x1 * sx + slot_x,
                    y1 * sy + tgt_canvas_h,
                    x2 * sx + slot_x,
                    y2 * sy + tgt_canvas_h,
                ], dtype=np.float32))

        canvas = torch.zeros(C, canvas_h, canvas_w, dtype=tgt_resized.dtype)
        canvas[:, :tgt_canvas_h, :] = tgt_resized
        canvas[:, tgt_canvas_h:, :] = ref_strip

        return canvas, per_cat_bboxes, (0, 0, canvas_w, tgt_canvas_h)

    def _predict_canvas_sequential(
        self,
        tgt_image: torch.Tensor,
        tgt_h: int,
        tgt_w: int,
    ) -> dict[str, torch.Tensor]:
        """Multi-category canvas: sequential ViT per category (T4 only).

        Builds separate per-category canvases and processes each sequentially.
        Only T4 (cached text) optimization is applied.

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
            pred = self._run_canvas_forward(
                canvas, canvas_bboxes, cat_refs["text"], (tgt_h, tgt_w),
            )
            remapped = self._extract_target_predictions(
                pred, tgt_region, tgt_h, tgt_w,
            )
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
            return self._merge_cross_category(
                all_boxes_list, all_masks_list, all_labels_list, (tgt_h, tgt_w),
            )
        return {
            "pred_boxes": torch.empty(0, 5),
            "pred_masks": torch.empty(0, tgt_h, tgt_w),
            "pred_labels": torch.empty(0, dtype=torch.int64),
        }

    def _run_canvas_forward_with_vision(
        self,
        vision_embeds: dict[str, torch.Tensor],
        original_sizes: torch.Tensor,
        canvas_size: tuple[int, int],
        canvas_bboxes: list[np.ndarray],
        text: str,
    ) -> dict[str, torch.Tensor]:
        """Run decoder on pre-computed vision embeddings with cached text.

        Args:
            vision_embeds: Pre-computed vision features from ViT.
            original_sizes: Original image sizes from preprocessor.
            canvas_size: (H, W) of the canvas.
            canvas_bboxes: Bounding boxes on the canvas for this category.
            text: Text prompt for this category.

        Returns:
            Prediction dict with pred_boxes and pred_masks.
        """
        with torch.no_grad(), self._get_autocast_context():
            # T4: use pre-cached text embeddings if available
            text_embeds, attention_mask = (None, None)
            if self.canvas_config.cache_text:
                text_embeds, attention_mask = self._canvas_text_cache.get(
                    text, (None, None),
                )

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []

            for bbox in canvas_bboxes:
                input_boxes, _ = self.prompt_preprocessor(
                    original_sizes, input_boxes=bbox,
                )
                input_boxes_labels = torch.ones(
                    (1, 1), dtype=torch.long, device=self.device,
                )

                if text_embeds is not None:
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        text_embeds=text_embeds,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                    )
                else:
                    tokenized = self._tokenize([text])
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                    )

                result = self.sam3_postprocessor(
                    outputs, target_sizes=[canvas_size],
                )
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
        n_refs = len(cat_images)
        if n_refs == 1:
            canvas, cbbox, tgt_region = self._build_canvas_vertical(
                cat_images[0], tgt_image, cat_bboxes[0],
            )
            return canvas, [cbbox], tgt_region
        return self._build_canvas_multishot(cat_images, tgt_image, cat_bboxes)

    def _run_canvas_forward(
        self,
        canvas: torch.Tensor,
        canvas_bboxes: list[np.ndarray],
        text: str,
        target_size: tuple[int, int],
        vision_embeds: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run model forward on canvas with cached text features.

        This replaces the _predict_classic delegation with a direct model call,
        using pre-cached text embeddings (T4) and optional pre-cached vision
        embeddings (T2).

        Args:
            canvas: Canvas image tensor (C, H, W).
            canvas_bboxes: Bounding boxes on the canvas.
            text: Text prompt for this category.
            target_size: (height, width) of original target image.
            vision_embeds: Pre-computed vision embeddings to reuse. If None,
                computes them from the canvas.

        Returns:
            Prediction dict with pred_boxes, pred_masks, and _vision_embeds
            (for caching by caller).
        """
        img_size = canvas.shape[-2:]
        image_tensor = canvas.unsqueeze(0) if canvas.ndim == 3 else canvas

        with torch.no_grad(), self._get_autocast_context():
            pixel_values, original_sizes = self.image_preprocessor(
                image_tensor.to(self.device),
            )

            # T2: reuse vision embeddings if provided
            if vision_embeds is None:
                vision_embeds = self.model.get_vision_features(pixel_values)

            # T4: use pre-cached text embeddings if available and enabled
            text_embeds, attention_mask = (None, None)
            if self.canvas_config.cache_text:
                text_embeds, attention_mask = self._canvas_text_cache.get(
                    text, (None, None),
                )

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []

            for bbox in canvas_bboxes:
                input_boxes, _ = self.prompt_preprocessor(
                    original_sizes, input_boxes=bbox,
                )
                input_boxes_labels = torch.ones(
                    (1, 1), dtype=torch.long, device=self.device,
                )

                if text_embeds is not None:
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        text_embeds=text_embeds,
                        attention_mask=attention_mask,
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                    )
                else:
                    # Fallback: tokenize on the fly
                    tokenized = self._tokenize([text])
                    outputs = self.model(
                        vision_embeds=vision_embeds,
                        input_ids=tokenized["input_ids"],
                        attention_mask=tokenized["attention_mask"],
                        input_boxes=input_boxes,
                        input_boxes_labels=input_boxes_labels,
                    )

                result = self.sam3_postprocessor(
                    outputs, target_sizes=[img_size],
                )
                boxes_with_scores = torch.cat(
                    [result[0]["boxes"], result[0]["scores"].unsqueeze(1)],
                    dim=1,
                )
                all_masks.append(result[0]["masks"].cpu())
                all_boxes.append(boxes_with_scores.cpu())

        # Aggregate per-bbox results
        if all_boxes:
            merged_boxes = torch.cat(all_boxes, dim=0)
            merged_masks = torch.cat(all_masks, dim=0)
        else:
            merged_boxes = torch.empty(0, 5)
            merged_masks = torch.empty(0, *img_size)

        return {
            "pred_boxes": merged_boxes,
            "pred_masks": merged_masks,
            "_vision_embeds": vision_embeds,
        }

    @staticmethod
    def _merge_cross_category(
        boxes_list: list[torch.Tensor],
        masks_list: list[torch.Tensor],
        labels_list: list[torch.Tensor],
        img_size: tuple[int, int],
        iou_threshold: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Merge per-category predictions with cross-category NMS.

        When the same object is detected by multiple category canvases,
        keeps only the highest-confidence prediction.

        Args:
            boxes_list: Per-category box tensors [N_k, 5] (x1,y1,x2,y2,score).
            masks_list: Per-category mask tensors [N_k, H, W].
            labels_list: Per-category label tensors [N_k].
            img_size: (height, width) of the target image.
            iou_threshold: IoU threshold for cross-category NMS.

        Returns:
            Merged prediction dict.
        """
        from torchvision.ops import nms as torchvision_nms

        all_boxes = torch.cat(boxes_list, dim=0)
        all_masks = torch.cat(masks_list, dim=0) if masks_list else torch.empty(0, *img_size)
        all_labels = torch.cat(labels_list, dim=0)

        if all_boxes.shape[0] == 0:
            return {
                "pred_boxes": torch.empty(0, 5),
                "pred_masks": torch.empty(0, *img_size),
                "pred_labels": torch.empty(0, dtype=torch.int64),
            }

        # Cross-category NMS: treat all detections as one pool
        coords = all_boxes[:, :4]
        scores = all_boxes[:, 4]
        keep = torchvision_nms(coords, scores, iou_threshold)

        return {
            "pred_boxes": all_boxes[keep],
            "pred_masks": all_masks[keep] if all_masks.shape[0] > 0 else torch.empty(0, *img_size),
            "pred_labels": all_labels[keep],
        }

    def _crop_around_bbox(
        self,
        image: torch.Tensor,
        bbox: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Crop image tightly around bbox with padding.

        Args:
            image: (C, H, W) image tensor.
            bbox: [x1, y1, x2, y2] bounding box.

        Returns:
            (cropped_image, adjusted_bbox) in crop coordinates.
        """
        _C, H, W = image.shape
        x1, y1, x2, y2 = bbox[:4].astype(float)
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        half_w = bw * self.canvas_config.crop_padding / 2
        half_h = bh * self.canvas_config.crop_padding / 2

        crop_x1 = int(max(0, cx - half_w))
        crop_y1 = int(max(0, cy - half_h))
        crop_x2 = int(min(W, cx + half_w))
        crop_y2 = int(min(H, cy + half_h))
        crop_x2 = max(crop_x2, crop_x1 + 1)
        crop_y2 = max(crop_y2, crop_y1 + 1)

        crop = image[:, crop_y1:crop_y2, crop_x1:crop_x2]
        adj_bbox = np.array([
            x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1,
        ], dtype=np.float32)
        return crop, adj_bbox

    def _build_canvas_vertical(
        self,
        ref_image: torch.Tensor,
        tgt_image: torch.Tensor,
        ref_bbox: np.ndarray,
    ) -> tuple[torch.Tensor, np.ndarray, tuple[int, int, int, int]]:
        """Build vertical canvas: target on top, reference on bottom.

        Returns:
            (canvas, canvas_bbox, tgt_region) where tgt_region is (x, y, w, h).
        """
        C = ref_image.shape[0]
        ref_h, ref_w = ref_image.shape[1], ref_image.shape[2]
        tgt_h, tgt_w = tgt_image.shape[1], tgt_image.shape[2]

        canvas_w = max(ref_w, tgt_w)
        canvas_h = canvas_w

        ref_canvas_h = int(canvas_h * self.canvas_config.split_ratio)
        tgt_canvas_h = canvas_h - ref_canvas_h

        ref_resized = F.interpolate(
            ref_image.unsqueeze(0).float(), size=(ref_canvas_h, canvas_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        tgt_resized = F.interpolate(
            tgt_image.unsqueeze(0).float(), size=(tgt_canvas_h, canvas_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        canvas = torch.zeros(C, canvas_h, canvas_w, dtype=ref_resized.dtype)
        canvas[:, :tgt_canvas_h, :canvas_w] = tgt_resized
        canvas[:, tgt_canvas_h:, :canvas_w] = ref_resized

        sx = canvas_w / ref_w
        sy = ref_canvas_h / ref_h
        x1, y1, x2, y2 = ref_bbox[:4]
        canvas_bbox = np.array([
            x1 * sx, y1 * sy + tgt_canvas_h,
            x2 * sx, y2 * sy + tgt_canvas_h,
        ], dtype=np.float32)

        return canvas, canvas_bbox, (0, 0, canvas_w, tgt_canvas_h)

    def _build_canvas_multishot(
        self,
        ref_images: list[torch.Tensor],
        tgt_image: torch.Tensor,
        ref_bboxes: list[np.ndarray],
    ) -> tuple[torch.Tensor, list[np.ndarray], tuple[int, int, int, int]]:
        """Build multi-shot canvas: multiple cropped references in a strip.

        Returns:
            (canvas, canvas_bboxes, tgt_region).
        """
        crops, adj_bboxes = [], []
        for ref_img, ref_bbox in zip(ref_images, ref_bboxes, strict=True):
            crop, adj_bbox = self._crop_around_bbox(ref_img, ref_bbox)
            crops.append(crop)
            adj_bboxes.append(adj_bbox)

        C = tgt_image.shape[0]
        canvas_w = max(tgt_image.shape[2], max(c.shape[2] for c in crops))
        canvas_h = canvas_w

        ref_strip_h = int(canvas_h * self.canvas_config.split_ratio)
        tgt_canvas_h = canvas_h - ref_strip_h

        tgt_resized = F.interpolate(
            tgt_image.unsqueeze(0).float(), size=(tgt_canvas_h, canvas_w),
            mode="bilinear", align_corners=False,
        ).squeeze(0)

        n_refs = len(crops)
        crop_w = canvas_w // n_refs
        remainder = canvas_w - crop_w * n_refs

        ref_strip = torch.zeros(C, ref_strip_h, canvas_w, dtype=tgt_resized.dtype)
        canvas_bboxes: list[np.ndarray] = []
        x_offset = 0
        for i, (crop, adj_bbox) in enumerate(zip(crops, adj_bboxes, strict=True)):
            this_w = crop_w + (remainder if i == n_refs - 1 else 0)
            crop_resized = F.interpolate(
                crop.unsqueeze(0).float(), size=(ref_strip_h, this_w),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            ref_strip[:, :, x_offset:x_offset + this_w] = crop_resized

            sx = this_w / crop.shape[2]
            sy = ref_strip_h / crop.shape[1]
            ax1, ay1, ax2, ay2 = adj_bbox[:4]
            canvas_bboxes.append(np.array([
                ax1 * sx + x_offset, ay1 * sy + tgt_canvas_h,
                ax2 * sx + x_offset, ay2 * sy + tgt_canvas_h,
            ], dtype=np.float32))
            x_offset += this_w

        canvas = torch.zeros(C, canvas_h, canvas_w, dtype=tgt_resized.dtype)
        canvas[:, :tgt_canvas_h, :] = tgt_resized
        canvas[:, tgt_canvas_h:, :] = ref_strip

        return canvas, canvas_bboxes, (0, 0, canvas_w, tgt_canvas_h)

    @staticmethod
    def _extract_target_predictions(
        pred: dict[str, torch.Tensor],
        tgt_region: tuple[int, int, int, int],
        tgt_h: int,
        tgt_w: int,
    ) -> dict[str, torch.Tensor]:
        """Extract predictions from the target region and remap to original coords.

        Args:
            pred: Prediction dict with 'pred_boxes' and optionally 'pred_masks'.
            tgt_region: (x, y, w, h) of target region on canvas.
            tgt_h: Original target image height.
            tgt_w: Original target image width.

        Returns:
            Prediction dict with boxes/masks remapped to original target coordinates.
        """
        tx, ty, tw, th = tgt_region
        pred_boxes = pred["pred_boxes"][:, :4].cpu()

        if pred_boxes.shape[0] == 0:
            return {
                "pred_boxes": torch.empty(0, 5),
                "pred_masks": torch.empty(0, tgt_h, tgt_w),
                "pred_labels": torch.empty(0, dtype=torch.int64),
            }

        cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
        cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
        in_target = (cx >= tx) & (cx < tx + tw) & (cy >= ty) & (cy < ty + th)

        scores = (
            pred["pred_boxes"][:, 4].cpu()
            if pred["pred_boxes"].shape[1] > 4
            else torch.ones(len(pred_boxes))
        )
        target_boxes = pred_boxes[in_target]
        target_scores = scores[in_target]

        result: dict[str, torch.Tensor] = {}

        if target_boxes.shape[0] > 0:
            scale_x = tgt_w / tw
            scale_y = tgt_h / th
            remapped = target_boxes.clone()
            remapped[:, 0] = (target_boxes[:, 0] - tx) * scale_x
            remapped[:, 1] = (target_boxes[:, 1] - ty) * scale_y
            remapped[:, 2] = (target_boxes[:, 2] - tx) * scale_x
            remapped[:, 3] = (target_boxes[:, 3] - ty) * scale_y
            remapped[:, 0].clamp_(min=0)
            remapped[:, 1].clamp_(min=0)
            remapped[:, 2].clamp_(max=tgt_w)
            remapped[:, 3].clamp_(max=tgt_h)
            result["pred_boxes"] = torch.cat([remapped, target_scores.unsqueeze(1)], dim=1)
        else:
            result["pred_boxes"] = torch.empty(0, 5)

        # Remap masks if present
        if "pred_masks" in pred and pred["pred_masks"].shape[0] > 0:
            canvas_masks = pred["pred_masks"].cpu()
            target_masks = canvas_masks[in_target]
            if target_masks.shape[0] > 0:
                # Crop mask to target region, then resize to original target size
                target_masks = target_masks[:, ty:ty + th, tx:tx + tw]
                target_masks = F.interpolate(
                    target_masks.unsqueeze(1).float(),
                    size=(tgt_h, tgt_w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
                result["pred_masks"] = (target_masks > 0.5).to(torch.uint8)
            else:
                result["pred_masks"] = torch.empty(0, tgt_h, tgt_w)
        else:
            result["pred_masks"] = torch.empty(0, tgt_h, tgt_w)

        if "pred_labels" in pred:
            result["pred_labels"] = pred["pred_labels"][in_target].cpu()
        else:
            result["pred_labels"] = torch.zeros(result["pred_boxes"].shape[0], dtype=torch.int64)

        return result

    # -- Utilities --

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
        """Aggregate results from multiple predictions.

        Args:
            all_masks: List of mask tensors.
            all_boxes: List of box tensors.
            all_labels: List of labels.
            img_size: The image size (height, width).

        Returns:
            Dictionary with aggregated predictions.
        """
        # Filter out empty tensors before concatenation
        non_empty_masks = [masks for masks in all_masks if masks.numel() > 0]
        non_empty_boxes = [boxes for boxes in all_boxes if boxes.numel() > 0]
        non_empty_labels = [labels for labels in all_labels if labels.numel() > 0]

        if non_empty_masks:
            aggregated_masks = torch.cat(non_empty_masks, dim=0)
            aggregated_boxes = torch.cat(non_empty_boxes, dim=0)
            aggregated_labels = torch.cat(non_empty_labels, dim=0)
        else:
            # No predictions found
            aggregated_masks = torch.empty(0, *img_size)
            aggregated_boxes = torch.empty(0, 5)
            aggregated_labels = torch.empty(0, dtype=torch.long)

        return {
            "pred_masks": aggregated_masks,
            "pred_boxes": aggregated_boxes,
            "pred_labels": aggregated_labels,
        }
