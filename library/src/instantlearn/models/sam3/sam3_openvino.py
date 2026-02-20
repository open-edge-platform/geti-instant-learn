# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 OpenVINO inference model for text and visual prompting.

This module provides SAM3OpenVINO, which loads pre-exported SAM3 ONNX or OpenVINO IR
models and provides the same inference API as the PyTorch SAM3 model. It supports
text prompts, box prompts, and combined prompts.

The model expects 3 sub-models (v2 split from the usls project):
    - vision-encoder: ViT + FPN backbone
    - text-encoder: CLIP text encoder + projection
    - geo-encoder-mask-decoder: Geometry encoder + DETR encoder/decoder + mask decoder

Models can be loaded from:
    - OpenVINO IR files (.xml/.bin)
    - ONNX files (.onnx)
    - HuggingFace Hub repository

See Also:
    - SAM3: PyTorch-based SAM3 model
    - convert_sam3_to_openvino.py: Script to convert ONNX models to OpenVINO IR
"""

import logging
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

logger = logging.getLogger(__name__)

# Default HuggingFace repo for tokenizer
_DEFAULT_TOKENIZER_REPO = "jetjodh/sam3"

# v2 model file names (canonical)
_VISION_ENCODER = "vision-encoder"
_TEXT_ENCODER = "text-encoder"
_DECODER = "geo-encoder-mask-decoder"


def _find_model_file(model_dir: Path, name: str) -> Path:
    """Find a model file in a directory, supporting OV IR (.xml) and ONNX (.onnx).

    Search order:
      1. ``{name}.xml`` — OpenVINO IR (preferred)
      2. ``{name}.onnx`` — canonical ONNX name
      3. ``{name}-fp16.onnx`` — FP16 ONNX variant
      4. Any remaining ``{name}*.onnx`` — other quantized variants (Q8, Q4F16, etc.)

    Args:
        model_dir: Directory to search.
        name: Base name of the model (without extension).

    Returns:
        Path to the found model file.

    Raises:
        FileNotFoundError: If no matching model file is found.
    """
    # Prefer OpenVINO IR over ONNX
    candidates = [
        model_dir / f"{name}.xml",
        model_dir / f"{name}.onnx",
        model_dir / f"{name}-fp16.onnx",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback: glob for any ONNX variant (e.g., {name}-q8.onnx, {name}-q4f16.onnx)
    onnx_variants = sorted(model_dir.glob(f"{name}*.onnx"))
    if onnx_variants:
        return onnx_variants[0]
    msg = f"Model '{name}' not found in {model_dir}. Expected one of: {[c.name for c in candidates]}"
    raise FileNotFoundError(msg)


class SAM3OpenVINO(Model):
    """SAM3 model for text and visual prompting using OpenVINO inference.

    This model provides the same inference capabilities as the PyTorch SAM3 model
    but runs on OpenVINO runtime for optimized inference on Intel hardware (CPU, GPU).

    It loads 3 pre-exported sub-models (v2 split):
        - Vision encoder: ViT + FPN backbone
        - Text encoder: CLIP text encoder + projection
        - Decoder: Geometry encoder + DETR + mask decoder

    Supports text prompts, box prompts, and combined prompts.

    Examples:
        >>> from instantlearn.models.sam3 import SAM3OpenVINO
        >>> from instantlearn.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> # Load from local directory with OpenVINO IR files
        >>> model = SAM3OpenVINO(model_dir="./sam3-openvino", device="CPU")

        >>> # Example 1: Text prompting via fit()
        >>> ref_sample = Sample(categories=["shoe", "person"], category_ids=[0, 1])
        >>> model.fit(ref_sample)
        >>> target = Sample(image=torch.zeros((3, 1024, 1024)))
        >>> results = model.predict(target)

        >>> # Example 2: Per-sample text prompting
        >>> target = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     categories=["shoe"],
        ...     category_ids=[0],
        ... )
        >>> results = model.predict(target)

        >>> # Example 3: Box prompting
        >>> target = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     bboxes=np.array([[100, 100, 200, 200]]),
        ... )
        >>> results = model.predict(target)
    """

    def __init__(
        self,
        model_dir: str | Path,
        device: str = "CPU",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        tokenizer_path: str | Path | None = None,
    ) -> None:
        """Initialize SAM3 OpenVINO model.

        Args:
            model_dir: Directory containing OpenVINO IR (.xml/.bin) or ONNX (.onnx)
                model files for the 3 v2 sub-models.
            device: OpenVINO device for inference ("CPU", "GPU", "AUTO").
                PyTorch-style names ("cuda", "cpu") are also accepted.
            confidence_threshold: Minimum confidence score for predictions.
            resolution: Input image resolution (must match exported model, typically 1008).
            tokenizer_path: Path to tokenizer directory or HuggingFace model ID.
                If None, loads from the model_dir (if tokenizer.json exists there)
                or falls back to "jetjodh/sam3" from HuggingFace Hub.
        """
        super().__init__()

        self.model_dir = Path(model_dir)
        self.ov_device = device_to_openvino_device(device)
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution

        # Category mapping from fit() — optional
        self.category_mapping: dict[str, int] | None = None

        # Load OpenVINO models
        core = ov.Core()

        vision_path = _find_model_file(self.model_dir, _VISION_ENCODER)
        text_path = _find_model_file(self.model_dir, _TEXT_ENCODER)
        decoder_path = _find_model_file(self.model_dir, _DECODER)

        msg = f"Loading SAM3 OpenVINO models from {self.model_dir} on {self.ov_device}..."
        logger.info(msg)

        self.vision_model = core.compile_model(vision_path, self.ov_device)
        self.text_model = core.compile_model(text_path, self.ov_device)
        self.decoder_model = core.compile_model(decoder_path, self.ov_device)

        logger.info("  Vision encoder: %s", vision_path.name)
        logger.info("  Text encoder: %s", text_path.name)
        logger.info("  Decoder: %s", decoder_path.name)

        # Preprocessors (run on CPU with PyTorch)
        self.image_preprocessor = Sam3Preprocessor(target_size=resolution)
        self.prompt_preprocessor = Sam3PromptPreprocessor(target_size=resolution)
        self.postprocessor = Sam3Postprocessor(
            target_size=resolution,
            threshold=confidence_threshold,
            mask_threshold=0.5,
        )

        # Tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)

        logger.info("SAM3 OpenVINO model loaded successfully.")

    def _load_tokenizer(self, tokenizer_path: str | Path | None) -> CLIPTokenizerFast:
        """Load CLIP tokenizer from local path or HuggingFace.

        Args:
            tokenizer_path: Explicit path/repo, or None for auto-detection.

        Returns:
            Loaded CLIPTokenizerFast instance.
        """
        if tokenizer_path is not None:
            return CLIPTokenizerFast.from_pretrained(str(tokenizer_path))

        # Try model_dir first (if tokenizer.json was copied there)
        if (self.model_dir / "tokenizer.json").exists():
            return CLIPTokenizerFast.from_pretrained(str(self.model_dir))

        # Fall back to HuggingFace
        return CLIPTokenizerFast.from_pretrained(_DEFAULT_TOKENIZER_REPO)

    def fit(self, reference: Sample | list[Sample] | Batch) -> None:
        """Store category mapping from reference data for consistent text prompting.

        This method is optional. If called, the stored categories will be used for all
        predictions. If not called, categories are taken from each target sample.

        Args:
            reference: Reference data containing category information. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)
        self.category_mapping = {}
        for sample in reference_batch.samples:
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)

    def _run_vision_encoder(self, pixel_values: np.ndarray) -> dict[str, np.ndarray]:
        """Run vision encoder inference.

        Args:
            pixel_values: Preprocessed image [1, 3, 1008, 1008] as float32 numpy array.

        Returns:
            Dictionary with FPN features and position encodings.
        """
        result = self.vision_model([pixel_values])
        return {
            "fpn_feat_0": result["fpn_feat_0"],
            "fpn_feat_1": result["fpn_feat_1"],
            "fpn_feat_2": result["fpn_feat_2"],
            "fpn_pos_2": result["fpn_pos_2"],
        }

    def _run_text_encoder(
        self,
        input_ids: np.ndarray,
        attention_mask: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run text encoder inference.

        Args:
            input_ids: Token IDs [1, seq_len] as int64 numpy array.
            attention_mask: Attention mask [1, seq_len] as int64 numpy array.

        Returns:
            Dictionary with text features and mask.
        """
        result = self.text_model([input_ids, attention_mask])
        return {
            "text_features": result["text_features"],
            "text_mask": result["text_mask"],
        }

    def _run_decoder(
        self,
        vision_features: dict[str, np.ndarray],
        text_features: np.ndarray,
        text_mask: np.ndarray,
        input_boxes: np.ndarray,
        input_boxes_labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run decoder (geometry encoder + DETR + mask decoder) inference.

        Args:
            vision_features: FPN features from vision encoder.
            text_features: Text features [1, seq_len, 256] as float32.
            text_mask: Text mask [1, seq_len] as bool.
            input_boxes: Box prompts [1, num_boxes, 4] in cxcywh normalized format.
            input_boxes_labels: Box labels [1, num_boxes] (1=pos, 0=neg, -10=ignore).

        Returns:
            Dictionary with predicted masks, boxes, logits, and presence logits.
        """
        result = self.decoder_model([
            vision_features["fpn_feat_0"],
            vision_features["fpn_feat_1"],
            vision_features["fpn_feat_2"],
            vision_features["fpn_pos_2"],
            text_features,
            text_mask,
            input_boxes,
            input_boxes_labels,
        ])
        return {
            "pred_masks": result["pred_masks"],
            "pred_boxes": result["pred_boxes"],
            "pred_logits": result["pred_logits"],
            "presence_logits": result["presence_logits"],
        }

    @staticmethod
    def _aggregate_results(
        all_masks: list[torch.Tensor],
        all_boxes: list[torch.Tensor],
        all_labels: list[torch.Tensor],
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Aggregate results from multiple prompt predictions for a single image.

        Args:
            all_masks: List of mask tensors from each prompt.
            all_boxes: List of box tensors from each prompt.
            all_labels: List of label tensors from each prompt.
            img_size: Original image size (height, width).

        Returns:
            Dictionary with aggregated predictions.
        """
        non_empty_masks = [m for m in all_masks if m.numel() > 0]
        non_empty_boxes = [b for b in all_boxes if b.numel() > 0]
        non_empty_labels = [label for label in all_labels if label.numel() > 0]

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

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images using OpenVINO inference.

        Supports the same prompt types as the PyTorch SAM3 model:
        - Text prompts (category names) via fit() or per-sample categories
        - Box prompts (bounding boxes) via the bboxes field
        - Combined text + box prompts

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            List of prediction dictionaries per image, each containing:
                "pred_masks": [num_masks, H, W] — binary masks
                "pred_boxes": [num_masks, 5] — boxes with scores (x1, y1, x2, y2, score)
                "pred_labels": [num_masks] — category IDs
        """
        target_batch = Batch.collate(target)
        results = []
        samples = target_batch.samples

        use_fitted_categories = self.category_mapping is not None

        for sample in samples:
            img_size = sample.image.shape[-2:]
            bboxes = sample.bboxes if sample.bboxes is not None else []

            # Preprocess image (PyTorch on CPU)
            image_tensor = sample.image.unsqueeze(0) if sample.image.ndim == 3 else sample.image
            with torch.no_grad():
                pixel_values, original_sizes = self.image_preprocessor(image_tensor)

            # Run vision encoder (OpenVINO)
            vision_features = self._run_vision_encoder(pixel_values.numpy())

            # Determine text prompts and category IDs
            if use_fitted_categories:
                texts = list(self.category_mapping.keys())
                category_ids = list(self.category_mapping.values())
            else:
                texts = sample.categories or []
                category_ids = sample.category_ids
                if len(bboxes) and len(texts) != len(bboxes):
                    texts = ["visual"] * len(bboxes)

            all_masks: list[torch.Tensor] = []
            all_boxes: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []

            for text, bbox, cat_id in zip_longest(texts, bboxes, category_ids, fillvalue=None):
                # Tokenize text prompt
                text_inputs = self.tokenizer([text or "visual"], return_tensors="np", padding=True)
                input_ids = text_inputs.input_ids.astype(np.int64)
                attention_mask = text_inputs.attention_mask.astype(np.int64)

                # Pad/truncate to expected sequence length (32)
                input_ids = self._pad_or_truncate(input_ids, 32)
                attention_mask = self._pad_or_truncate(attention_mask, 32)

                # Run text encoder (OpenVINO)
                text_outputs = self._run_text_encoder(input_ids, attention_mask)

                # Prepare box inputs
                if bbox is not None:
                    with torch.no_grad():
                        box_tensor = self.prompt_preprocessor(bbox, original_sizes)
                    input_boxes = box_tensor.numpy().astype(np.float32)
                    input_boxes_labels = np.ones((1, input_boxes.shape[1]), dtype=np.int64)
                else:
                    # No box prompt — pass sentinel values (-10 = ignore)
                    input_boxes = np.zeros((1, 1, 4), dtype=np.float32)
                    input_boxes_labels = np.full((1, 1), -10, dtype=np.int64)

                # Run decoder (OpenVINO)
                decoder_outputs = self._run_decoder(
                    vision_features=vision_features,
                    text_features=text_outputs["text_features"],
                    text_mask=text_outputs["text_mask"],
                    input_boxes=input_boxes,
                    input_boxes_labels=input_boxes_labels,
                )

                # Convert outputs to torch tensors for postprocessing
                outputs_torch = {
                    "pred_masks": torch.from_numpy(np.array(decoder_outputs["pred_masks"])),
                    "pred_boxes": torch.from_numpy(np.array(decoder_outputs["pred_boxes"])),
                    "pred_logits": torch.from_numpy(np.array(decoder_outputs["pred_logits"])),
                    "presence_logits": torch.from_numpy(np.array(decoder_outputs["presence_logits"])),
                }

                # Postprocess (PyTorch on CPU)
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

    @staticmethod
    def _pad_or_truncate(arr: np.ndarray, target_len: int) -> np.ndarray:
        """Pad or truncate a 2D array to the target sequence length.

        Args:
            arr: Input array of shape [batch, seq_len].
            target_len: Target sequence length.

        Returns:
            Array of shape [batch, target_len].
        """
        current_len = arr.shape[1]
        if current_len == target_len:
            return arr
        if current_len > target_len:
            return arr[:, :target_len]
        # Pad with zeros
        padding = np.zeros((arr.shape[0], target_len - current_len), dtype=arr.dtype)
        return np.concatenate([arr, padding], axis=1)

    def export(
        self,
        export_dir: str | Path = Path("./exports/sam3"),  # noqa: ARG002
        backend: str = "openvino",  # noqa: ARG002
    ) -> Path:
        """Export is not applicable — this model already uses exported models.

        This method exists to satisfy the Model interface. For converting ONNX
        models to OpenVINO IR, use the convert_sam3_to_openvino.py script.

        Args:
            export_dir: Not used.
            backend: Not used.

        Returns:
            Path to the model directory.
        """
        msg = (
            "SAM3OpenVINO already uses pre-exported models. "
            "To convert ONNX → OpenVINO, use: python scripts/convert_sam3_to_openvino.py"
        )
        logger.info(msg)
        return self.model_dir
