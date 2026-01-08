# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 model for text and visual prompting.

This model uses efficient student backbones (RepViT, TinyViT) instead of
the full ViT, providing faster inference through knowledge distillation.

Note:
    Stage 1 checkpoints only support point/box prompts, NOT text prompting.
    Full text prompting support requires Stage 1+ fine-tuned weights.
"""

from itertools import zip_longest

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.ops import box_convert

from getiprompt.data.base.batch import Batch
from getiprompt.models.foundation import Sam3Processor
from getiprompt.models.foundation.efficientsam3 import (
    EfficientSAM3BackboneType,
    EfficientSAM3TextEncoderType,
    build_efficientsam3_image_model,
)

from .base import Model


class EfficientSAM3(Model):
    """EfficientSAM3 model for text and visual prompting.

    This model uses efficient student backbones (RepViT, TinyViT) instead of
    the full ViT, providing faster inference while maintaining segmentation
    quality through knowledge distillation.

    **Important: EfficientSAM3 differs from other prompt-based models** in that
    it does NOT require a separate learning phase. Instead, it performs zero-shot
    segmentation directly during inference using:
    - Text prompts (category names) provided in the `categories` field of each sample, OR
    - Visual prompts (bounding boxes) provided in the `bboxes` field of each sample

    At least one of these prompt types must be provided for each sample during inference.

    The key difference from SAM3:
    - Uses lightweight student backbones instead of full ViT
    - Smaller model size with faster inference
    - Optionally uses MobileCLIP text encoders for even more efficiency

    Available backbone types:
    - RepViT: repvit-m0.9, repvit-m1.1, repvit-m2.3
    - TinyViT: tinyvit-5m, tinyvit-11m, tinyvit-21m

    Available text encoder types:
    - sam3-full: Full SAM3 text encoder (default)
    - MobileCLIP-S0, MobileCLIP-S1, MobileCLIP-B: Efficient text encoders

    Examples:
        >>> from getiprompt.models import EfficientSAM3
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> # Create with TinyViT-21M backbone (default)
        >>> model = EfficientSAM3(backbone_type="tinyvit-21m")

        >>> # Create with RepViT backbone and MobileCLIP text encoder
        >>> model = EfficientSAM3(
        ...     backbone_type="repvit-m2.3",
        ...     text_encoder_type="MobileCLIP-S1",
        ... )

        >>> # Example 1: Text-based prompting
        >>> target_image = torch.zeros((3, 1024, 1024))
        >>> target_sample = Sample(
        ...     image=target_image,
        ...     categories=["shoe", "person"],  # Text prompts
        ... )
        >>> target_batch = Batch.collate([target_sample])
        >>> infer_results = model.infer(target_batch)

        >>> # Example 2: Visual prompting with bounding boxes
        >>> target_sample = Sample(
        ...     image=target_image,
        ...     bboxes=np.array([[100, 100, 200, 200]]),  # [x, y, w, h]
        ... )
        >>> target_batch = Batch.collate([target_sample])
        >>> infer_results = model.infer(target_batch)

        >>> isinstance(infer_results, list)
        True
    """

    @staticmethod
    def _setup_autocast(device: str, precision: torch.dtype) -> torch.autocast:
        """Setup autocast context based on device and precision.

        Args:
            device: The device to use ('cuda', 'xpu', or 'cpu').
            precision: The precision to use ('bf16' or 'fp32').

        Returns:
            Autocast context manager.
        """
        # Determine device type and availability
        if device == "xpu" and hasattr(torch, "xpu") and torch.xpu.is_available():
            device_type = "xpu"
            supports_bf16 = precision == torch.bfloat16
        elif device == "cuda" and torch.cuda.is_available():
            device_type = "cuda"
            supports_bf16 = precision == torch.bfloat16
        else:
            # CPU or unsupported device
            device_type = "cpu"
            supports_bf16 = False

        # Setup autocast context
        if supports_bf16:
            return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        return torch.autocast(device_type=device_type, dtype=torch.float32)

    def __init__(
        self,
        bpe_path: str | None = None,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
        resolution: int = 1008,
        precision: str = "fp32",
        checkpoint_path: str | None = None,
        enable_segmentation: bool = True,
        enable_inst_interactivity: bool = False,
        compile_models: bool = False,
        backbone_type: EfficientSAM3BackboneType | str = EfficientSAM3BackboneType.TINYVIT_21M,
        text_encoder_type: EfficientSAM3TextEncoderType | str | None = None,
    ) -> None:
        """Initialize the EfficientSAM3 model.

        Args:
            bpe_path: Path to the BPE tokenizer vocabulary.
            device: The device to use ('cuda', 'xpu', or 'cpu').
            confidence_threshold: The confidence threshold for filtering predictions.
            resolution: The input image resolution.
            precision: The precision to use for the model ('bf16' or 'fp32').
            checkpoint_path: Path to model checkpoint.
            enable_segmentation: Whether to enable segmentation head.
            enable_inst_interactivity: Whether to enable instance interactivity.
            compile_models: Whether to compile the models.
            backbone_type: Type of student backbone to use.
            text_encoder_type: Type of text encoder. If None, uses full SAM3 encoder.
        """
        super().__init__()

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution
        self.backbone_type = backbone_type
        self.text_encoder_type = text_encoder_type

        # Parse precision to torch dtype
        precision_map = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
        }
        precision_dtype = precision_map.get(precision, torch.float32)

        # Setup precision
        self.autocast_ctx = self._setup_autocast(device=device, precision=precision_dtype)

        # Build the EfficientSAM3 model
        self.model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            device=device,
            checkpoint_path=checkpoint_path,
            enable_segmentation=enable_segmentation,
            enable_inst_interactivity=enable_inst_interactivity,
            compile=compile_models,
            backbone_type=backbone_type,
            text_encoder_type=text_encoder_type,
        )

        # Create processor (same as SAM3)
        self.processor = Sam3Processor(
            model=self.model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    def _prepare_image(self, image: torch.Tensor | np.ndarray | tv_tensors.Image) -> Image.Image:
        """Convert image to PIL Image format.

        Args:
            image: Input image as tensor or numpy array.

        Returns:
            PIL Image.
        """
        if isinstance(image, Image.Image):
            return image

        # Convert to numpy if tensor
        if isinstance(image, torch.Tensor):
            # Handle (C, H, W) format
            if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
                image = image.permute(1, 2, 0)
            image = image.cpu().numpy()

        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Convert to PIL
        if image.ndim == 2:
            return Image.fromarray(image, mode="L")
        if image.shape[-1] == 1:
            return Image.fromarray(image.squeeze(-1), mode="L")
        if image.shape[-1] == 3:
            return Image.fromarray(image, mode="RGB")
        # Handle 4-channel images
        return Image.fromarray(image[..., :3], mode="RGB")

    def _process_predictions(
        self,
        inference_state: dict,
        cat_id: int,
        img_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process predictions from inference state.

        Args:
            inference_state: The inference state containing predictions.
            cat_id: Category ID for labels.
            img_size: The image size (height, width).

        Returns:
            Tuple of (processed_masks, boxes_with_scores, labels).
        """
        # Get predictions from state
        masks = inference_state.get("masks", torch.empty(0, *img_size))
        boxes = inference_state.get("boxes", torch.empty(0, 4))
        scores = inference_state.get("scores", torch.empty(0))
        labels = torch.full((boxes.shape[0],), cat_id, dtype=torch.long, device=boxes.device)

        # Process masks
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

        # Add scores to boxes
        if boxes.numel() > 0 and scores.numel() > 0:
            boxes = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)  # [N, 4] -> [N, 5]
        return masks, boxes, labels

    def _aggregate_results(
        self,
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
        non_empty_masks = [m for m in all_masks if m.numel() > 0]
        non_empty_boxes = [b for b in all_boxes if b.numel() > 0]
        non_empty_labels = [lbl for lbl in all_labels if lbl.numel() > 0]

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

    @staticmethod
    def normalize_boxes(boxes: torch.Tensor, img_size: tuple[int, int]) -> torch.Tensor:
        """Normalize boxes from absolute to relative coordinates."""
        img_h, img_w = img_size
        boxes = boxes.clone().to(torch.float32)
        boxes[:, [0, 2]] /= img_w  # x1, x2
        boxes[:, [1, 3]] /= img_h  # y1, y2
        return box_convert(boxes, "xyxy", "cxcywh")

    def predict(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images.

        Uses batch image encoding for efficiency when processing multiple images.

        Args:
            target_batch: Batch of target samples with images and prompts.

        Returns:
            List of prediction dictionaries with keys:
            - pred_masks: Predicted segmentation masks [N, H, W]
            - pred_boxes: Predicted bounding boxes with scores [N, 5]
            - pred_labels: Predicted category labels [N]
        """
        results = []
        samples = target_batch.samples

        with self.autocast_ctx:
            # Batch encode all images at once (expensive backbone forward pass)
            pil_images = [self._prepare_image(sample.image) for sample in samples]
            batch_state = self.processor.set_image_batch(pil_images)

            # Process each image's prompts individually
            for idx, sample in enumerate(samples):
                img_size = sample.image.shape[-2:]
                bboxes = self.normalize_boxes(sample.bboxes, img_size) if sample.bboxes is not None else []
                texts = sample.categories if sample.categories is not None else []
                category_ids = sample.category_ids

                # Extract single-image state from batch state
                inference_state = self.processor.get_single_image_state(batch_state, idx)

                all_masks: list[torch.Tensor] = []
                all_boxes: list[torch.Tensor] = []
                all_labels: list[torch.Tensor] = []
                for text, bbox, cat_id in zip_longest(texts, bboxes, category_ids, fillvalue=None):
                    inference_state = self.processor.set_prompt(inference_state, text=text, box=bbox)
                    pred_masks, pred_boxes, pred_labels = self._process_predictions(inference_state, cat_id, img_size)
                    all_masks.append(pred_masks)
                    all_boxes.append(pred_boxes)
                    all_labels.append(pred_labels)
                    self.processor.reset_all_prompts(inference_state)

                pred_result = self._aggregate_results(all_masks, all_boxes, all_labels, img_size)
                self.processor.reset_all_prompts(inference_state)
                results.append(pred_result)

        return results

    def fit(self, reference_batch: Batch) -> None:
        """No-op for EfficientSAM3 as it performs zero-shot inference.

        EfficientSAM3 does not require a learning phase. This method exists
        only for API compatibility with the base Model class.

        Args:
            reference_batch: Ignored. EfficientSAM3 uses prompts directly.
        """
        # EfficientSAM3 is a zero-shot model - no training/fitting required

    def export(self, export_dir: str, backend: str = "onnx", **kwargs) -> str:
        """Export the model (not yet implemented).

        Args:
            export_dir: Directory to export the model to.
            backend: Export backend (e.g., 'onnx').
            **kwargs: Additional export arguments.

        Returns:
            Path to exported model.

        Raises:
            NotImplementedError: Export is not yet supported for EfficientSAM3.
        """
        raise NotImplementedError("Export is not yet supported for EfficientSAM3")
