# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

from itertools import zip_longest

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors
from torchvision.ops import box_convert

from getiprompt.data.base.batch import Batch

from .base import Model


class SAM3(Model):
    """SAM3 model for text and visual prompting.

    This model uses SAM3 (Segment Anything Model 3) for zero-shot segmentation
    using either text prompts or visual prompts (bounding boxes).

    **Important: SAM3 differs from other prompt-based models** in that it does NOT
    require a separate learning phase. Instead, it performs zero-shot segmentation
    directly during inference using:
    - Text prompts (category names) provided in the `categories` field of each sample, OR
    - Visual prompts (bounding boxes) provided in the `bboxes` field of each sample

    At least one of these prompt types must be provided for each sample during inference.

    Examples:
        >>> from getiprompt.models import SAM3
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> sam3 = SAM3()

        >>> # Example 1: Text-based prompting
        >>> target_image = torch.zeros((3, 1024, 1024))
        >>> target_sample = Sample(
        ...     image=target_image,
        ...     categories=["shoe", "person"],  # Text prompts
        ... )
        >>> target_batch = Batch.collate([target_sample])
        >>> infer_results = sam3.infer(target_batch)

        >>> # Example 2: Visual prompting with bounding boxes
        >>> target_sample = Sample(
        ...     image=target_image,
        ...     bboxes=np.array([[100, 100, 200, 200]]),  # [x, y, w, h]
        ... )
        >>> target_batch = Batch.collate([target_sample])
        >>> infer_results = sam3.infer(target_batch)

        >>> isinstance(infer_results, list)
        True
    """

    @staticmethod
    def _setup_autocast(device: str, precision: str) -> torch.autocast:
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
            supports_bf16 = precision == "bf16"
        elif device == "cuda" and torch.cuda.is_available():
            device_type = "cuda"
            supports_bf16 = precision == "bf16"
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
        precision: str = "bf16",
        checkpoint_path: str | None = None,
        load_from_HF: bool = True,
        enable_segmentation: bool = True,
        enable_inst_interactivity: bool = False,
        compile_models: bool = False,
    ) -> None:
        """Initialize the SAM3 model.

        Args:
            bpe_path: Path to the BPE tokenizer vocabulary.
            device: The device to use ('cuda', 'xpu', or 'cpu').
            confidence_threshold: The confidence threshold for filtering predictions.
            resolution: The input image resolution.
            precision: The precision to use for the model ('bf16' or 'fp32').
            checkpoint_path: Optional path to model checkpoint.
            load_from_HF: Whether to load checkpoint from HuggingFace.
            enable_segmentation: Whether to enable segmentation head.
            enable_inst_interactivity: Whether to enable instance interactivity.
            compile_models: Whether to compile the models.
        """
        super().__init__()
        from getiprompt.models.foundation import Sam3Processor, build_sam3_image_model

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution

        # Setup precision
        self.autocast_ctx = self._setup_autocast(device=device, precision=precision)

        # Build the SAM3 model
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=device,
            checkpoint_path=checkpoint_path,
            load_from_HF=load_from_HF,
            enable_segmentation=enable_segmentation,
            enable_inst_interactivity=enable_inst_interactivity,
            compile=compile_models,
        )

        # Create processor
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
            img_size: The image size (height, width).
            apply_threshold: Whether to apply threshold to masks.

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
        if all_masks:
            aggregated_masks = torch.cat(all_masks, dim=0)
            aggregated_boxes = torch.cat(all_boxes, dim=0)
            aggregated_labels = torch.cat(all_labels, dim=0)
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

    def infer(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images."""
        results = []
        with self.autocast_ctx:
            for sample in target_batch.samples:
                img_size = sample.image.shape[-2:]
                bboxes = self.normalize_boxes(sample.bboxes, img_size) if sample.bboxes is not None else []
                texts = sample.categories
                category_ids = sample.category_ids
                image = self._prepare_image(sample.image)
                inference_state = self.processor.set_image(image)

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
