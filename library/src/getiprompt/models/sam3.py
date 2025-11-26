# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

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
        img_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process predictions from inference state.

        Args:
            inference_state: The inference state containing predictions.
            img_size: The image size (height, width).
            apply_threshold: Whether to apply threshold to masks.

        Returns:
            Tuple of (processed_masks, boxes_with_scores).
        """
        # Get predictions from state
        masks = inference_state.get("masks", torch.empty(0, *img_size))
        boxes = inference_state.get("boxes", torch.empty(0, 4))
        scores = inference_state.get("scores", torch.empty(0))
        # Process masks
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

        # Add scores to boxes
        if boxes.numel() > 0 and scores.numel() > 0:
            boxes = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)  # [N, 4] -> [N, 5]
        return masks, boxes

    def _aggregate_results(
        self,
        all_masks: list[torch.Tensor],
        all_boxes: list[torch.Tensor],
        all_labels: list,
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
        else:
            # No predictions found
            aggregated_masks = torch.empty(0, *img_size)
            aggregated_boxes = torch.empty(0, 5)
        return {
            "pred_masks": aggregated_masks,
            "pred_boxes": aggregated_boxes,
            "pred_labels": all_labels,
        }

    def _infer_with_text_prompts(
        self,
        categories: list[str],
        inference_state: dict,
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        all_masks = []
        all_boxes = []
        all_labels: list[str] = []

        for category in categories:
            # Set text prompt for single category
            inference_state = self.processor.set_text_prompt(
                state=inference_state,
                prompt=category,  # Single category only
            )

            # Process predictions
            masks, boxes = self._process_predictions(inference_state, img_size)

            # Collect results
            num_predictions = len(masks) if masks.numel() > 0 else 0
            if num_predictions > 0:
                all_masks.append(masks)
                all_boxes.append(boxes)
                all_labels.extend([category] * num_predictions)
            self.processor.reset_all_prompts(inference_state)

        return self._aggregate_results(all_masks, all_boxes, all_labels, img_size)

    def _infer_with_box_prompts(
        self,
        bboxes: torch.Tensor | np.ndarray,
        inference_state: dict,
        img_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)

        all_masks = []
        all_boxes = []
        all_labels: list[int] = []
        height, width = img_size

        for i, bbox in enumerate(bboxes):
            # Convert from [x, y, w, h] to [cx, cy, w, h] normalized format
            x1, y1, x2, y2 = bbox.tolist()
            cx = (x1 + x2) / 2 / width
            cy = (y1 + y2) / 2 / height
            norm_w = (x2 - x1) / width
            norm_h = (y2 - y1) / height

            norm_box_cxcywh = [cx, cy, norm_w, norm_h]
            # Add as positive box prompt
            inference_state = self.processor.add_geometric_prompt(
                state=inference_state,
                box=norm_box_cxcywh,
                label=True,
            )

            # Process predictions
            masks, boxes = self._process_predictions(inference_state, img_size)

            # Collect results
            num_predictions = len(masks) if masks.numel() > 0 else 0
            if num_predictions > 0:
                all_masks.append(masks)
                all_boxes.append(boxes)
                all_labels.extend([i] * num_predictions)
            self.processor.reset_all_prompts(inference_state)

        return self._aggregate_results(all_masks, all_boxes, all_labels, img_size)

    def learn(self, reference_batch: Batch) -> None:
        """SAM3 is a zero-shot model and does NOT actually learn from reference samples."""

    def infer(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images."""
        results = []
        with self.autocast_ctx:
            for sample in target_batch.samples:
                # Convert image to PIL if needed
                img_size = sample.image.shape[-2:]
                image = self._prepare_image(sample.image)
                inference_state = self.processor.set_image(image)

                # Extract text prompts from sample's categories or use dataset categories from learn()
                has_text_prompts = sample.categories is not None and len(sample.categories) > 0
                has_box_prompts = sample.bboxes is not None and len(sample.bboxes) > 0

                # Validate that at least one prompt type is provided
                if not has_text_prompts and not has_box_prompts:
                    msg = (
                        "SAM3 requires at least one prompt type for inference. "
                        "Please provide either 'categories' (text prompts) or 'bboxes' (visual prompts) in each sample, "
                    )
                    raise ValueError(msg)

                if has_text_prompts and has_box_prompts:
                    msg = (
                        "SAM3 does not support both text and box prompts at the same time. "
                        "Please provide either 'categories' (text prompts) or 'bboxes' (visual prompts) in each sample, "
                    )
                    raise ValueError(msg)

                if has_text_prompts:
                    pred_result = self._infer_with_text_prompts(
                        categories=sample.categories,
                        inference_state=inference_state,
                        img_size=img_size,
                    )
                elif has_box_prompts:
                    pred_result = self._infer_with_box_prompts(
                        bboxes=sample.bboxes,
                        inference_state=inference_state,
                        img_size=img_size,
                    )
                else:
                    # Should never reach here due to validation above
                    pred_result = {
                        "pred_masks": torch.empty(0, *img_size),
                        "pred_boxes": torch.empty(0, 5),
                    }
                self.processor.reset_all_prompts(inference_state)
                results.append(pred_result)

        return results
