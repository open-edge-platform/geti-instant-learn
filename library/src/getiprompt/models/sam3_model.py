# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 model for text and visual prompting."""

import numpy as np
import torch
from PIL import Image
from torchvision import tv_tensors

from getiprompt.data.base.batch import Batch
from getiprompt.models.sam3.model_builder import build_sam3_image_model
from getiprompt.models.sam3.sam3_image_processor import Sam3Processor

from .base import Model


class SAM3(Model):
    """SAM3 model for text and visual prompting.

    This model uses SAM3 (Segment Anything Model 3) for zero-shot segmentation
    using either text prompts or visual prompts (bounding boxes).

    Examples:
        >>> from getiprompt.models import SAM3Model
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> sam3_model = SAM3Model()

        >>> # Create mock inputs
        >>> ref_image = torch.zeros((3, 1024, 1024))
        >>> target_image = torch.zeros((3, 1024, 1024))

        >>> # Create reference sample with text-based categories
        >>> ref_sample = Sample(
        ...     image=ref_image,
        ...     category_ids=np.array([1]),
        ...     is_reference=[True],
        ...     categories=["shoe"],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=target_image,
        ...     is_reference=[False],
        ...     categories=["shoe"],
        ... )
        >>> target_batch = Batch.collate([target_sample])

        >>> # Run learn and infer
        >>> sam3_model.learn(ref_batch)
        >>> infer_results = sam3_model.infer(target_batch)

        >>> isinstance(infer_results, list)
        True
    """

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
            device: The device to use ('cuda' or 'cpu').
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
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.resolution = resolution

        # Setup precision
        if precision == "bf16" and torch.cuda.is_available():
            self.autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            self.autocast_ctx = torch.autocast("cuda", dtype=torch.float32)

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

        # Store category mapping from learn phase
        self.category_mapping = {}
        self.text_prompts = []

    def learn(self, reference_batch: Batch) -> None:
        """Perform learning step on the reference images and priors.

        This extracts the text categories from the reference batch to use as prompts.

        Args:
            reference_batch(Batch): The reference batch containing categories.
        """
        self.category_mapping = {}
        self.text_prompts = []

        for sample in reference_batch.samples:
            for category_id, category in zip(sample.category_ids, sample.categories, strict=False):
                if category not in self.category_mapping:
                    self.category_mapping[category] = int(category_id)
                    self.text_prompts.append(category)

        # Deduplicate text prompts while preserving order
        seen = set()
        self.text_prompts = [x for x in self.text_prompts if not (x in seen or seen.add(x))]

    def infer(self, target_batch: Batch) -> list[dict[str, torch.Tensor]]:
        """Perform inference step on the target images.

        Args:
            target_batch(Batch): The target batch.

        Returns:
            predictions(list[dict[str, torch.Tensor]]): A list of predictions.
            Each prediction contains:
                "pred_masks": torch.Tensor of shape [num_masks, H, W]
                "pred_points": torch.Tensor of shape [num_points, 4] with last dimension [x, y, score, fg_label]
                "pred_boxes": torch.Tensor of shape [num_boxes, 5] with last dimension [x1, y1, x2, y2, score]
                "pred_labels": torch.Tensor of shape [num_masks]
        """
        results = []

        with self.autocast_ctx:
            for sample in target_batch.samples:
                # Convert image to PIL if needed
                image = self._prepare_image(sample.image)

                # Set image
                inference_state = self.processor.set_image(image)

                # Use text prompts if available
                if self.text_prompts:
                    # Combine all text prompts into a single query
                    combined_prompt = ", ".join(self.text_prompts)
                    inference_state = self.processor.set_text_prompt(
                        state=inference_state,
                        prompt=combined_prompt,
                    )
                else:
                    # If no text prompts, use visual prompts (bounding boxes) if available
                    if sample.bboxes is not None and len(sample.bboxes) > 0:
                        inference_state = self._add_box_prompts(
                            sample.bboxes,
                            inference_state,
                            image.size if isinstance(image, Image.Image) else (sample.image.shape[-1], sample.image.shape[-2]),
                        )

                # Extract results
                pred_result = self._extract_predictions(inference_state, sample)
                results.append(pred_result)

                # Reset prompts for next image
                self.processor.reset_all_prompts(inference_state)

        return results

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
            return Image.fromarray(image, mode='L')
        elif image.shape[-1] == 1:
            return Image.fromarray(image.squeeze(-1), mode='L')
        elif image.shape[-1] == 3:
            return Image.fromarray(image, mode='RGB')
        else:
            # Handle 4-channel images
            return Image.fromarray(image[..., :3], mode='RGB')

    def _add_box_prompts(
        self,
        bboxes: torch.Tensor | np.ndarray,
        inference_state: dict,
        image_size: tuple[int, int],
    ) -> dict:
        """Add bounding box prompts to the inference state.

        Args:
            bboxes: Bounding boxes in [x, y, w, h] format.
            inference_state: Current inference state.
            image_size: (width, height) of the image.

        Returns:
            Updated inference state.
        """
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes)

        width, height = image_size

        for bbox in bboxes:
            # Convert from [x, y, w, h] to [cx, cy, w, h] normalized format
            x, y, w, h = bbox.tolist()
            cx = (x + w / 2) / width
            cy = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height

            norm_box_cxcywh = [cx, cy, norm_w, norm_h]

            # Add as positive box prompt
            inference_state = self.processor.add_geometric_prompt(
                state=inference_state,
                box=norm_box_cxcywh,
                label=True,
            )

        return inference_state

    def _extract_predictions(
        self,
        inference_state: dict,
        sample,
    ) -> dict[str, torch.Tensor]:
        """Extract predictions from inference state.

        Args:
            inference_state: Inference state containing predictions.
            sample: Original sample for metadata.

        Returns:
            Dictionary containing prediction tensors.
        """
        # Get predictions from state
        masks = inference_state.get("masks", torch.empty(0, 0, 0))
        boxes = inference_state.get("boxes", torch.empty(0, 5))
        scores = inference_state.get("scores", torch.empty(0))

        # Remove batch dimension from masks if present
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # [N, 1, H, W] -> [N, H, W]

        # Convert masks to boolean
        if masks.numel() > 0:
            masks = masks.squeeze(1) if masks.ndim == 4 else masks
            masks = masks > 0.5

        # Add scores to boxes to match expected format [x1, y1, x2, y2, score]
        if boxes.numel() > 0 and scores.numel() > 0:
            boxes = torch.cat([boxes, scores.unsqueeze(-1)], dim=-1)

        # Create labels based on category mapping
        num_predictions = len(masks) if masks.numel() > 0 else 0
        if num_predictions > 0 and self.category_mapping:
            # Assign the first category ID to all predictions
            first_category_id = list(self.category_mapping.values())[0]
            pred_labels = torch.full(
                (num_predictions,),
                first_category_id,
                dtype=torch.long,
                device=masks.device if masks.numel() > 0 else self.device,
            )
        else:
            pred_labels = torch.empty(0, dtype=torch.long)

        return {
            "pred_masks": masks,
            "pred_boxes": boxes,
            "pred_labels": pred_labels,
            "pred_points": torch.empty(0, 4),  # SAM3 doesn't output points directly
        }

