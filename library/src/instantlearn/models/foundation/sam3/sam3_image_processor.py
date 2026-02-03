# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import numpy as np
import PIL
import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from instantlearn.models.foundation.sam3.data_misc import FindStage, interpolate
from instantlearn.models.foundation.sam3.model.box_ops import box_cxcywh_to_xyxy
from instantlearn.models.foundation.sam3.sam3_image import Sam3Image


class Sam3Processor:
    """ """

    def __init__(
        self,
        model: Sam3Image,
        resolution: int = 1008,
        device: str = "cuda",
        confidence_threshold: float = 0.5,
    ):
        self.model = model
        self.resolution = resolution
        self.device = device
        self.transform = v2.Compose(
            [
                v2.ToDtype(torch.uint8, scale=True),
                v2.Resize(size=(resolution, resolution)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ],
        )
        self.confidence_threshold = confidence_threshold

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], device=device, dtype=torch.long),
            text_ids=torch.tensor([0], device=device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    @torch.inference_mode()
    def set_image(self, image, state: dict | None = None) -> dict:
        """Sets the image on which we want to do predictions."""
        if state is None:
            state = {}

        if isinstance(image, PIL.Image.Image):
            width, height = image.size
        elif isinstance(image, (torch.Tensor, np.ndarray)):
            height, width = image.shape[-2:]
        else:
            raise ValueError("Image must be a PIL image or a tensor")

        image = v2.functional.to_image(image).to(self.device)
        image = self.transform(image).unsqueeze(0)

        state["original_height"] = height
        state["original_width"] = width
        state["backbone_out"] = self.model.backbone.forward_image(image)
        return state

    @torch.inference_mode()
    def set_image_batch(
        self,
        images: list[PIL.Image.Image | torch.Tensor | np.ndarray],
        state: dict | None = None,
    ) -> dict:
        """Sets the image batch on which we want to do predictions.

        Args:
            images: List of images as PIL Images, torch Tensors, or numpy arrays.
                Tensors/arrays should be in (C, H, W) or (H, W, C) format.
            state: Optional state dict to update. If None, a new one is created.

        Returns:
            Updated state dict with backbone outputs.
        """
        if state is None:
            state = {}

        if not isinstance(images, list) or len(images) == 0:
            raise ValueError("Images must be a non-empty list")

        # Extract original dimensions based on image type
        original_heights = []
        original_widths = []
        for image in images:
            if isinstance(image, PIL.Image.Image):
                original_widths.append(image.width)
                original_heights.append(image.height)
            elif isinstance(image, (torch.Tensor, np.ndarray, tv_tensors.Image)):
                # Assume (C, H, W) or (H, W, C) format - use last two dims
                original_heights.append(image.shape[-2])
                original_widths.append(image.shape[-1])
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        state["original_heights"] = original_heights
        state["original_widths"] = original_widths

        images = [self.transform(v2.functional.to_image(image).to(self.device)) for image in images]
        images = torch.stack(images, dim=0)
        state["backbone_out"] = self.model.backbone.forward_image(images)
        return state

    def get_single_image_state(self, batch_state: dict, image_idx: int) -> dict:
        """Extract single-image state from a batch state.

        After calling set_image_batch, use this method to get a state dict
        for a specific image that can be used with set_prompt and other methods.

        Args:
            batch_state: State dict returned by set_image_batch.
            image_idx: Index of the image in the batch (0-indexed).

        Returns:
            A new state dict for the specified image.
        """
        if "original_heights" not in batch_state:
            raise ValueError("batch_state must be from set_image_batch (missing original_heights)")

        single_state = {
            "original_height": batch_state["original_heights"][image_idx],
            "original_width": batch_state["original_widths"][image_idx],
        }

        # Slice backbone outputs for this specific image
        backbone_out = batch_state["backbone_out"]
        single_backbone_out = {}

        # Handle vision_features: [B, ...] -> [1, ...]
        if "vision_features" in backbone_out:
            single_backbone_out["vision_features"] = backbone_out["vision_features"][image_idx : image_idx + 1]

        # Handle vision_pos_enc: list of tensors, each [B, ...] -> [1, ...]
        if "vision_pos_enc" in backbone_out:
            single_backbone_out["vision_pos_enc"] = [
                pos[image_idx : image_idx + 1] for pos in backbone_out["vision_pos_enc"]
            ]

        # Handle backbone_fpn: list of tensors, each [B, ...] -> [1, ...]
        if "backbone_fpn" in backbone_out:
            single_backbone_out["backbone_fpn"] = [
                feat[image_idx : image_idx + 1] for feat in backbone_out["backbone_fpn"]
            ]

        # Handle sam2_backbone_out if present
        if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
            sam2_out = backbone_out["sam2_backbone_out"]
            single_sam2_out = {}
            if "vision_features" in sam2_out:
                single_sam2_out["vision_features"] = sam2_out["vision_features"][image_idx : image_idx + 1]
            if "vision_pos_enc" in sam2_out:
                single_sam2_out["vision_pos_enc"] = [
                    pos[image_idx : image_idx + 1] for pos in sam2_out["vision_pos_enc"]
                ]
            if "backbone_fpn" in sam2_out:
                single_sam2_out["backbone_fpn"] = [feat[image_idx : image_idx + 1] for feat in sam2_out["backbone_fpn"]]
            single_backbone_out["sam2_backbone_out"] = single_sam2_out
        else:
            single_backbone_out["sam2_backbone_out"] = None

        single_state["backbone_out"] = single_backbone_out
        return single_state

    @torch.inference_mode()
    def set_text_prompt(self, prompt: str, state: dict) -> dict:
        """Sets the text prompt and run the inference"""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        text_outputs = self.model.backbone.forward_text([prompt], device=self.device)
        # will erase the previous text prompt if any
        state["backbone_out"].update(text_outputs)
        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        return self._forward_grounding(state)

    @torch.inference_mode()
    def add_geometric_prompt(self, box: list, label: bool, state: dict):
        """Adds a box prompt and run the inference.
        The image needs to be set, but not necessarily the text prompt.
        The box is assumed to be in [center_x, center_y, width, height] format and normalized in [0, 1] range.
        The label is True for a positive box, False for a negative box.
        """
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_text_prompt")

        if "language_features" not in state["backbone_out"]:
            # Looks like we don't have a text prompt yet. This is allowed, but we need to set the text prompt to "visual" for the model to rely only on the geometric prompt
            dummy_text_outputs = self.model.backbone.forward_text(
                ["visual"],
                device=self.device,
            )
            state["backbone_out"].update(dummy_text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        # adding a batch and sequence dimension
        boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([label], device=self.device, dtype=torch.bool).view(1, 1)
        state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    @torch.inference_mode()
    def set_prompt(
        self,
        state: dict,
        text: str | None = None,
        box: list | None = None,
        label: bool | None = None,
    ) -> dict:
        """Sets the prompt (text and/or box) and run the inference"""
        if "backbone_out" not in state:
            raise ValueError("You must call set_image before set_prompt")

        if text is not None:
            text_outputs = self.model.backbone.forward_text([text], device=self.device)
            # will erase the previous text prompt if any
        else:
            text_outputs = self.model.backbone.forward_text(["visual"], device=self.device)
        state["backbone_out"].update(text_outputs)

        if "geometric_prompt" not in state:
            state["geometric_prompt"] = self.model._get_dummy_prompt()

        if box is not None:
            # adding a batch and sequence dimension
            boxes = torch.tensor(box, device=self.device, dtype=torch.float32).view(1, 1, 4)
            labels = torch.tensor([1] * len(boxes), device=self.device, dtype=torch.bool).view(1, 1)
            state["geometric_prompt"].append_boxes(boxes, labels)

        return self._forward_grounding(state)

    def reset_all_prompts(self, state: dict) -> None:
        """Removes all the prompts and results"""
        if "backbone_out" in state:
            backbone_keys_to_del = [
                "language_features",
                "language_mask",
                "language_embeds",
            ]
            for key in backbone_keys_to_del:
                if key in state["backbone_out"]:
                    del state["backbone_out"][key]

        keys_to_del = ["geometric_prompt", "boxes", "masks", "masks_logits", "scores"]
        for key in keys_to_del:
            state.pop(key, None)

    @torch.inference_mode()
    def set_confidence_threshold(self, threshold: float, state: dict | None = None) -> dict:
        """Sets the confidence threshold for the masks"""
        self.confidence_threshold = threshold
        if state is not None and "boxes" in state:
            # we need to filter the boxes again
            # In principle we could do this more efficiently since we would only need
            # to rerun the heads. But this is simpler and not too inefficient
            return self._forward_grounding(state)
        return state

    @torch.inference_mode()
    def _forward_grounding(self, state: dict) -> dict:
        outputs = self.model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=self.find_stage,
            geometric_prompt=state["geometric_prompt"],
            find_target=None,
        )

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > self.confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep]
        out_bbox = out_bbox[keep]

        # convert to [x0, y0, x1, y1] format
        boxes = box_cxcywh_to_xyxy(out_bbox)

        img_h = state["original_height"]
        img_w = state["original_width"]
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h]).to(self.device)
        boxes = boxes * scale_fct[None, :]

        out_masks = interpolate(
            out_masks.unsqueeze(1),
            (img_h, img_w),
            mode="bilinear",
            align_corners=False,
        ).sigmoid()

        state["masks_logits"] = out_masks
        state["masks"] = out_masks > 0.5
        state["boxes"] = boxes
        state["scores"] = out_probs
        return state
