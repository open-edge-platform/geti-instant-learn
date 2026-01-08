# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch

from getiprompt.models.foundation.sam3.utils.sam1_utils import SAM2Transforms


class SAM3InteractiveImagePredictor:
    """SAM1-style interactive image predictor for point/box prompts."""

    def __init__(
        self,
        sam_model,
        mask_threshold=0.0,
        max_hole_area=0.0,
        max_sprinkle_area=0.0,
    ):
        """
        Args:
            sam_model: SAM3 model instance
            mask_threshold: Threshold for mask binarization
            max_hole_area: Maximum area for hole filling
            max_sprinkle_area: Maximum area for sprinkle removal
        """
        self.model = sam_model
        self.mask_threshold = mask_threshold
        self._transforms = SAM2Transforms(
            resolution=sam_model.image_size,
            mask_threshold=mask_threshold,
            max_hole_area=max_hole_area,
            max_sprinkle_area=max_sprinkle_area,
        )
        self._transforms.set_device(sam_model.device)

        # Predictor state
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False

    @property
    def device(self):
        return self.model.device

    def _prep_prompts(
        self, point_coords, point_labels, box=None, mask_input=None, normalize_coords=True, img_idx=-1
    ):
        """Prepare prompts for prediction.

        Args:
            point_coords: Point coordinates [N, 2]
            point_labels: Point labels [N]
            box: Bounding box [4] or [1, 4]
            mask_input: Mask input [1, H, W] or [1, 1, H, W]
            normalize_coords: Whether to normalize coordinates
            img_idx: Index for batch mode (-1 uses last element)

        Returns:
            Tuple of (mask_input, unnorm_coords, labels, unnorm_box)
        """
        unnorm_coords, labels, unnorm_box = None, None, None

        # Get orig_hw for this image (supports both list and tuple formats)
        if isinstance(self._orig_hw, list):
            orig_hw = self._orig_hw[img_idx]
        else:
            orig_hw = self._orig_hw

        if point_coords is not None:
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float32, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=orig_hw
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int32, device=self.device)
            if unnorm_coords.ndim == 2:
                unnorm_coords = unnorm_coords.unsqueeze(0)  # add batch dimension
            if labels.ndim == 1:
                labels = labels.unsqueeze(0)

        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float32, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=orig_hw
            )

        if mask_input is not None:
            mask_input = torch.as_tensor(
                mask_input, dtype=torch.float32, device=self.device
            )
            if mask_input.ndim == 2:
                mask_input = mask_input.unsqueeze(0).unsqueeze(0)
            elif mask_input.ndim == 3:
                mask_input = mask_input.unsqueeze(0)

        return mask_input, unnorm_coords, labels, unnorm_box

    @torch.no_grad()
    def set_image(self, image):
        """Set image for prediction and compute embeddings.

        Args:
            image: Input image as numpy array [H, W, 3]
        """
        self.reset_predictor()
        self._orig_hw = image.shape[:2]

        # Prepare input
        if isinstance(image, np.ndarray):
            self._is_batch = False
            img_batch = self._transforms(image).unsqueeze(0)
            # Ensure it's on the correct device
            img_batch = img_batch.to(self.device)
        else:
            raise NotImplementedError("Image type not supported")

        # Compute features
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        # vision_feats shapes: [(B, C, H, W), (B, C, H, W), (B, C, H, W)]
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # Transform from (B, C, H, W) to (B, H*W, C)
        feats = [
            feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1])
            for feat in vision_feats[::-1]
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True

    @torch.no_grad()
    def set_image_batch(self, image_list):
        """Set batch of images for prediction.

        Args:
            image_list: List of input images as numpy arrays
        """
        self.reset_predictor()
        self._is_batch = True
        img_batch = self._transforms.forward_batch(image_list).to(self.device)
        self._orig_hw = [img.shape[:2] for img in image_list]

        # Compute features
        backbone_out = self.model.forward_image(img_batch)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed

        # Transform from (B, C, H, W) to (B, H*W, C)
        feats = [
            feat.permute(0, 2, 3, 1).reshape(feat.shape[0], -1, feat.shape[1])
            for feat in vision_feats[::-1]
        ][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        self._is_image_set = True

    @torch.no_grad()
    def _predict(
        self,
        point_coords,
        point_labels,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
        img_idx=-1,
    ):
        """Internal prediction with tensors.

        Args:
            point_coords: Point coordinates
            point_labels: Point labels
            box: Bounding box
            mask_input: Mask input
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return logits instead of masks
            normalize_coords: Whether to normalize coordinates
            img_idx: Index for batch mode (-1 uses last element)

        Returns:
            Tuple of (masks, iou_predictions, low_res_masks)
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        # Prepare prompts
        mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
            point_coords, point_labels, box, mask_input, normalize_coords, img_idx
        )

        # Merge boxes into points (following efficientsam3 approach)
        # This converts box coordinates to corner points with special labels
        concat_points = None
        if unnorm_coords is not None:
            concat_points = (unnorm_coords, labels)

        if unnorm_box is not None:
            # Convert boxes [B, 4] to box corners [B, 2, 2]
            box_coords = unnorm_box.reshape(-1, 2, 2)
            # Box corners use labels 2 (top-left) and 3 (bottom-right)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=unnorm_box.device)
            box_labels = box_labels.repeat(unnorm_box.size(0), 1)

            if concat_points is not None:
                # Merge boxes with existing points
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        # Run prompt encoder (boxes are now encoded as points)
        if hasattr(self.model, "sam_prompt_encoder"):
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=concat_points,
                boxes=None,  # Boxes are already converted to points
                masks=mask_input,
            )
        else:
            raise RuntimeError("Model does not have sam_prompt_encoder")

        # Get image embeddings for this batch index
        if img_idx >= 0 and self._is_batch:
            image_embed = self._features["image_embed"][img_idx : img_idx + 1]
            high_res_feats = [f[img_idx : img_idx + 1] for f in self._features["high_res_feats"]]
        else:
            image_embed = self._features["image_embed"]
            high_res_feats = self._features["high_res_feats"]

        # Run mask decoder
        # batched_mode is true when we have multiple objects (multi-object prediction)
        batched_mode = concat_points is not None and concat_points[0].shape[0] > 1
        if hasattr(self.model, "sam_mask_decoder"):
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=image_embed,
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=multimask_output,
                repeat_image=batched_mode,
                high_res_features=high_res_feats,
            )
        else:
            raise RuntimeError("Model does not have sam_mask_decoder")

        # Upscale masks
        # Get orig_hw for this image
        if isinstance(self._orig_hw, list):
            orig_hw = self._orig_hw[img_idx]
        else:
            orig_hw = self._orig_hw

        # Postprocess and threshold masks
        masks = self._transforms.postprocess_masks(low_res_masks, orig_hw=orig_hw)
        low_res_masks = torch.clamp(low_res_masks, -32.0, 32.0)
        if not return_logits:
            # Apply threshold to convert logits to binary mask
            masks = masks > self.mask_threshold

        return masks, iou_predictions, low_res_masks

    @torch.no_grad()
    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    ):
        """Predict masks for single image.

        Args:
            point_coords: Point coordinates [N, 2] as numpy array
            point_labels: Point labels [N] as numpy array
            box: Bounding box [4] or [1, 4] as numpy array
            mask_input: Mask input [H, W] or [1, H, W] as numpy array
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return logits instead of masks
            normalize_coords: Whether coordinates are in absolute image space

        Returns:
            Tuple of (masks_np, iou_predictions_np, low_res_masks_np)
        """
        if not self._is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")

        if self._is_batch:
            raise RuntimeError(
                "This predictor is in batch mode. Use predict_batch for batched inference."
            )

        masks, iou_predictions, low_res_masks = self._predict(
            point_coords,
            point_labels,
            box,
            mask_input,
            multimask_output,
            return_logits,
            normalize_coords,
        )

        # Convert to numpy
        masks_np = masks.squeeze(0).float().detach().cpu().numpy()
        iou_predictions_np = iou_predictions.squeeze(0).float().detach().cpu().numpy()
        low_res_masks_np = low_res_masks.squeeze(0).float().detach().cpu().numpy()

        return masks_np, iou_predictions_np, low_res_masks_np

    @torch.no_grad()
    def predict_batch(
        self,
        point_coords_batch=None,
        point_labels_batch=None,
        box_batch=None,
        mask_input_batch=None,
        multimask_output=True,
        return_logits=False,
        normalize_coords=True,
    ):
        """Predict masks for batch of images.

        Args:
            point_coords_batch: List of point coordinates
            point_labels_batch: List of point labels
            box_batch: List of bounding boxes
            mask_input_batch: List of mask inputs
            multimask_output: Whether to return multiple masks
            return_logits: Whether to return logits instead of masks
            normalize_coords: Whether coordinates are in absolute image space

        Returns:
            Lists of (masks_batch, iou_predictions_batch, low_res_masks_batch)
        """
        if not self._is_batch:
            raise RuntimeError(
                "This predictor is not in batch mode. Use predict for single image inference."
            )

        num_images = len(self._orig_hw) if isinstance(self._orig_hw, list) else 1
        if point_coords_batch is None:
            point_coords_batch = [None] * num_images
        if point_labels_batch is None:
            point_labels_batch = [None] * num_images

        masks_batch = []
        iou_predictions_batch = []
        low_res_masks_batch = []

        for i in range(num_images):
            masks, iou_predictions, low_res_masks = self._predict(
                point_coords_batch[i],
                point_labels_batch[i],
                box_batch[i] if box_batch else None,
                mask_input_batch[i] if mask_input_batch else None,
                multimask_output,
                return_logits,
                normalize_coords,
                img_idx=i,
            )

            masks_batch.append(masks.squeeze(0).float().detach().cpu().numpy())
            iou_predictions_batch.append(iou_predictions.squeeze(0).float().detach().cpu().numpy())
            low_res_masks_batch.append(low_res_masks.squeeze(0).float().detach().cpu().numpy())

        return masks_batch, iou_predictions_batch, low_res_masks_batch

    def reset_predictor(self):
        """Reset predictor state."""
        self._is_image_set = False
        self._features = None
        self._orig_hw = None
        self._is_batch = False
