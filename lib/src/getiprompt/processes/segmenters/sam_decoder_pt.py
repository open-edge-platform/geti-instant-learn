# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM decoder."""

from collections import defaultdict
from itertools import zip_longest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import copy


from segment_anything_hq.predictor import SamPredictor as SamHQPredictor
from getiprompt.processes.segmenters.segmenter_base import Segmenter
from getiprompt.types import Boxes, Image, Masks, Points, Priors, Similarities

from torchvision.ops import masks_to_boxes, batched_nms


DEBUG = True

def get_preprocess_shape(oldh: torch.Tensor, oldw: torch.Tensor, long_side_length: int) -> tuple[int, int]:
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def apply_coords(coords: torch.Tensor, original_size: tuple[int, int], long_side_length: int) -> torch.Tensor:
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(old_h, old_w, long_side_length)
    coords = copy.deepcopy(coords).to(torch.float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

def apply_inverse_coords(coords: torch.Tensor, original_size: tuple[int, ...], long_side_length: int) -> torch.Tensor:
    """
    Inverts the coordinate transformation back to the original image size.
    """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(
        original_size[0], original_size[1], long_side_length,
    )
    # coords = copy.deepcopy(coords).to(torch.float)
    coords = torch.clone(coords).to(torch.float)
    coords[..., 0] = coords[..., 0] * (old_w / new_w)
    coords[..., 1] = coords[..., 1] * (old_h / new_h)
    return coords



class PTSamDecoder(nn.Module):
    def __init__(
        self,
        sam_predictor: SamHQPredictor,
        apply_mask_refinement: bool = False,
        target_guided_attention: bool = False,
        mask_similarity_threshold: float = 0.45,
        skip_points_in_existing_masks: bool = True,
        nms_iou_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.predictor = sam_predictor
        self.apply_mask_refinement = apply_mask_refinement
        self.target_guided_attention = target_guided_attention
        self.mask_similarity_threshold = mask_similarity_threshold
        self.skip_points_in_existing_masks = skip_points_in_existing_masks
        self.nms_iou_threshold = nms_iou_threshold

    def preprocess_inputs(
        self,
        images: list[Image],
        priors: list[Priors] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict[int, torch.Tensor]], list[tuple[int, int]]]:
        preprocessed_images = []
        preprocessed_points = []
        original_sizes = []
        
        for (image, priors_per_image) in zip_longest(images, priors, fillvalue=None):
            # Preprocess image using SamPredictor transform
            input_image = self.predictor.transform.apply_image(image.data)
            input_image_torch = torch.as_tensor(input_image, device=self.predictor.device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
            ori_size = image.data.shape[:2]
            preprocessed_images.append(input_image_torch)
            original_sizes.append(ori_size)
            class_points = {}
            # Preprocess points for each class
            for class_id, points_per_class in priors_per_image.points.data.items():
                if len(points_per_class) > 1:
                    raise ValueError("Only one piror map per class is supported")
                points_per_class = points_per_class[0]
                if len(points_per_class) == 0:
                    class_points[class_id] = torch.tensor(np.array([]))
                    continue

                # Extract coordinates (x, y) from points
                coords = points_per_class[:, :2].cpu().numpy().astype(np.float32)

                # Apply coordinate transformation using SamPredictor's transform
                transformed_coords = self.predictor.transform.apply_coords(coords, ori_size)
                
                # Convert back to tensor and update the points
                transformed_coords_tensor = torch.from_numpy(transformed_coords).to(device=points_per_class.device, dtype=points_per_class.dtype)

                # Create new points tensor with transformed coordinates
                _points = points_per_class.clone()
                _points[:, :2] = transformed_coords_tensor
                class_points[class_id] = _points
            preprocessed_points.append(class_points)
        return preprocessed_images, preprocessed_points, original_sizes

    @torch.inference_mode()
    def forward(
        self,
        preprocessed_images: list[torch.Tensor],
        preprocessed_points: list[dict[int, torch.Tensor]],
        similarities: list[Similarities] | None = None,
        original_sizes: list[tuple[int, int]] | None = None,
        image_ids: list[int] | None = None,
    ) -> tuple[list[Masks], list[Points], list[Boxes]]:
        if similarities is None:
            similarities = []
        if original_sizes is None:
            original_sizes = []
        masks_per_image: list[Masks] = []
        points_per_image: list[Points] = []
        boxes_per_image: list[Boxes] = []
        
        for (
            preprocessed_image,
            preprocessed_points_per_image,
            similarities_per_image,
            original_size,
            image_id,
        ) in zip_longest(
            preprocessed_images,
            preprocessed_points,
            similarities,
            original_sizes,
            image_ids,
            fillvalue=None,
        ):
            if preprocessed_points_per_image is None:
                continue

            # Set the preprocessed image in the predictor
            self.predictor.set_torch_image(preprocessed_image, original_size)            
            if len(preprocessed_points_per_image) > 0:
                masks, points_used = self.predict_by_points(
                    preprocessed_points_per_image,
                    similarities_per_image,
                    original_size,
                    image_id=image_id,
                )
                points_per_image.append(points_used)
            else:
                masks = Masks()
                points_used = Points()
                points_per_image.append(points_used)
            masks_per_image.append(masks)
        return masks_per_image, points_per_image, boxes_per_image

    def point_preprocess(
        self, 
        points: torch.Tensor, 
        labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the points.

        Pair a positive point with all negative points.

        This is to make the SAM model learn to generate a mask for the positive point and a mask for the negative points.

        Args:
            points: The points to preprocess.
            labels: The labels to preprocess.

        Returns:
            The preprocessed points and labels.
        """
        # Separate Positive and Negative Points ---        
        positive_mask = (labels == 1).squeeze()
        negative_mask = (labels == 0).squeeze()

        # Get the corresponding coordinates
        positive_coords = points[positive_mask]
        negative_coords = points[negative_mask]

        # Get the counts
        num_positive = positive_coords.shape[0]
        num_negative = negative_coords.shape[0]

        # Combine each positive point with all negative points 
        expanded_negative_coords = negative_coords.squeeze(1).expand(num_positive, -1, -1)

        # Concatenate the positive coordinates with the expanded negative coordinates
        final_point_coords = torch.cat([positive_coords, expanded_negative_coords], dim=1)

        # Generate the Corresponding Labels
        # Create a label set for one group: one positive and N negatives
        positive_label = torch.tensor([1], device=points.device, dtype=torch.float32)
        negative_labels = torch.zeros(num_negative, device=points.device, dtype=torch.float32)
        single_group_labels = torch.cat([positive_label, negative_labels])
        final_point_labels = single_group_labels.expand(num_positive, -1)
        return final_point_coords, final_point_labels

    def plot_masks(self, masks: torch.Tensor, scores: torch.Tensor, save_path: str):
        column = 5
        row = (masks.shape[0] + column - 1) // column
        fig, axes = plt.subplots(row, column, figsize=(30, 30))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            if len(axes.shape) == 1:    
                axes = axes[None, :]
            axes[i // column, i % column].imshow(mask.detach().cpu().numpy())
            axes[i // column, i % column].set_title(f"Score: {score:.2f}")
            axes[i // column, i % column].axis('off')
        plt.savefig(save_path)
        plt.close(fig)

    def predict_by_points(
        self,
        class_points: dict[int, torch.Tensor],
        similarities: Similarities | None = None,
        original_size: tuple[int, int] | None = None,
        image_id: int = 0,
    ) -> tuple[Masks, Points]:
        all_masks = Masks()
        all_used_points = Points()

        label_ids = sorted(list(similarities.data.keys()))
        similarity_maps = torch.cat([similarities.data[label_id] for label_id in label_ids])
        class_points_list = [class_points[label_id] for label_id in label_ids]

        for label_id, points_per_map, similarity_map in zip(label_ids, class_points_list, similarity_maps):
            if (points_per_map[:, 3] == 1).any():
                point_coords = points_per_map[:, :2].unsqueeze(1)
                point_labels = points_per_map[:, 3].unsqueeze(1)
                point_coords, point_labels = self.point_preprocess(point_coords, point_labels)
                masks, mask_weights, low_res_logits = self._predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )
                
                final_masks, final_used_points, final_mask_scores = self.mask_refinement(
                    masks,
                    mask_weights,
                    low_res_logits,
                    point_coords,
                    point_labels,
                    similarity_map,
                    original_size,
                    score_threshold=self.mask_similarity_threshold,
                )

                if len(final_masks) == 0:
                    continue

                for final_mask in final_masks:
                    all_masks.add(final_mask, label_id)

                final_used_points = apply_inverse_coords(
                    final_used_points, 
                    original_size, 
                    self.predictor.transform.target_length,
                )
                all_used_points.add(final_used_points, label_id)
                
        return all_masks, all_used_points

    def _predict(
        self,
        point_coords: torch.Tensor | None = None,
        point_labels: torch.Tensor | None = None,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        masks, mask_scores, low_res_logits = self.predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=multimask_output,
        )
        return masks.bool(), mask_scores, low_res_logits

    def mask_refinement(
        self,
        masks: torch.Tensor,
        mask_weights: torch.Tensor,
        low_res_logits: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        similarities: torch.Tensor,
        original_size: tuple[int, int],
        score_threshold: float = 0.45,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine the masks.
        
        Args:
            masks: The masks to refine.
            low_res_logits: The low-res logits to refine.
            point_coords: The point coordinates to refine.
            point_labels: The point labels to refine.
            mask_weights: The mask weights to postprocess.
            similarities: The similarities to postprocess.
            original_size: The original size of the image.
            score_threshold: The score threshold to postprocess.

        Returns:
            The postprocessed masks, point coordinates, and mask scores.
        """

        keep = masks.squeeze(1).sum(dim=(-1, -2)) > 0
        if not keep.any():
            return (
                masks.new_zeros((0, *masks.shape[-2:])),
                point_coords.new_zeros(0),
                mask_scores.new_zeros(0),
            )

        masks = masks[keep]
        mask_weights = mask_weights[keep]
        low_res_logits = low_res_logits[keep]
        point_coords = point_coords[keep]
        point_labels = point_labels[keep]

        # refine masks with boxes
        boxes = masks_to_boxes(masks.squeeze(1))
        boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size, self.predictor.model.image_encoder.img_size)
        boxes = boxes.reshape(-1, 4)
        boxes = boxes.unsqueeze(1)

        masks, mask_weights, _ = self._predict(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=low_res_logits,
            multimask_output=True,
        )

        # nms the masks
        nms_indices = batched_nms(
            boxes.squeeze(1),
            mask_weights.squeeze(1),
            torch.zeros(len(boxes)),  # categories
            iou_threshold=self.nms_iou_threshold,
        )

        masks = masks[nms_indices]
        mask_weights = mask_weights[nms_indices]
        low_res_logits = low_res_logits[nms_indices]
        boxes = boxes[nms_indices]
        point_coords = point_coords[nms_indices]
        point_labels = point_labels[nms_indices]

        if masks.shape[-2:] != similarities.shape[0:]:
            masks = F.interpolate(
                masks,
                size=(similarities.shape[0], similarities.shape[1]),
                mode="nearest",
            )

        masks = masks.squeeze(1)
        mask_sum = (similarities * masks).sum(dim=(1, 2))
        mask_area = masks.sum(dim=(1, 2))
        mask_scores = (mask_sum / (mask_area + 1e-6))
        weighted_scores = (mask_scores * mask_weights.T).squeeze(0)
        keep = weighted_scores > score_threshold
        if not keep.any():
            return (
                masks.new_zeros((0, *masks.shape[-2:])),
                point_coords.new_zeros(0),
                mask_scores.new_zeros(0),
            )
        
        return masks[keep], point_coords[keep], weighted_scores[keep]
        