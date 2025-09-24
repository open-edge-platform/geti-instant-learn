# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM decoder."""

from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import copy


from segment_anything_hq.predictor import SamPredictor as SamHQPredictor
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
        ) in zip_longest(
            preprocessed_images,
            preprocessed_points,
            similarities,
            original_sizes,
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
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Preprocess the points.

        Pair a positive point with all negative points.

        This is to make the SAM model learn to generate a mask for the positive point and a mask for the negative points.

        Args:
            points: The points to preprocess.
            labels: The labels to preprocess.
            scores: The scores to preprocess.
        Returns:
            The preprocessed points (with scores in last dimension) and labels
        """
        # Separate Positive and Negative Points ---        
        positive_mask = (labels == 1).squeeze()
        negative_mask = (labels == 0).squeeze()

        # Get the corresponding coordinates and scores
        positive_coords = points[positive_mask]
        negative_coords = points[negative_mask]
        positive_scores = scores[positive_mask]
        negative_scores = scores[negative_mask]

        # Get the counts
        num_positive = positive_coords.shape[0]
        num_negative = negative_coords.shape[0]

        # Combine each positive point with all negative points 
        # negative_coords shape: [num_negative, 1, 2] -> squeeze -> [num_negative, 2] -> expand -> [num_positive, num_negative, 2]
        expanded_negative_coords = negative_coords.squeeze(1).expand(num_positive, -1, -1)
        # negative_scores shape: [num_negative, 1] -> squeeze -> [num_negative] -> unsqueeze -> [num_negative, 1] -> expand -> [num_positive, num_negative, 1]
        expanded_negative_scores = negative_scores.squeeze(1).unsqueeze(-1).expand(num_positive, -1, -1)

        # Concatenate the positive coordinates with the expanded negative coordinates
        final_point_coords_2d = torch.cat([positive_coords, expanded_negative_coords], dim=1)
        
        # Expand positive scores to match the dimension of expanded_negative_scores
        # positive_scores shape: [num_positive, 1] -> unsqueeze -> [num_positive, 1, 1]
        expanded_positive_scores = positive_scores.unsqueeze(1)
        
        # Concatenate the positive scores with the expanded negative scores
        final_point_scores = torch.cat([expanded_positive_scores, expanded_negative_scores], dim=1)        
        final_point_coords = torch.cat([final_point_coords_2d, final_point_scores], dim=-1)

        positive_label = torch.tensor([1], device=points.device, dtype=torch.float32)
        negative_labels = torch.zeros(num_negative, device=points.device, dtype=torch.float32)
        single_group_labels = torch.cat([positive_label, negative_labels])
        final_point_labels = single_group_labels.expand(num_positive, -1)
        
        return final_point_coords, final_point_labels

    def remap_preprocessed_points(self, preprocessed_points: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        """
        Remap preprocessed points from grouped format to flat format.
        
        Args:
            preprocessed_points: Tensor of shape [num_positive_points, 1_positive + N_negative, 3]
                                where last dimension is [x, y, score]
            point_labels: Tensor of shape [num_positive_points, 1_positive + N_negative]
                         where values are 1 for positive, 0 for negative
                         
        Returns:
            Tensor of shape [num_positive_points + num_negative_points, 4]
            where last dimension is [x, y, score, label]
        """
        # Get dimensions
        num_positive_groups, points_per_group, coord_dims = preprocessed_points.shape

        # get negative points
        num_negative_points = (point_labels == 0).sum() // num_positive_groups
        negative_points = preprocessed_points[0, -num_negative_points:, :]
        
        flattened_points = preprocessed_points.reshape(-1, coord_dims)
        flattened_labels = point_labels.reshape(-1)
        positive_points = flattened_points[flattened_labels == 1]
        
        postive_labels = torch.ones(positive_points.shape[0], device=preprocessed_points.device, dtype=torch.float32)
        negative_labels = torch.zeros(negative_points.shape[0], device=preprocessed_points.device, dtype=torch.float32)
        labels = torch.cat([postive_labels, negative_labels], dim=0).unsqueeze(-1)

        # Combine coordinates, scores, and labels
        # Shape: [num_positive_groups * points_per_group, 4]
        remapped_points = torch.cat([
            positive_points,  # [x, y, score]
            negative_points,  # [x, y, score]
        ], dim=0)

        remapped_points = torch.cat([remapped_points, labels], dim=-1)
        
        return remapped_points

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
    ) -> tuple[Masks, Points]:
        all_masks = Masks()
        all_used_points = Points()

        label_ids = sorted(list(similarities.data.keys()))
        similarity_maps = torch.cat([similarities.data[label_id] for label_id in label_ids])
        class_points_list = [class_points[label_id] for label_id in label_ids]

        for label_id, points_per_map, similarity_map in zip(label_ids, class_points_list, similarity_maps):
            if (points_per_map[:, 3] == 1).any():
                point_coords = points_per_map[:, :2].unsqueeze(1)
                point_scores = points_per_map[:, 2].unsqueeze(1)
                point_labels = points_per_map[:, 3].unsqueeze(1)
                point_coords, point_labels = self.point_preprocess(point_coords, point_labels, point_scores)
                masks, mask_weights, low_res_logits = self._predict(
                    point_coords=point_coords[:, :, :2],  # Extract only x, y coordinates for SAM predictor
                    point_labels=point_labels,
                    multimask_output=True,
                )
                
                final_masks, final_points, final_labels = self.mask_refinement(
                    masks= masks,
                    mask_weights=mask_weights,
                    low_res_logits=low_res_logits,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    similarity_map=similarity_map,
                    original_size=original_size,
                    score_threshold=self.mask_similarity_threshold,
                )

                if len(final_masks) == 0:
                    continue

                for final_mask in final_masks:
                    all_masks.add(final_mask, label_id)

                # Apply inverse coordinate transformation only to x, y coordinates
                final_points[:, :2] = apply_inverse_coords(
                    final_points[:, :2],  # Just the x, y coordinates
                    original_size, 
                    self.predictor.transform.target_length,
                )
                
                # Remap from [total_points, 3] to [total_points, 4] where last dim is [x, y, score, label]
                remapped_points = self.remap_preprocessed_points(final_points, final_labels)

                all_used_points.add(remapped_points, label_id)
                
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
        similarity_map: torch.Tensor,
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
            similarity_map: The similarity map to postprocess.
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
                point_labels.new_zeros(0),
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
            point_coords=point_coords[:, :, :2],  # Extract only x, y coordinates for SAM predictor
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

        masks = masks.squeeze(1)
        mask_sum = (similarity_map * masks).sum(dim=(1, 2))
        mask_area = masks.sum(dim=(1, 2))
        mask_scores = (mask_sum / (mask_area + 1e-6))
        weighted_scores = (mask_scores * mask_weights.T).squeeze(0)
        keep = weighted_scores > score_threshold
        if not keep.any():
            return (
                masks.new_zeros((0, *masks.shape[-2:])),
                point_coords.new_zeros(0),
                point_labels.new_zeros(0),
            )

        return masks[keep], point_coords[keep], point_labels[keep]
        