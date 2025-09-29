# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX-exportable SAM decoder."""

import torch
import torch.nn.functional as F
from torch import nn
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor
from torchvision.ops import nms
import numpy as np
import copy
import onnx
import openvino


def get_preprocess_shape(oldh: torch.Tensor, oldw: torch.Tensor, long_side_length: int) -> tuple[int, int]:
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def apply_coords(coords: torch.Tensor, original_size: tuple[int, int] | torch.Tensor, long_side_length: int) -> torch.Tensor:
    # Handle both tuple and tensor inputs for ONNX compatibility
    old_h, old_w = original_size
    
    # Convert to tensors for ONNX traceability
    old_h = torch.tensor(old_h, dtype=torch.float32)
    old_w = torch.tensor(old_w, dtype=torch.float32)
    long_side_length = torch.tensor(long_side_length, dtype=torch.float32)
    
    # Calculate scale and new dimensions
    scale = long_side_length / torch.max(old_h, old_w)
    new_h = (old_h * scale + 0.5).int().float()
    new_w = (old_w * scale + 0.5).int().float()
    
    coords = torch.clone(coords).to(torch.float)
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


class PTSamDecoderONNX(nn.Module):
    """ONNX-exportable version of PTSamDecoder.
    
    This version simplifies the interface to make it ONNX-compatible:
    - Fixed batch size and input shapes
    - No control flow operations
    - Tensor-only inputs/outputs
    - Single class prediction per forward pass
    """

    def __init__(
        self,
        sam_predictor: SamHQPredictor,
        mask_similarity_threshold: float = 0.38,
        nms_iou_threshold: float = 0.1,
        max_points: int = 40,
    ) -> None:
        super().__init__()
        self.predictor = sam_predictor
        self.mask_similarity_threshold = mask_similarity_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_points = max_points
        
        # Store model components for direct access
        self.sam = sam_predictor.model
        self.image_encoder = sam_predictor.model.image_encoder
        self.prompt_encoder = sam_predictor.model.prompt_encoder
        self.mask_decoder = sam_predictor.model.mask_decoder

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
        positive_mask = (labels == 1).squeeze(1)
        negative_mask = (labels == 0).squeeze(1)

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

    def _predict(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        boxes: torch.Tensor | None = None,
        mask_input: torch.Tensor | None = None,
        multimask_output: bool = False,
        original_size: tuple[int, int] | torch.Tensor = (480, 640),
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        points = (point_coords, point_labels)
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )
        # Predict masks
        low_res_masks, iou_predictions = self.sam.mask_decoder(
            image_embeddings=self._stored_image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            hq_token_only=False,
            interm_embeddings=self._stored_interm_features,
        )

        masks = self.sam.postprocess_masks(low_res_masks, self.input_size, original_size)
        masks = masks > self.sam.mask_threshold
        return masks, iou_predictions, low_res_masks

    def remap_preprocessed_points(self, preprocessed_points: torch.Tensor) -> torch.Tensor:
        """
        Remap preprocessed points from grouped format to flat format.
        
        Args:
            preprocessed_points: Tensor of shape [num_positive_points, 1_positive + N_negative, 3]
                                where last dimension is [x, y, score]

        Returns:
            Tensor of shape [num_positive_points + num_negative_points, 4]
            where last dimension is [x, y, score, label]
        """
        num_pos, total_num, last_dim = preprocessed_points.shape
        num_neg = total_num - 1
        remapped_points = preprocessed_points.new_zeros(
            num_pos + num_neg,
            last_dim + 1  # last dimension is [x, y, score, label]
        )
        positive_points = preprocessed_points[:, 0]
        positive_labels = torch.ones(num_pos, device=preprocessed_points.device, dtype=torch.float32).unsqueeze(-1)
        positive_points = torch.cat([positive_points, positive_labels], dim=-1)

        negative_points = preprocessed_points[0, 1:]
        negative_labels = torch.zeros(num_neg, device=preprocessed_points.device, dtype=torch.float32).unsqueeze(-1)
        negative_points = torch.cat([negative_points, negative_labels], dim=-1)

        remapped_points[:num_pos, :] = positive_points
        remapped_points[num_pos:, :] = negative_points
        return remapped_points

    @torch.no_grad()
    def forward(
        self,
        image: torch.Tensor,  # [1, 3, 1024, 1024]
        class_point_coords: torch.Tensor,  # [num_classes, N, 4] - [x, y, score, label]
        similarity_maps: torch.Tensor,      # [num_classes, H, W] - similarity map for scoring
        original_size: torch.Tensor,       # [2] - original image size [H, W]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:        
        # Encode image
        # TODO: Set N to 40, and if there are less than 40 points, fill class_point_coords with zeros. 

        self.input_size = tuple(image.shape[-2:])
        image = self.sam.preprocess(image)
        image_embeddings, interm_features = self.image_encoder(image)

        h, w = original_size[0], original_size[1]
        original_size = (h, w)
        
        # Store embeddings for second stage refinement
        self._stored_image_embeddings = image_embeddings
        self._stored_interm_features = interm_features
        
        _masks = []
        _labels = []
        _coords = []
        for class_id, (points_per_map, similarity_map) in enumerate(zip(class_point_coords, similarity_maps)):
            processed_point_coords = points_per_map[:, :2].unsqueeze(1)
            point_scores = points_per_map[:, 2].unsqueeze(1)
            processed_point_labels = points_per_map[:, 3].unsqueeze(1)
            processed_point_coords, processed_point_labels = self.point_preprocess(processed_point_coords, processed_point_labels, point_scores)
            masks, _, low_res_logits = self._predict(
                processed_point_coords[:, :, :2],
                processed_point_labels,
                boxes=None,
                mask_input=None,
                multimask_output=True,
                original_size=original_size,
            )
        
            final_masks, final_points, final_labels = self.mask_refinement(
                masks=masks,
                low_res_logits=low_res_logits,
                point_coords=processed_point_coords,
                point_labels=processed_point_labels,
                similarity_map=similarity_map,
                original_size=original_size,
                score_threshold=self.mask_similarity_threshold,
                nms_iou_threshold=self.nms_iou_threshold,
            )

            class_labels = torch.full((final_masks.size(0),), class_id, device=final_masks.device)
            pos_points = final_points[:, 0]

            _masks.append(final_masks)
            _labels.append(class_labels)
            _coords.append(pos_points) 
                        
        # Concatenate once at the end
        _masks = torch.cat(_masks, dim=0)    # (num_masks, 128, 128)
        _labels = torch.cat(_labels, dim=0)  # (num_masks)
        _coords = torch.cat(_coords, dim=0)  # (num_masks, 4)

        return _masks, _labels, _coords


    def mask_refinement(
        self,
        masks: torch.Tensor,
        low_res_logits: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        similarity_map: torch.Tensor,
        original_size: tuple[int, int],
        score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Check which masks have non-zero area
        batch_size, _, height, width = masks.shape
        masks = masks.view(batch_size, height, width)        
        # Generate boxes, handling empty masks
        boxes = self._safe_masks_to_boxes(masks)
        
        # Apply coordinate transformation
        boxes_reshaped = boxes.reshape(-1, 2, 2)
        # Use explicit img_size to avoid ONNX tracing issues
        img_size = self.sam.image_encoder.img_size
        boxes_transformed = apply_coords(
            boxes_reshaped, 
            original_size, 
            img_size
        )
        boxes = boxes_transformed.reshape(-1, 4).unsqueeze(1)

        refined_masks, refined_weights, _ = self._predict(
            point_coords=point_coords[:, :, :2],
            point_labels=point_labels,
            boxes=boxes,
            mask_input=low_res_logits.float(),
            multimask_output=True,
            original_size=original_size,
        )

        nms_indices = nms(
            boxes.squeeze(1),
            refined_weights.squeeze(1),
            iou_threshold=nms_iou_threshold,
        )

        # Apply NMS selection
        final_masks = refined_masks[nms_indices]
        final_weights = refined_weights[nms_indices]
        final_point_coords = point_coords[nms_indices]
        final_point_labels = point_labels[nms_indices]

        # Calculate similarity scores
        N, _, H, W = final_masks.shape
        final_masks = final_masks.view(N, H, W)
        similarity_map = similarity_map.view(1, H, W)

        mask_sum = (similarity_map * final_masks).sum(dim=(1, 2))
        mask_area = final_masks.sum(dim=(1, 2))
        
        # Avoid division by zero
        mask_scores = mask_sum / torch.clamp(mask_area, min=1e-6)
        weighted_scores = (mask_scores * final_weights.T).squeeze(0)
        
        # Apply score threshold using masking
        score_keep = weighted_scores > score_threshold
        
        # Final filtering based on scores
        result_masks = torch.where(
            score_keep.unsqueeze(1).unsqueeze(2),
            final_masks,
            torch.zeros_like(final_masks)
        )
        result_coords = torch.where(
            score_keep.unsqueeze(1).unsqueeze(2),
            final_point_coords,
            torch.zeros_like(final_point_coords)
        )
        result_labels = torch.where(
            score_keep.unsqueeze(1),
            final_point_labels,
            torch.zeros_like(final_point_labels)
        )
        return result_masks, result_coords, result_labels



    def _safe_masks_to_boxes(self, masks: torch.Tensor) -> torch.Tensor:
        """
        ONNX-traceable version of masks_to_boxes that handles empty/zero masks gracefully.
        Based on batched_mask_to_box from the codebase.
        
        Args:
            masks (Tensor[N, H, W]): masks to transform
            
        Returns:
            Tensor[N, 4]: bounding boxes in (x1, y1, x2, y2) format
        """
        # Get dimensions
        shape = masks.shape
        h, w = shape[-2:]

        # Get top and bottom edges
        in_height, _ = torch.max(masks, dim=-1)
        in_height_coords = in_height * torch.arange(h, device=masks.device)[None, :]
        bottom_edges, _ = torch.max(in_height_coords, dim=-1)
        in_height_coords = in_height_coords + h * (~in_height)
        top_edges, _ = torch.min(in_height_coords, dim=-1)

        # Get left and right edges
        in_width, _ = torch.max(masks, dim=-2)
        in_width_coords = in_width * torch.arange(w, device=masks.device)[None, :]
        right_edges, _ = torch.max(in_width_coords, dim=-1)
        in_width_coords = in_width_coords + w * (~in_width)
        left_edges, _ = torch.min(in_width_coords, dim=-1)

        # If the mask is empty the right edge will be to the left of the left edge.
        # Replace these boxes with [0, 0, 1, 1] for empty masks
        empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
        out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
        
        # For empty masks, use a small default box instead of [0,0,0,0] to avoid issues
        default_box = torch.tensor([0, 0, 1, 1], device=masks.device, dtype=torch.float)
        out = torch.where(empty_filter.unsqueeze(-1), default_box.unsqueeze(0), out)

        return out

    def export_onnx(
        self,
        output_path: str,
        num_classes: int = 1,
        image_size: tuple[int, int] = (1024, 1024),
        max_pos_points: int = 40,
        max_neg_points: int = 2,
        original_size: tuple[int, int] = (480, 640),
    ) -> None:
        """Export the model to ONNX format."""
        self.eval()
        
        # Create dummy inputs
        batch_size = 1
        image = torch.zeros((batch_size, 3, image_size[0], image_size[1]), dtype=torch.float32)
        
        point_coords = torch.rand((num_classes, max_pos_points + max_neg_points, 2), dtype=torch.float32) * 1024
        pos_labels = torch.ones((num_classes, max_pos_points, 1), dtype=torch.float32)
        neg_labels = torch.zeros((num_classes, max_neg_points, 1), dtype=torch.float32)
        point_labels = torch.cat([pos_labels, neg_labels], dim=1)
        point_scores = torch.rand((num_classes, max_pos_points + max_neg_points, 1), dtype=torch.float32)
        class_point_coords = torch.cat([point_coords, point_scores, point_labels], dim=-1)
        
        similarity_maps = torch.rand(num_classes, original_size[0], original_size[1], dtype=torch.float32)
        original_size = torch.tensor(original_size, dtype=torch.int32)
        
        # Define input and output names
        input_names = [
            "image",
            "class_point_coords", 
            "similarity_maps",
            "original_size"
        ]
        output_names = ["masks", "labels", "coords"]
        
        # Define dynamic axes
        dynamic_axes = {
            "image": {0: "batch_size", 2: "height", 3: "width" },
            "class_point_coords": {0: "num_classes", 1: "num_points"},
            "similarity_maps": {0: "num_classes", 1: "height", 2: "width"},
            "masks": {0: "num_masks"},
            "labels": {0: "num_masks"},
            "coords": {0: "num_masks"},
        }
        
        # Export to ONNX
        torch.onnx.export(
            self,
            (image, class_point_coords, similarity_maps, original_size),
            output_path,
            export_params=True,
            opset_version=20,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=True,
        )
        
        print(f"Model exported to {output_path}")

        exported_model = openvino.convert_model(output_path)
        for i, name in enumerate(output_names):
            traced_names = exported_model.outputs[i].get_names()
            name_found = False
            for traced_name in traced_names:
                if name in traced_name:
                    name_found = True
                    break
            name_found = name_found and bool(len(traced_names))

            if not name_found:
                msg = (
                    f"{name} is not matched with the converted model's traced output names: {traced_names}."
                    " Please check output_names argument of the exporter's constructor."
                )
                raise ValueError(msg)
            exported_model.outputs[i].tensor.set_names({name})

        for i, name in enumerate(input_names):
            traced_names = exported_model.inputs[i].get_names()
            name_found = False
            for traced_name in traced_names:
                if name in traced_name:
                    name_found = True
                    break
            name_found = name_found and bool(len(traced_names))

            if not name_found:
                msg = (
                    f"{name} is not matched with the converted model's traced input names: {traced_names}."
                    " Please check input_names argument of the exporter's constructor."
                )
                raise ValueError(msg)

            exported_model.inputs[i].tensor.set_names({name})
        save_path = output_path.replace(".onnx", ".xml")
        openvino.save_model(exported_model, save_path)


if __name__ == "__main__":
    from getiprompt.models.models import load_sam_model
    from getiprompt.utils.constants import SAMModelName
    sam_predictor = load_sam_model(
        SAMModelName.SAM_HQ_TINY,
        device="cpu",
        precision="fp32",
        apply_onnx_patches=True,
    )
    onnx_decoder = PTSamDecoderONNX(sam_predictor=sam_predictor)
    onnx_decoder.export_onnx(
        output_path="sam_decoder.onnx",
        num_classes=2,
    )