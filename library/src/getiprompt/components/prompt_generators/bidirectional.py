# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional prompt generator."""

from collections import defaultdict
from logging import getLogger

import torch
from torch import nn
from torch.nn import functional

from getiprompt.components.linear_sum_assignment import linear_sum_assignment

logger = getLogger("Geti Prompt")

__all__ = ["BidirectionalPromptGenerator"]


def _empty_match_result(similarity_map: torch.Tensor) -> tuple[list, torch.Tensor]:
    """Utility to create empty match results consistent with map dtype/device."""
    empty_idx = [torch.empty(0, dtype=torch.int64, device=similarity_map.device)] * 2
    empty_scores = torch.empty(0, dtype=similarity_map.dtype, device=similarity_map.device)
    return empty_idx, empty_scores


class BidirectionalPromptGenerator(nn.Module):
    """Generates point prompts for segmentation based on bidirectional matching.

    This prompt generator uses bidirectional matching between reference and target image features
    to generate foreground point prompts, and selects background points based on low similarity.

    Args:
        encoder_input_size: Size of the encoder input image (e.g., 224, 1024).
        encoder_patch_size: Size of each encoder patch (e.g., 14, 16).
        encoder_feature_size: Size of the feature map grid (e.g., 16, 64).
        num_background_points: Number of background points to generate per class. Default: 2.
    """

    def __init__(
        self,
        encoder_input_size: int,
        encoder_patch_size: int,
        encoder_feature_size: int,
        num_background_points: int = 2,
    ) -> None:
        """Initialize the BidirectionalPromptGenerator."""
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.num_background_points = num_background_points
        self.encoder_patch_size = encoder_patch_size
        self.encoder_feature_size = encoder_feature_size

    @staticmethod
    def ref_to_target_matching(
        similarity_map: torch.Tensor,
        ref_mask_idx: torch.Tensor,
    ) -> tuple[list, torch.Tensor, list]:
        """Perform forward matching (reference -> target) using the similarity map for foreground points.

        Args:
            similarity_map: torch.Tensor - Similarity matrix [num_ref_features, num_target_features]
            ref_mask_idx: torch.Tensor - Indices of masked reference features

        Returns:
            tuple containing:
                matched_ref_idx: torch.Tensor - Indices of matched reference features
                sim_scores: torch.Tensor - Similarity scores of matched reference features
        """
        # Ensure ref_mask_idx is on the same device as similarity_map
        ref_mask_idx = ref_mask_idx.to(similarity_map.device)

        ref_to_target_sim = similarity_map[ref_mask_idx]
        if ref_to_target_sim.numel() == 0:
            return _empty_match_result(similarity_map)

        row_ind, col_ind = linear_sum_assignment(ref_to_target_sim, maximize=True)

        matched_ref_idx = ref_mask_idx[row_ind]
        sim_scores = similarity_map[matched_ref_idx, col_ind]
        return [matched_ref_idx, col_ind], sim_scores

    @staticmethod
    def _perform_matching(similarity_map: torch.Tensor, ref_mask: torch.Tensor) -> tuple[list, torch.Tensor, list]:
        """Perform bidirectional matching using the similarity map for foreground points.

        Linear sum assignment finds the optimal pairing between masked reference features and target features
          to maximize overall similarity.
        Applies a bidirectional check to filter matches.

        Args:
            similarity_map: torch.Tensor - Similarity matrix [num_ref_features, num_target_features]
            ref_mask: torch.Tensor - Mask [num_ref_features]

        Returns:
            tuple containing:
                valid_indices: torch.Tensor - Indices of matched reference features
                valid_scores: torch.Tensor - Similarity scores of matched reference features
        """
        # Ensure ref_idx is on the same device as similarity_map
        ref_idx = ref_mask.nonzero(as_tuple=True)[0].to(similarity_map.device)
        if ref_idx.numel() == 0:
            return _empty_match_result(similarity_map)

        # Forward pass (ref → target)
        (fw_indices, fw_scores) = BidirectionalPromptGenerator.ref_to_target_matching(similarity_map, ref_idx)
        target_idx_fw = fw_indices[1]
        if target_idx_fw.numel() == 0:
            return _empty_match_result(similarity_map)

        # Backward pass (target → ref)
        target_to_ref_sim = similarity_map.t()[target_idx_fw]
        row_ind, col_ind = linear_sum_assignment(target_to_ref_sim, maximize=True)

        # Consistency filter
        valid_ref = torch.isin(col_ind, ref_idx)
        if not valid_ref.any():
            return _empty_match_result(similarity_map)

        valid_fw = row_ind[valid_ref]
        valid_indices = [fw_indices[0][valid_fw], fw_indices[1][valid_fw]]
        valid_scores = fw_scores[valid_fw]
        return valid_indices, valid_scores

    def _select_background_points(
        self,
        similarity_map: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Select the N background points based on lowest average similarity to masked reference features.

        Args:
            similarity_map: torch.Tensor - Similarity matrix [num_ref_features, num_target_features]
            ref_mask: torch.Tensor - Mask indicating relevant reference features [num_ref_features]

        Returns: tuple containing:

        avg_sim_to_masked_ref: torch.Tensor[num_target_features] -
        Average similarity of each target feature to the masked reference features.
        bg_point_indices: torch.Tensor[N, 2] | None - Indices [ref_indices, target_indices] (
        relative to original map) or None if no points found.
        bg_similarity_scores: torch.Tensor[
        N] | None - Similarity scores of background points or None.
        """
        ref_idx = ref_mask.nonzero(as_tuple=True)[0]
        if ref_idx.numel() == 0:
            return None, None, None

        avg_similarity = similarity_map[ref_idx].mean(dim=0)
        if avg_similarity.numel() == 0:
            return None, None, None

        k = min(self.num_background_points, avg_similarity.numel())
        bg_scores, bg_target_idx = torch.topk(avg_similarity, k, largest=False)
        return avg_similarity, bg_target_idx, bg_scores

    def _extract_point_coordinates(self, matched_idx: list, similarity_scores: torch.Tensor) -> torch.Tensor:
        """Extract point coordinates from matched indices.

        Args:
            matched_idx: List of matched indices [reference_indices, target_indices] or [None, target_indices]
            similarity_scores: Similarity scores for the matched points

        Returns:
            torch.Tensor: Points with their similarity scores (N, 3) [x, y, score]
        """
        if not matched_idx or matched_idx[1] is None or matched_idx[1].numel() == 0:
            return torch.empty(0, 3, device=similarity_scores.device)

        tgt_idx = matched_idx[1]
        feat_size = self.encoder_feature_size
        y, x = tgt_idx // feat_size, tgt_idx % feat_size
        x = x.to(similarity_scores.device)
        y = y.to(similarity_scores.device)
        similarity_scores = similarity_scores.flatten()
        return torch.stack((x, y, similarity_scores), dim=1)

    def _convert_to_image_coords(self, points: torch.Tensor, ori_size: tuple[int, int]) -> torch.Tensor:
        """Convert points from feature grid coordinates to original image coordinates.

        Args:
            points: Points in feature grid coordinates (x, y, score)
            ori_size: Original image size (height, width)

        Returns:
            torch.Tensor: Points in image coordinates (x, y, score)
        """
        if points.numel() == 0:
            return torch.empty(0, 3).to(points)

        # Convert feature grid coordinates to patch coordinates
        patch_size = self.encoder_patch_size
        encoder_input_size = self.encoder_input_size
        x_image = points[:, 0] * patch_size + patch_size // 2
        y_image = points[:, 1] * patch_size + patch_size // 2

        # Scale to original image size
        scale_w = ori_size[1] / encoder_input_size
        scale_h = ori_size[0] / encoder_input_size

        x_image *= scale_w
        y_image *= scale_h

        # Combine with similarity scores and round coordinates to nearest integer
        return torch.stack(
            [
                torch.round(x_image).to(torch.int64),
                torch.round(y_image).to(torch.int64),
                points[:, 2],
            ],
            dim=1,
        )

    def _resize_similarity_map(self, similarity_map: torch.Tensor, ori_size: torch.Tensor) -> torch.Tensor:
        """Resize the similarity map to the original image size.

        Args:
            similarity_map: torch.Tensor - Similarity map [num_target_features]
            ori_size: torch.Tensor - Original image size (height, width)

        Returns:
            torch.Tensor - Resized similarity map [ori_height, ori_width]
        """
        similarity_map = (
            similarity_map.reshape(
                self.encoder_input_size // self.encoder_patch_size,
                self.encoder_input_size // self.encoder_patch_size,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return functional.interpolate(similarity_map, size=ori_size, mode="bilinear").squeeze(0)

    def forward(
        self,
        ref_embeddings: dict[int, torch.Tensor],
        masked_ref_embeddings: dict[int, torch.Tensor],
        flatten_ref_masks: dict[int, torch.Tensor],
        target_embeddings: torch.Tensor,
        original_sizes: list[tuple[int, int]],
    ) -> tuple[list[dict[int, torch.Tensor]], list[dict[int, torch.Tensor]]]:
        """This generates prompt candidates (or priors) based on the similarities.

        This is done between the reference and target images.

        It uses bidirectional matching to create prompts for the segmenter.
        This Prompt Generator computes the similarity map internally.

        Args:
            ref_embeddings(dict[int, torch.Tensor]): Reference embeddings grouped by class_id.
            masked_ref_embeddings(dict[int, torch.Tensor]): Dictionary with class_id as key and
                masked reference embeddings as value.
            flatten_ref_masks(dict[int, torch.Tensor]): Dictionary of flattened reference masks, with class_id as key
                and flattened reference masks as value.
            target_embeddings(torch.Tensor): Target embeddings
            original_sizes(list[tuple[int, int]]): Original sizes of the target images

        Returns:
            point_prompts(list[dict[int, torch.Tensor]]):
                List of point prompts (with class_id as key and points as value)
            similarities_per_images(list[dict[int, torch.Tensor]]): List of similarities dictionaries
        """
        point_prompts: list[dict[int, torch.Tensor]] = []
        similarities_per_image: list[dict[int, torch.Tensor]] = []

        for target_embed, original_size in zip(target_embeddings, original_sizes, strict=False):
            class_point_prompts: dict[int, torch.Tensor] = {}
            similarities: dict[int, list[torch.Tensor]] = defaultdict(list)
            h, w = original_size

            for class_id, flatten_ref_mask in flatten_ref_masks.items():
                similarity_map = ref_embeddings[class_id] @ target_embed.T
                local_ref_embedding = masked_ref_embeddings[class_id]
                local_similarity = local_ref_embedding @ target_embed.T
                local_similarity = self._resize_similarity_map(local_similarity, original_size)
                similarities[class_id].append(local_similarity)

                # Select background points based on similarity to averaged local feature
                _, background_indices, background_scores = self._select_background_points(
                    similarity_map,
                    flatten_ref_mask,
                )

                # Perform foreground matching
                foreground_indices, foreground_scores = self._perform_matching(similarity_map, flatten_ref_mask)

                # Process foreground points
                if len(foreground_scores) > 0:
                    foreground_points = self._extract_point_coordinates(foreground_indices, foreground_scores)
                    foreground_points = self._convert_to_image_coords(foreground_points, ori_size=(h, w))
                    foreground_labels = torch.ones((len(foreground_points), 1)).to(foreground_points)
                    foreground_points = torch.cat([foreground_points, foreground_labels], dim=1)
                else:
                    foreground_points = torch.empty(0, 4).to(similarity_map)

                # Process background points
                if background_indices is not None and background_scores is not None and background_indices.numel() > 0:
                    background_points = self._extract_point_coordinates([None, background_indices], background_scores)
                    background_points = self._convert_to_image_coords(background_points, ori_size=(h, w))
                    background_labels = -torch.ones((len(background_points), 1)).to(background_points)
                    background_points = torch.cat([background_points, background_labels], dim=1)
                else:
                    background_points = torch.empty(0, 4).to(similarity_map)

                class_point_prompts[class_id] = torch.cat([foreground_points, background_points])
            point_prompts.append(class_point_prompts)

            # Concatenate all tensors once per class
            concatenated_similarities = {
                class_id: torch.cat(tensor_list, dim=0) for class_id, tensor_list in similarities.items()
            }
            similarities_per_image.append(concatenated_similarities)
        return point_prompts, similarities_per_image
