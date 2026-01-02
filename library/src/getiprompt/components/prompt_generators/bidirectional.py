# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional prompt generator."""

from logging import getLogger

import torch
from torch import nn

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
    It also filters to keep only the top-scoring foreground points.

    All outputs are tensors for full traceability (ONNX/TorchScript compatible).

    Args:
        encoder_input_size: Size of the encoder input image (e.g., 224, 1024).
        encoder_patch_size: Size of each encoder patch (e.g., 14, 16).
        encoder_feature_size: Size of the feature map grid (e.g., 16, 64).
        num_foreground_points: Maximum number of foreground points to keep per class. Default: 40.
        num_background_points: Number of background points to generate per class. Default: 2.
        max_points: Maximum total points per category for output padding. Default: 64.
    """

    def __init__(
        self,
        encoder_input_size: int,
        encoder_patch_size: int,
        encoder_feature_size: int,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        max_points: int = 64,
    ) -> None:
        """Initialize the BidirectionalPromptGenerator."""
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.encoder_patch_size = encoder_patch_size
        self.encoder_feature_size = encoder_feature_size
        self.num_foreground_points = num_foreground_points
        self.num_background_points = num_background_points
        self.max_points = max_points

    @staticmethod
    def ref_to_target_matching(
        similarity_map: torch.Tensor,
        ref_mask_idx: torch.Tensor,
    ) -> tuple[list, torch.Tensor]:
        """Perform forward matching (reference -> target) using the similarity map for foreground points.

        Args:
            similarity_map: Similarity matrix [num_ref_features, num_target_features]
            ref_mask_idx: Indices of masked reference features

        Returns:
            tuple containing:
                matched_indices: List of [matched_ref_idx, matched_target_idx]
                sim_scores: Similarity scores of matched features
        """
        ref_mask_idx = ref_mask_idx.to(similarity_map.device)

        ref_to_target_sim = similarity_map[ref_mask_idx]
        if ref_to_target_sim.numel() == 0:
            return _empty_match_result(similarity_map)

        row_ind, col_ind = linear_sum_assignment(ref_to_target_sim, maximize=True)

        matched_ref_idx = ref_mask_idx[row_ind]
        sim_scores = similarity_map[matched_ref_idx, col_ind]
        return [matched_ref_idx, col_ind], sim_scores

    @staticmethod
    def _perform_matching(similarity_map: torch.Tensor, ref_mask: torch.Tensor) -> tuple[list, torch.Tensor]:
        """Perform bidirectional matching using the similarity map for foreground points.

        Linear sum assignment finds the optimal pairing between masked reference features and target
        features to maximize overall similarity. Applies a bidirectional check to filter matches.

        Args:
            similarity_map: Similarity matrix [num_ref_features, num_target_features]
            ref_mask: Mask [num_ref_features]

        Returns:
            tuple containing:
                valid_indices: List of [valid_ref_idx, valid_target_idx]
                valid_scores: Similarity scores of valid matches
        """
        ref_idx = ref_mask.nonzero(as_tuple=True)[0].to(similarity_map.device)
        if ref_idx.numel() == 0:
            return _empty_match_result(similarity_map)

        # Forward pass (ref → target)
        fw_indices, fw_scores = BidirectionalPromptGenerator.ref_to_target_matching(similarity_map, ref_idx)
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
            similarity_map: Similarity matrix [num_ref_features, num_target_features]
            ref_mask: Mask indicating relevant reference features [num_ref_features]

        Returns:
            tuple containing:
                avg_similarity: Average similarity to masked ref features [num_target_features]
                bg_target_idx: Indices of background points in target
                bg_scores: Similarity scores of background points
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
            matched_idx: List of [reference_indices, target_indices] or [None, target_indices]
            similarity_scores: Similarity scores for the matched points

        Returns:
            Points with their similarity scores (N, 3) [x, y, score]
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

    def _convert_to_image_coords(
        self,
        points: torch.Tensor,
        original_size: torch.Tensor,
    ) -> torch.Tensor:
        """Convert points from feature grid coordinates to original image coordinates.

        Args:
            points: Points in feature grid coordinates (x, y, score)
            original_size: Original image size tensor [H, W]

        Returns:
            Points in image coordinates (x, y, score)
        """
        if points.numel() == 0:
            return torch.empty(0, 3).to(points)

        patch_size = self.encoder_patch_size
        encoder_input_size = self.encoder_input_size
        x_image = points[:, 0] * patch_size + patch_size // 2
        y_image = points[:, 1] * patch_size + patch_size // 2

        scale_h = original_size[0].float() / encoder_input_size
        scale_w = original_size[1].float() / encoder_input_size

        x_image = x_image * scale_w
        y_image = y_image * scale_h

        return torch.stack(
            [
                torch.round(x_image).to(torch.int64),
                torch.round(y_image).to(torch.int64),
                points[:, 2],
            ],
            dim=1,
        )

    def _filter_foreground_points(self, foreground_points: torch.Tensor) -> torch.Tensor:
        """Filter foreground points to keep only top-scoring ones.

        Args:
            foreground_points: Foreground points [N, 4] with (x, y, score, label)

        Returns:
            Filtered foreground points [M, 4] where M <= num_foreground_points
        """
        if foreground_points.shape[0] <= self.num_foreground_points:
            return foreground_points

        # Sort by score (column 2) descending and take top k
        _, top_indices = torch.topk(foreground_points[:, 2], self.num_foreground_points)
        return foreground_points[top_indices]

    def _process_single_category(
        self,
        ref_embed: torch.Tensor,
        masked_ref_embed: torch.Tensor,
        flatten_ref_mask: torch.Tensor,
        target_embed: torch.Tensor,
        original_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process a single category against a target image.

        Args:
            ref_embed: Reference embeddings [num_patches_total, embed_dim]
            masked_ref_embed: Averaged masked embedding [embed_dim]
            flatten_ref_mask: Flattened mask [num_patches_total]
            target_embed: Target embeddings [num_patches, embed_dim]
            original_size: Original image size tensor [H, W]

        Returns:
            points: Point prompts [N, 4] with (x, y, score, label), filtered to max points
            similarity: Similarity map at feature grid size [feat_size, feat_size]
        """
        # Compute similarity maps
        # Local similarity for output (at feature grid size, not resized)
        local_similarity = masked_ref_embed.unsqueeze(0) @ target_embed.T  # [1, num_patches]
        feat_size = self.encoder_feature_size
        local_similarity_grid = local_similarity.reshape(feat_size, feat_size)

        # Full similarity map for matching
        similarity_map = ref_embed @ target_embed.T  # [num_patches_total, num_patches]

        # Select background points
        _, background_indices, background_scores = self._select_background_points(
            similarity_map,
            flatten_ref_mask,
        )

        # Perform foreground matching
        foreground_indices, foreground_scores = self._perform_matching(similarity_map, flatten_ref_mask)

        # Process foreground points
        if len(foreground_scores) > 0:
            foreground_points = self._extract_point_coordinates(foreground_indices, foreground_scores)
            foreground_points = self._convert_to_image_coords(foreground_points, original_size)
            foreground_labels = torch.ones((len(foreground_points), 1)).to(foreground_points)
            foreground_points = torch.cat([foreground_points, foreground_labels], dim=1)
            # Filter to keep only top-scoring foreground points
            foreground_points = self._filter_foreground_points(foreground_points)
        else:
            foreground_points = torch.empty(0, 4).to(similarity_map)

        # Process background points
        if background_indices is not None and background_scores is not None and background_indices.numel() > 0:
            background_points = self._extract_point_coordinates([None, background_indices], background_scores)
            background_points = self._convert_to_image_coords(background_points, original_size)
            background_labels = -torch.ones((len(background_points), 1)).to(background_points)
            background_points = torch.cat([background_points, background_labels], dim=1)
        else:
            background_points = torch.empty(0, 4).to(similarity_map)

        points = torch.cat([foreground_points, background_points])
        return points, local_similarity_grid

    def _pad_points(self, points: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Pad points tensor to max_points size.

        Args:
            points: Points tensor [N, 4]
            device: Target device
            dtype: Target dtype

        Returns:
            Padded points tensor [max_points, 4]
        """
        num_points = points.shape[0]
        if num_points >= self.max_points:
            return points[: self.max_points]

        # Pad with zeros
        padding = torch.zeros(self.max_points - num_points, 4, device=device, dtype=dtype)
        return torch.cat([points, padding], dim=0)

    def forward(
        self,
        ref_embeddings: torch.Tensor,
        masked_ref_embeddings: torch.Tensor,
        flatten_ref_masks: torch.Tensor,
        category_ids: torch.Tensor,
        target_embeddings: torch.Tensor,
        original_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate prompt candidates based on reference-target similarities.

        Uses bidirectional matching to create point prompts for the segmenter.
        Automatically filters to keep only top-scoring foreground points.
        All outputs are tensors for full traceability.

        Args:
            ref_embeddings(dict[int, torch.Tensor]): Reference embeddings grouped by class_id.
            masked_ref_embeddings(dict[int, torch.Tensor]): Dictionary with class_id as key and
                masked reference embeddings as value.
            flatten_ref_masks(dict[int, torch.Tensor]): Dictionary of flattened reference masks, with class_id as key
                and flattened reference masks as value.
            target_embeddings(torch.Tensor): Target embeddings
            original_sizes(list[tuple[int, int]]): Original sizes of the target images

        Returns:
            point_prompts: [T, C, max_points, 4] - filtered and padded point prompts
            num_points: [T, C] - actual valid point counts per (target, category)
            similarities: [T, C, feat_size, feat_size] - similarity maps at feature grid size
        """
        num_targets = target_embeddings.shape[0]
        num_categories = category_ids.shape[0]
        feat_size = self.encoder_feature_size
        device = target_embeddings.device
        dtype = target_embeddings.dtype

        # Pre-allocate output tensors
        # point_prompts = torch.zeros(num_targets, num_categories, self.max_points, 4, device=device, dtype=dtype)
        # num_points = torch.zeros(num_targets, num_categories, device=device, dtype=torch.int64)
        # similarities = torch.zeros(num_targets, num_categories, feat_size, feat_size, device=device, dtype=dtype)

        # for t_idx in range(num_targets):
        #     target_embed = target_embeddings[t_idx]
        #     original_size = original_sizes[t_idx]

        #     for c_idx in range(num_categories):
        #         ref_embed = ref_embeddings[c_idx]
        #         masked_embed = masked_ref_embeddings[c_idx]
        #         mask = flatten_ref_masks[c_idx]

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

                points, similarity = self._process_single_category(
                    ref_embed,
                    masked_embed,
                    mask,
                    target_embed,
                    original_size,
                )

                # Store actual count
                actual_num_points = min(points.shape[0], self.max_points)
                num_points[t_idx, c_idx] = actual_num_points

                # Pad and store points
                point_prompts[t_idx, c_idx] = self._pad_points(points, device, dtype)

                # Store similarity
                similarities[t_idx, c_idx] = similarity

        return point_prompts, num_points, similarities
