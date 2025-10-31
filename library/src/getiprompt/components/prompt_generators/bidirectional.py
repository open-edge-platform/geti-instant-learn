# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional prompt generator."""

from logging import getLogger

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional
from torchvision import tv_tensors

from getiprompt.components.prompt_generators.base import PromptGenerator
from getiprompt.types import Features, Masks, Similarities

logger = getLogger("Geti Prompt")


class BidirectionalPromptGenerator(PromptGenerator):
    """This class generates prompts for the segmenter.

    This is based on the similarities between the reference and target images.

    Examples:
        >>> import torch
        >>> from getiprompt.processes.prompt_generators import BidirectionalPromptGenerator
        >>> from getiprompt.types import Features, Image, Masks, Priors, Similarities
        >>>
        >>> # Setup
        >>> encoder_input_size=224
        >>> encoder_patch_size=14
        >>> encoder_feature_size=16
        >>> feature_dim = 64
        >>> num_patches = encoder_feature_size * encoder_feature_size
        >>>
        >>> # Create inputs
        >>> ref_feats = Features(torch.rand(num_patches, feature_dim))
        >>> ref_feats.add_local_features(ref_feats.global_features[:6], 1)
        >>> target_feats = Features(torch.rand(num_patches, feature_dim))
        >>> mask = torch.zeros(num_patches); mask[:6] = 1
        >>> ref_masks = Masks(); ref_masks.add(mask, 1)
        >>> image = Image(torch.zeros(encoder_input_size, encoder_input_size, 3))
        >>>
        >>> # Instantiate generator
        >>> prompt_generator = BidirectionalPromptGenerator(
        ...     encoder_input_size=encoder_input_size,
        ...     encoder_patch_size=encoder_patch_size,
        ...     encoder_feature_size=encoder_feature_size,
        ...     num_background_points=2,
        ... )
        >>>
        >>> # Run
        >>> priors, similarities = prompt_generator(
        ...    reference_features=[ref_feats],
        ...    target_features=[target_feats],
        ...    reference_masks=[ref_masks],
        ...    target_images=[image],
        ... )
        >>> isinstance(priors[0], Priors) and priors[0].points.get(1) is not None
        True
        >>> isinstance(similarities[0], Similarities) and similarities[0].get(1) is not None
        True
    """

    def __init__(
        self,
        encoder_input_size: int,
        encoder_patch_size: int,
        encoder_feature_size: int,
        num_background_points: int = 2,
    ) -> None:
        super().__init__()
        self.encoder_input_size = encoder_input_size
        self.num_background_points = num_background_points
        self.encoder_patch_size = encoder_patch_size
        self.encoder_feature_size = encoder_feature_size

    def forward(
        self,
        reference_features: Features,
        reference_masks: list[Masks],
        target_embeddings: torch.Tensor,
        target_images: list[tv_tensors.Image],
    ) -> tuple[list[dict[int, torch.Tensor]], list[Similarities]]:
        """This generates prompt candidates (or priors) based on the similarities.

        This is done between the reference and target images.

        It uses bidirectional matching to create prompts for the segmenter.
        This Prompt Generator computes the similarity map internally.

        Args:
            reference_features(Features): Features object containing reference features
            reference_masks(list[Masks]): List of reference masks, one per reference image instance
            target_embeddings(torch.Tensor): Target embeddings
            target_images(list[tv_tensors.Image]): Target images

        Returns:
            point_prompts(list[dict[int, torch.Tensor]]): List of point prompts (with class_id as key and points as value)
            similarities_per_images(list[Similarities]): List of similarities
        """
        point_prompts: list[dict[int, torch.Tensor]] = []
        similarities_per_image: list[Similarities] = []

        target_features = [Features(global_features=emb) for emb in target_embeddings.unbind(0)]

        # this basically makes a vertical stack + flatten
        flattened_global_features = reference_features.global_features.reshape(
            -1,
            reference_features.global_features.shape[-1],
        )
        reference_masks = self._merge_masks(reference_masks)

        # TODO(Eugene): move _convert_to_image_coords to postprocess, (target_image.shape) used to convert coords
        for target_feature, target_image in zip(target_features, target_images, strict=False):
            class_point_prompts: dict[int, torch.Tensor] = {}
            similarities = Similarities()
            similarity_map = flattened_global_features @ target_feature.global_features.T
            h, w = target_image.shape[-2:]

            for class_id, mask in reference_masks.data.items():
                # NOTE: why select index 0?
                local_mean_reference_feature = reference_features.get_local_features(class_id)[0].mean(
                    dim=0,
                    keepdim=True,
                )
                local_mean_reference_feature /= local_mean_reference_feature.norm(dim=-1, keepdim=True)
                local_similarity = local_mean_reference_feature @ target_feature.global_features.T
                local_similarity = self._resize_similarity_map(local_similarity, target_image.shape[-2:])
                similarities.add(local_similarity, class_id)

                # Select background points based on similarity to averaged local feature
                _, background_indices, background_scores = self._select_background_points(similarity_map, mask)

                # Perform foreground matching
                foreground_indices, foreground_scores, _ = self._perform_matching(similarity_map, mask)

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
                    background_labels = torch.zeros((len(background_points), 1)).to(background_points)
                    background_points = torch.cat([background_points, background_labels], dim=1)
                else:
                    background_points = torch.empty(0, 4).to(similarity_map)

                class_point_prompts[class_id] = torch.cat([foreground_points, background_points])
            point_prompts.append(class_point_prompts)
            similarities_per_image.append(similarities)
        return point_prompts, similarities_per_image

    @staticmethod
    def _perform_matching(similarity_map: torch.Tensor, mask: torch.Tensor) -> tuple[list, torch.Tensor, list]:
        """Perform bidirectional matching using the similarity map for foreground points.

        Linear sum assignment finds the optimal pairing between masked reference features and target features
          to maximize overall similarity.
        Applies a bidirectional check to filter matches.

        Args:
            similarity_map: torch.Tensor - Similarity matrix [num_ref_features, num_target_features]
            mask: torch.Tensor - Mask [num_ref_features]

        Returns:
            tuple containing:
                matched_indices: list - Indices of matched foreground points [ref_indices, target_indices]
                  after bidirectional filtering
                similarity_scores: torch.Tensor - Similarity scores of matched foreground points
                indices_forward: list - Original forward matching indices [original_ref_indices, target_indices]
                  before bidirectional filtering
        """
        masked_ref_indices = mask.flatten().nonzero(as_tuple=True)[0]

        # Handle case where mask is empty
        if masked_ref_indices.numel() == 0:
            empty_indices = [torch.empty(0, dtype=torch.int64)] * 2
            empty_scores = torch.empty(0, dtype=similarity_map.dtype)
            return empty_indices, empty_scores, empty_indices

        # Forward matching (reference -> target) for foreground points
        # select only the features within the mask [num_masked_ref, num_target]
        forward_sim = similarity_map[masked_ref_indices]

        # Perform linear sum assignment for foreground points
        indices_forward = linear_sum_assignment(forward_sim.float().cpu().numpy(), maximize=True)
        indices_forward = [torch.as_tensor(index, dtype=torch.int64) for index in indices_forward]

        # Map masked reference indices back to original similarity map indices
        original_ref_indices = masked_ref_indices[indices_forward[0]]
        sim_scores_forward = similarity_map[original_ref_indices, indices_forward[1]]
        # Store original forward match indices
        original_indices_forward = [original_ref_indices, indices_forward[1]]

        # Backward matching (target -> reference)
        # Select target features that were matched in the forward pass
        target_indices_from_forward = indices_forward[1]
        if target_indices_from_forward.numel() > 0:
            backward_sim = similarity_map.t()[target_indices_from_forward]  # [num_matched_targets, num_ref_features]
            indices_backward = linear_sum_assignment(backward_sim.float().cpu().numpy(), maximize=True)
            indices_backward = [torch.as_tensor(index, dtype=torch.int64) for index in indices_backward]
            # indices_backward[0] refers to the index within target_indices_from_forward
            # indices_backward[1] refers to the index within the full reference features (original ref index)

            # Map backward result indices back to the original forward match indices
            corresponding_forward_indices = indices_backward[0]

            # Check if the backward match's reference feature is within the original mask
            matched_original_ref_indices_backward = indices_backward[1]
            indices_to_keep_mask = torch.isin(
                matched_original_ref_indices_backward,
                masked_ref_indices,
            )

            # Filter the original forward matches based on the backward check
            filtered_forward_indices_idx = corresponding_forward_indices[indices_to_keep_mask]
            if filtered_forward_indices_idx.numel() > 0:
                filtered_indices = [
                    original_indices_forward[0][filtered_forward_indices_idx],
                    original_indices_forward[1][filtered_forward_indices_idx],
                ]
                filtered_sim_scores = sim_scores_forward[filtered_forward_indices_idx]
            else:
                # If no matches pass the filter, result is empty
                filtered_indices = [torch.empty(0, dtype=torch.int64)] * 2
                filtered_sim_scores = torch.empty(0, dtype=similarity_map.dtype)
        else:
            # If no forward matches, result is empty
            filtered_indices = [torch.empty(0, dtype=torch.int64, device=similarity_map.device)] * 2
            filtered_sim_scores = torch.empty(0, dtype=similarity_map.dtype)

        return filtered_indices, filtered_sim_scores, original_indices_forward

    def _select_background_points(
        self,
        similarity_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        """Select the N background points based on lowest average similarity to masked reference features.

        Args:
            similarity_map: torch.Tensor - Similarity matrix [num_ref_features, num_target_features]
            mask: torch.Tensor - Mask indicating relevant reference features [num_ref_features]

        Returns: tuple containing:

        avg_sim_to_masked_ref: torch.Tensor[num_target_features] -
        Average similarity of each target feature to the masked reference features.
        bg_point_indices: torch.Tensor[N, 2] | None - Indices [ref_indices, target_indices] (
        relative to original map) or None if no points found.
        bg_similarity_scores: torch.Tensor[
        N] | None - Similarity scores of background points or None.
        """
        masked_ref_indices = mask.flatten().nonzero(as_tuple=True)[0]
        if masked_ref_indices.numel() == 0:
            return None, None, None

        avg_sim_to_masked_ref = similarity_map[masked_ref_indices].mean(dim=0)
        if avg_sim_to_masked_ref.numel() == 0:
            return None, None, None

        k = min(self.num_background_points, avg_sim_to_masked_ref.numel())
        if k <= 0:
            return None, None, None

        bg_similarity_scores, bg_target_indices = torch.topk(
            avg_sim_to_masked_ref,
            k,
            largest=False,
            sorted=False,
        )

        return avg_sim_to_masked_ref, bg_target_indices, bg_similarity_scores

    def _extract_point_coordinates(self, matched_indices: list, similarity_scores: torch.Tensor) -> torch.Tensor:
        """Extract point coordinates from matched indices.

        Args:
            matched_indices: List of matched indices [reference_indices, target_indices] or [None, target_indices]
            similarity_scores: Similarity scores for the matched points

        Returns:
            torch.Tensor: Points with their similarity scores (N, 3) [x, y, score]
        """
        if not matched_indices or matched_indices[1] is None or matched_indices[1].numel() == 0:
            return torch.empty(0, 3).to(similarity_scores)

        target_indices = matched_indices[1]

        # Extract y and x coordinates from the target indices
        feature_size = self.encoder_feature_size
        y_coords = target_indices // feature_size
        x_coords = target_indices % feature_size

        # Stack coordinates with similarity scores
        # Ensure scores are reshaped correctly if they are not already 1D
        if similarity_scores.dim() != 1:
            similarity_scores = similarity_scores.reshape(-1)

        return torch.stack(
            [
                x_coords.to(similarity_scores),
                y_coords.to(similarity_scores),
                similarity_scores,
            ],
            dim=1,
        )

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

    @staticmethod
    def _merge_masks(reference_masks: list[Masks]) -> Masks:
        """Concatenate the per-image masks in the *height* direction.

        This is done so that, after .flatten(), patch-indices line up with the way reference
        features are stacked.

        Args:
            reference_masks: List[Masks] - List of reference masks, one per reference image instance

        Returns:
            Masks - Merged masks
        """
        if not reference_masks:
            return Masks()

        device = next(iter(reference_masks[0].data.values())).device
        n_images = len(reference_masks)
        h, w = next(iter(reference_masks[0].data.values())).shape[-2:]

        # All class-ids that appear anywhere
        class_ids: set[int] = set()
        for m in reference_masks:
            class_ids.update(m.data.keys())

        merged = Masks()
        for cid in class_ids:
            # Tall canvas: (1, h * n_images, w)
            tall = torch.zeros((1, h * n_images, w), dtype=torch.bool, device=device)

            for img_idx, m in enumerate(reference_masks):
                if cid not in m.data:
                    continue
                block = m.data[cid].any(dim=0, keepdim=True)  # (1, h, w)
                start = img_idx * h
                tall[:, start : start + h, :] = block  # paste with OR

            merged.add(tall, cid)

        return merged

    def _resize_similarity_map(self, similarity_map: torch.Tensor, original_image_size: torch.Tensor) -> torch.Tensor:
        """Resize the similarity map to the original image size.

        Args:
            similarity_map: torch.Tensor - Similarity map [num_target_features]
            original_image_size: torch.Tensor - Original image size (height, width)

        Returns:
            torch.Tensor - Resized similarity map [original_height, original_width]
        """
        similarity_map = (
            similarity_map.reshape(
                self.encoder_input_size // self.encoder_patch_size,
                self.encoder_input_size // self.encoder_patch_size,
            )
            .unsqueeze(0)
            .unsqueeze(0)
        )
        return functional.interpolate(similarity_map, size=original_image_size, mode="bilinear").squeeze(0)
