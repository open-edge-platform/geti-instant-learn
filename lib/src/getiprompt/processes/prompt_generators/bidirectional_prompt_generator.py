# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional prompt generator."""

import torch
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from getiprompt.processes.prompt_generators.prompt_generator_base import FeaturePromptGenerator
from getiprompt.types import Features, Masks, Priors, Similarities
from getiprompt.types.image import Image


class BidirectionalPromptGenerator(FeaturePromptGenerator):
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

    def __call__(
        self,
        reference_features: Features,
        target_features: list[Features],
        reference_masks: list[Masks],
        target_images: list[Image],
    ) -> tuple[list[Priors], list[Similarities]]:
        """This generates prompt candidates (or priors) based on the similarities.

        This is done between the reference and target images.

        It uses bidirectional matching to create prompts for the segmenter.
        This Prompt Generator computes the similarity map internally.

        Args:
            reference_features: Features object containing reference features
            target_features: List[Features] List of target features, one per target image instance
            reference_masks: List[Masks] List of reference masks, one per reference image instance
            target_images: ListImage] The target images

        Returns:
            List[Priors] List of priors, one per target image instance
        """
        priors_per_image: list[Priors] = []
        flattened_global_features = reference_features.global_features.reshape(
            -1,
            reference_features.global_features.shape[-1],
        )  # this basically makes a vertical stack + flatten
        reference_masks = self._merge_masks(reference_masks)

        similarities_per_images = []

        for target_image_features, target_image in zip(target_features, target_images, strict=False):
            priors = Priors()
            similarities = Similarities()
            similarity_map = flattened_global_features @ target_image_features.global_features.T

            for class_id, mask in reference_masks.data.items():
                # Construct local similarity map. This can later be used to filter out masks.
                local_mean_reference_feature = reference_features.get_local_features(
                    class_id,
                )[0].mean(dim=0, keepdim=True)
                local_mean_reference_feature = local_mean_reference_feature / local_mean_reference_feature.norm(
                    dim=-1, keepdim=True
                )
                local_similarity_map = local_mean_reference_feature @ target_image_features.global_features.T
                local_similarity_map = self._resize_similarity_map(
                    local_similarity_map,
                    target_image.size,
                )
                similarities.add(local_similarity_map, class_id)

                # Select background points based on similarity to averaged local feature
                _averaged_feature_sim_map, bg_target_indices, bg_similarity_scores = self._select_background_points(
                    similarity_map, mask
                )

                # Perform foreground matching
                matched_indices, similarity_scores, _ = self._perform_matching(
                    similarity_map,
                    mask,
                )

                # Process foreground points
                if len(similarity_scores) > 0:
                    fg_points = self._extract_point_coordinates(
                        matched_indices,
                        similarity_scores,
                    )
                    image_level_fg_points = self._transform_to_image_coordinates(
                        fg_points,
                        original_image_size=target_image.size,
                    )
                    fg_point_labels = torch.ones(
                        (len(image_level_fg_points), 1),
                        device=image_level_fg_points.device,
                    )
                    image_level_fg_points = torch.cat(
                        [image_level_fg_points, fg_point_labels],
                        dim=1,
                    )
                    fg_bg_points = image_level_fg_points
                else:
                    fg_bg_points = torch.empty(0, 4, device=similarity_map.device)

                # Process background points
                if bg_target_indices is not None and bg_similarity_scores is not None and bg_target_indices.numel() > 0:
                    bg_points = self._extract_point_coordinates(
                        [None, bg_target_indices],
                        bg_similarity_scores,
                    )
                    image_level_bg_points = self._transform_to_image_coordinates(
                        bg_points,
                        original_image_size=target_image.size,
                    )
                    bg_point_labels = torch.zeros(
                        (len(image_level_bg_points), 1),
                        device=image_level_bg_points.device,
                    )
                    image_level_bg_points = torch.cat(
                        [image_level_bg_points, bg_point_labels],
                        dim=1,
                    )
                    fg_bg_points = torch.cat([fg_bg_points, image_level_bg_points])
                else:
                    print(f"No BG points found for class {class_id}")

                priors.points.add(fg_bg_points, class_id)
            priors_per_image.append(priors)
            similarities_per_images.append(similarities)
        return priors_per_image, similarities_per_images

    @staticmethod
    def _perform_matching(
        similarity_map: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[list, torch.Tensor, list]:
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
        if masked_ref_indices.numel() == 0:
            # Handle case where mask is empty
            empty_indices = [
                torch.empty(0, dtype=torch.int64, device=similarity_map.device),
            ] * 2
            empty_scores = torch.empty(
                0,
                dtype=similarity_map.dtype,
                device=similarity_map.device,
            )
            return (
                empty_indices,
                empty_scores,
                empty_indices,
            )

        # Forward matching (reference -> target) for foreground points
        forward_sim = similarity_map[
            masked_ref_indices
        ]  # select only the features within the mask [num_masked_ref, num_target]

        # Perform linear sum assignment for foreground points
        indices_forward = linear_sum_assignment(
            forward_sim.float().cpu().numpy(),
            maximize=True,
        )
        indices_forward = [
            torch.as_tensor(index, dtype=torch.int64, device=similarity_map.device) for index in indices_forward
        ]
        # Map masked reference indices back to original similarity map indices
        original_ref_indices = masked_ref_indices[indices_forward[0]]
        sim_scores_forward = similarity_map[original_ref_indices, indices_forward[1]]
        original_indices_forward = [
            original_ref_indices,
            indices_forward[1],
        ]  # Store original forward match indices

        # Backward matching (target -> reference)
        # Select target features that were matched in the forward pass
        target_indices_from_forward = indices_forward[1]
        if target_indices_from_forward.numel() > 0:
            backward_sim = similarity_map.t()[target_indices_from_forward]  # [num_matched_targets, num_ref_features]
            indices_backward = linear_sum_assignment(
                backward_sim.float().cpu().numpy(),
                maximize=True,
            )
            indices_backward = [
                torch.as_tensor(index, dtype=torch.int64, device=similarity_map.device) for index in indices_backward
            ]
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
                filtered_indices = [
                    torch.empty(0, dtype=torch.int64, device=similarity_map.device),
                ] * 2
                filtered_sim_scores = torch.empty(
                    0,
                    dtype=similarity_map.dtype,
                    device=similarity_map.device,
                )
        else:
            # If no forward matches, result is empty
            filtered_indices = [
                torch.empty(0, dtype=torch.int64, device=similarity_map.device),
            ] * 2
            filtered_sim_scores = torch.empty(
                0,
                dtype=similarity_map.dtype,
                device=similarity_map.device,
            )

        return (
            filtered_indices,
            filtered_sim_scores,
            original_indices_forward,
        )

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

    def _extract_point_coordinates(
        self,
        matched_indices: list,
        similarity_scores: torch.Tensor,
    ) -> torch.Tensor:
        """Extract point coordinates from matched indices.

        Args:
            matched_indices: List of matched indices [reference_indices, target_indices] or [None, target_indices]
            similarity_scores: Similarity scores for the matched points

        Returns:
            torch.Tensor: Points with their similarity scores (N, 3) [x, y, score]
        """
        if not matched_indices or matched_indices[1] is None or matched_indices[1].numel() == 0:
            return torch.empty(
                0,
                3,
                dtype=similarity_scores.dtype,
                device=similarity_scores.device,
            )

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
                x_coords.to(similarity_scores.dtype),
                y_coords.to(similarity_scores.dtype),
                similarity_scores,
            ],
            dim=1,
        )

    def _transform_to_image_coordinates(
        self,
        points: torch.Tensor,
        original_image_size: torch.Tensor,
    ) -> torch.Tensor:
        """Transform points from feature grid coordinates to original image coordinates.

        Args:
            points: Points in feature grid coordinates (x, y, score)
            original_image_size: Original image size (height, width)

        Returns:
            torch.Tensor: Points in image coordinates (x, y, score)
        """
        if points.numel() == 0:
            return torch.empty(0, 3, dtype=points.dtype, device=points.device)

        # Get encoder configuration from state
        patch_size = self.encoder_patch_size
        encoder_input_size = self.encoder_input_size

        # Convert feature grid coordinates to patch coordinates
        x_image = points[:, 0] * patch_size + patch_size // 2
        y_image = points[:, 1] * patch_size + patch_size // 2

        # Scale to original image size
        scale_w = original_image_size[1] / encoder_input_size
        scale_h = original_image_size[0] / encoder_input_size

        x_image = x_image * scale_w
        y_image = y_image * scale_h

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

    def _resize_similarity_map(
        self,
        similarity_map: torch.Tensor,
        original_image_size: torch.Tensor,
    ) -> torch.Tensor:
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
        return F.interpolate(
            similarity_map,
            size=original_image_size,
            mode="bilinear",
        ).squeeze(0)
