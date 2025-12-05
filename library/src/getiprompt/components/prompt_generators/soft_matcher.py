# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Soft-matching prompt generator."""

from collections import defaultdict
from logging import getLogger

import torch
from torch.nn import functional

from getiprompt.components.prompt_generators import BidirectionalPromptGenerator

logger = getLogger("Geti Prompt")


class SoftmatcherPromptGenerator(BidirectionalPromptGenerator):
    """This class generates prompts for the segmenter based on the similarities between the reference and target images.

    Args:
        encoder_input_size: int - The size of the encoder input image.
        encoder_patch_size: int - The size of the encoder patch.
        encoder_feature_size: int - The size of the encoder feature.
        num_background_points: int - The number of background points to generate.
        num_foreground_points: int - The number of foreground points to generate.
        use_sampling: bool - Whether to use sampling instead of thresholding to select points.
        use_spatial_sampling: bool - Whether to use spatial sampling (NMS) to select points.
        approximate_matching: bool - Whether to use approximate matching, this will use RFF to approximate
            the similarity map.
        softmatching_score_threshold: float - Threshold for selecting points based on
            normalized soft correspondence scores.
        softmatching_bidirectional: bool - Whether to use bidirectional softmatching.

    Examples:
        >>> import torch
        >>> from getiprompt.components.prompt_generators import SoftmatcherPromptGenerator
        >>> from getiprompt.types import Masks
        >>> from torchvision import tv_tensors
        >>>
        >>> # Setup
        >>> encoder_input_size=224
        >>> encoder_patch_size=14
        >>> encoder_feature_size=16
        >>> feature_dim = 64
        >>> num_patches = encoder_feature_size * encoder_feature_size
        >>>
        >>> # Create inputs
        >>> ref_embeds = torch.rand(num_patches, feature_dim)
        >>> target_embeddings = torch.rand(1, num_patches, feature_dim)
        >>> mask = torch.zeros(num_patches); mask[:6] = 1
        >>> reference_masks = Masks(); reference_masks.add(mask, 1)
        >>> image = tv_tensors.Image(torch.zeros(encoder_input_size, encoder_input_size, 3))
        >>>
        >>> # Instantiate and run
        >>> generator = SoftmatcherPromptGenerator(
        ...     encoder_input_size=encoder_input_size,
        ...     encoder_patch_size=encoder_patch_size,
        ...     encoder_feature_size=encoder_feature_size,
        ... )
        >>> point_prompts, sims = generator(
        ...    ref_embeds=ref_embeds,
        ...    masked_ref_embeds={1: ref_embeds},
        ...    reference_masks=[reference_masks],
        ...    target_embeddings=target_embeddings,
        ...    target_images=[image]
        ... )
        >>> isinstance(point_prompts[0], dict) and 1 in point_prompts[0]
        True
        >>> isinstance(sims[0], dict) and sims[0].get(1) is not None
        True
    """

    def __init__(
        self,
        encoder_input_size: int,
        encoder_patch_size: int,
        encoder_feature_size: int,
        num_background_points: int = 2,
        num_foreground_points: int = 40,
        use_sampling: bool = False,
        use_spatial_sampling: bool = False,
        approximate_matching: bool = False,
        softmatching_score_threshold: float = 0.4,
        softmatching_bidirectional: bool = False,
    ) -> None:
        """Initialize the SoftmatcherPromptGenerator."""
        super().__init__(
            encoder_input_size=encoder_input_size,
            encoder_patch_size=encoder_patch_size,
            encoder_feature_size=encoder_feature_size,
            num_background_points=num_background_points,
        )
        self.num_foreground_points = num_foreground_points
        self.use_sampling = use_sampling
        self.use_spatial_sampling = use_spatial_sampling
        self.approximate_matching = approximate_matching
        self.softmatching_score_threshold = softmatching_score_threshold
        self.softmatching_bidirectional = softmatching_bidirectional

    @staticmethod
    def _unidirectional_soft_matching(forward_sim: torch.Tensor, reg: float) -> torch.Tensor:
        """Computes unidirectional soft correspondence scores."""
        log_softmax_sim = functional.log_softmax(forward_sim / reg, dim=1)
        # The geometric mean in log-space is the arithmetic mean of logs.
        return log_softmax_sim.mean(dim=0)

    @staticmethod
    def _bidirectional_soft_matching(
        forward_sim: torch.Tensor,
        similarity_map_for_backward: torch.Tensor,
        masked_ref_indices: torch.Tensor,
        reg: float,
    ) -> torch.Tensor:
        """Computes bidirectional soft correspondence scores."""
        # Forward pass: P(target | masked_ref), softmax over targets
        log_softmax_forward = functional.log_softmax(forward_sim / reg, dim=1)

        # Backward pass: P(masked_ref | target), softmax over all references
        log_softmax_backward_full = functional.log_softmax(similarity_map_for_backward / reg, dim=0)
        log_softmax_backward_masked = log_softmax_backward_full[masked_ref_indices, :]

        # Combine probabilities in log-space (equivalent to multiplication)
        log_bidirectional_prob = log_softmax_forward + log_softmax_backward_masked

        # Aggregate scores for each target feature. Using mean over reference features.
        return log_bidirectional_prob.mean(dim=0)

    @staticmethod
    def _calculate_similarity(
        use_rff: bool,
        masked_ref_indices: torch.Tensor,
        ref_features: torch.Tensor | None,
        target_features: torch.Tensor | None,
        similarity_map: torch.Tensor | None,
        bidirectional: bool,
        rff_dim: int,
        rff_sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Calculate forward and backward similarity maps based on configuration.

        Args:
            use_rff: bool - Whether to use RFF approximation.
            masked_ref_indices: torch.Tensor - Indices of masked reference features.
            ref_features: torch.Tensor | None - Reference features.
            target_features: torch.Tensor | None - Target features.
            similarity_map: torch.Tensor | None - Similarity map.
            bidirectional: bool - Whether to use bidirectional softmatching.
            rff_dim: int - Dimension of the RFF projection.
            rff_sigma: float - Width of the Gaussian kernel for RFF.

        Returns:
            tuple[torch.Tensor | None, torch.Tensor | None]:
                Tuple containing:
                    forward_sim: torch.Tensor | None - Forward similarity map.
                    similarity_map_for_backward: torch.Tensor | None - Backward similarity map.

        Raises:
            ValueError: If ref_features and target_features are not provided when use_rff is True.
            ValueError: If similarity_map is not provided when use_rff is False.
        """
        similarity_map_for_backward = None
        if use_rff:
            if ref_features is None or target_features is None:
                msg = "ref_features and target_features must be provided when use_rff is True."
                raise ValueError(msg)

            masked_ref_features = ref_features[masked_ref_indices]
            n_ref, d = masked_ref_features.shape
            n_target, _ = target_features.shape

            if n_ref == 0 or n_target == 0:
                return None, None

            # RFF approximation
            projection_matrix = (torch.randn(d, rff_dim // 2, device=ref_features.device) / rff_sigma).to(
                ref_features.dtype,
            )
            phi_ref = SoftmatcherPromptGenerator._apply_rff(masked_ref_features, projection_matrix)
            phi_target = SoftmatcherPromptGenerator._apply_rff(target_features, projection_matrix)
            forward_sim = phi_ref @ phi_target.T

            if bidirectional:
                phi_all_ref = SoftmatcherPromptGenerator._apply_rff(ref_features, projection_matrix)
                similarity_map_for_backward = phi_all_ref @ phi_target.T
        else:
            if similarity_map is None:
                msg = "similarity_map must be provided when use_rff is False."
                raise ValueError(msg)
            forward_sim = similarity_map[masked_ref_indices]
            if bidirectional:
                similarity_map_for_backward = similarity_map

        return forward_sim, similarity_map_for_backward

    @staticmethod
    def _process_scores_and_select_points(
        soft_correspondence_scores: torch.Tensor,
        score_threshold: float,
        device: torch.device,
        use_sampling: bool = False,
        use_spatial_sampling: bool = False,
        num_samples: int = 40,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Normalize scores, select points above threshold, and then optionally sample from them."""
        map_size = int(soft_correspondence_scores.shape[0] ** 0.5)
        soft_sim_map = soft_correspondence_scores.reshape(1, 1, map_size, map_size)

        # Normalize scores to [0, 1] to make thresholding independent of `reg` and score distribution.
        scores_min, scores_max = soft_correspondence_scores.min(), soft_correspondence_scores.max()
        normalized_scores = (soft_correspondence_scores - scores_min) / (scores_max - scores_min + 1e-6)

        if use_spatial_sampling:
            # Non-Maximum Suppression to get spatially diverse points.
            score_map = soft_correspondence_scores.reshape(1, 1, map_size, map_size)

            # Find local maxima using max pooling
            pooled_map = functional.max_pool2d(score_map, kernel_size=3, stride=1, padding=1)
            local_maxima = score_map == pooled_map

            # Also apply the score threshold to the normalized scores
            normalized_score_map = normalized_scores.reshape(1, 1, map_size, map_size)
            threshold_mask = normalized_score_map >= score_threshold

            # Combine NMS with thresholding
            peak_mask = local_maxima & threshold_mask
            peak_indices_flat = torch.where(peak_mask.flatten())[0]

            if peak_indices_flat.numel() > 0:
                # Select the top N points from the peaks
                peak_scores = soft_correspondence_scores[peak_indices_flat]
                num_to_sample = min(num_samples, peak_scores.numel())

                if num_to_sample > 0:
                    _, top_peak_indices_of_peaks = torch.topk(peak_scores, num_to_sample)
                    target_indices = peak_indices_flat[top_peak_indices_of_peaks]
                else:
                    target_indices = torch.empty(0, dtype=torch.int64, device=device)
            else:
                target_indices = torch.empty(0, dtype=torch.int64, device=device)

        else:
            # Select candidate points with scores above the threshold.
            candidate_indices = torch.where(normalized_scores >= score_threshold)[0]

            if use_sampling and candidate_indices.numel() > 0:
                # Probabilistic sampling from the candidate points.
                candidate_scores = soft_correspondence_scores[candidate_indices]
                if candidate_scores.sum() == 0:
                    # If all scores are zero, sample uniformly from candidates.
                    probabilities = torch.ones_like(candidate_scores) / candidate_scores.numel()
                else:
                    probabilities = candidate_scores / candidate_scores.sum()

                # Ensure we don't request more samples than available points.
                num_to_sample = min(num_samples, probabilities.numel())

                if num_to_sample > 0:
                    sampled_indices_of_candidates = torch.multinomial(probabilities, num_to_sample, replacement=False)
                    target_indices = candidate_indices[sampled_indices_of_candidates]
                else:
                    target_indices = torch.empty(0, dtype=torch.int64, device=device)
            else:
                # Use all points that passed the threshold (or if no candidates).
                target_indices = candidate_indices

        similarity_scores = soft_correspondence_scores[target_indices]
        matched_indices = [
            torch.empty(0, dtype=torch.int64, device=device),
            target_indices,
        ]

        return matched_indices, similarity_scores, soft_sim_map

    @staticmethod
    def _perform_soft_matching(
        mask: torch.Tensor,
        reg: float = 0.1,
        use_rff: bool = False,
        similarity_map: torch.Tensor | None = None,
        ref_features: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
        rff_dim: int = 256,
        rff_sigma: float = 0.75,
        score_threshold: float = 0.4,
        bidirectional: bool = True,
        use_sampling: bool = False,
        use_spatial_sampling: bool = False,
        num_samples: int = 40,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        """Perform matching using the Softmatcher algorithm.

        This method orchestrates the soft matching process by calling helper functions.
        It is an efficient approximation of an Optimal Transport (OT) approach,
        particularly in the large regularization limit. It computes a soft correspondence
        score for each target feature by taking the geometric mean of softmaxed cosine
        similarity maps over all reference features. This avoids the iterative nature of
        solvers like Sinkhorn and has a quadratic complexity in the number of patches.

        This approach is loosely based on the findings in:
        https://www.ijcai.org/proceedings/2024/1000.pdf

        If use_rff is True, this implementation approximates the reference-target feature similarities
        using Random Fourier Features (RFF), effectively changing the similarity metric from
        cosine similarity to a Gaussian kernel.

        The correspondence scores are normalized to a [0, 1] range, and points with a score
        above `score_threshold` are selected. This makes the selection robust to the `reg`
        parameter and score distribution.

        If `bidirectional` is True, this performs a more robust soft matching. It combines
        a forward score (how well a masked reference feature matches a target) with a
        backward score (how well that target feature matches back to one of the masked
        reference features, compared to all other reference features). This helps to
        filter out ambiguous matches.

        Args:
            mask: torch.Tensor - Mask [num_ref_features]
            reg: float - Entropy regularization parameter for Sinkhorn algorithm.
            use_rff: bool - Whether to use RFF approximation.
            similarity_map: torch.Tensor | None - Similarity matrix [num_ref_features, num_target_features].
                Used when use_rff is False.
            ref_features: torch.Tensor | None - Reference features [num_ref_features, feature_dim].
                Used when use_rff is True.
            target_features: torch.Tensor | None - Target features [num_target_features, feature_dim].
                Used when use_rff is True.
            rff_dim: int - Dimension of the RFF projection.
            rff_sigma: float - Width of the Gaussian kernel for RFF.
            score_threshold: float - Threshold for selecting points based on normalized soft correspondence scores.
            bidirectional: bool - Whether to use bidirectional softmatching.
            use_sampling: bool - Whether to use sampling instead of thresholding to select points.
            use_spatial_sampling: bool - Whether to use spatial sampling (NMS) to select points.
            num_samples: int - Number of points to sample if use_sampling is True.

        Returns:
            tuple containing:
                matched_indices: list - Indices of matched foreground points [ref_indices, target_indices]
                  after bidirectional filtering.
                similarity_scores: torch.Tensor - Similarity scores of matched foreground points.
                soft_sim_map: torch.Tensor - Soft similarity map, for visualization.

        Raises:
            ValueError: If full similarity map is required for bidirectional matching.
        """
        device = similarity_map.device
        dtype = similarity_map.dtype

        empty_indices: list[torch.Tensor] = [torch.empty(0, dtype=torch.int64, device=device)] * 2
        empty_scores = torch.empty(0, dtype=dtype, device=device)
        empty_sim_map = torch.empty(0, device=device)

        masked_ref_indices = mask.flatten().nonzero(as_tuple=True)[0]
        if masked_ref_indices.numel() == 0:
            return empty_indices, empty_scores, empty_sim_map

        forward_sim, similarity_map_for_backward = SoftmatcherPromptGenerator._calculate_similarity(
            use_rff=use_rff,
            masked_ref_indices=masked_ref_indices,
            ref_features=ref_features,
            target_features=target_features,
            similarity_map=similarity_map,
            bidirectional=bidirectional,
            rff_dim=rff_dim,
            rff_sigma=rff_sigma,
        )

        if forward_sim is None or forward_sim.numel() == 0:
            return empty_indices, empty_scores, empty_sim_map

        if bidirectional:
            if similarity_map_for_backward is None:
                msg = "Full similarity map is required for bidirectional matching."
                raise ValueError(msg)
            log_soft_correspondence = SoftmatcherPromptGenerator._bidirectional_soft_matching(
                forward_sim=forward_sim,
                similarity_map_for_backward=similarity_map_for_backward,
                masked_ref_indices=masked_ref_indices,
                reg=reg,
            )
        else:
            log_soft_correspondence = SoftmatcherPromptGenerator._unidirectional_soft_matching(
                forward_sim=forward_sim,
                reg=reg,
            )

        soft_correspondence_scores = torch.exp(log_soft_correspondence)

        return SoftmatcherPromptGenerator._process_scores_and_select_points(
            soft_correspondence_scores=soft_correspondence_scores,
            score_threshold=score_threshold,
            device=device,
            use_sampling=use_sampling,
            use_spatial_sampling=use_spatial_sampling,
            num_samples=num_samples,
        )

    @staticmethod
    def _apply_rff(features: torch.Tensor, projection_matrix: torch.Tensor) -> torch.Tensor:
        """Apply Random Fourier Features projection to features to approximate a Gaussian kernel.

        Args:
            features: torch.Tensor - Input features to project.
            projection_matrix: torch.Tensor - Random projection matrix.

        Returns:
            torch.Tensor - Projected features.
        """
        proj = features @ projection_matrix
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1) / (projection_matrix.shape[1] ** 0.5)

    def forward(
        self,
        ref_embeddings: torch.Tensor,
        masked_ref_embeddings: dict[int, torch.Tensor],
        flatten_ref_masks: dict[int, torch.Tensor],
        target_embeddings: torch.Tensor,
        original_sizes: list[tuple[int, int]],
    ) -> tuple[list[dict[int, torch.Tensor]], list[dict[int, torch.Tensor]]]:
        """This generates prompt candidates (or priors) based on the similarities.

        This is done between the reference and target images.

        It uses soft matching to create prompts for the segmenter.
        This Prompt Generator computes the similarity map internally.

        Args:
            ref_embeddings(torch.Tensor): Reference embeddings
            masked_ref_embeddings(dict[int, torch.Tensor]): Dictionary of masked reference embeddings
            flatten_ref_masks(dict[int, torch.Tensor]): Dictionary of flattened reference masks
            target_embeddings(torch.Tensor): Target embeddings
            original_sizes(list[tuple[int, int]]): Original sizes of the target images

        Returns:
            point_prompts(list[dict[int, torch.Tensor]]):
                List of point prompts (with class_id as key and points as value)
            similarities_per_images(list[dict[int, torch.Tensor]]): List of similarities dictionaries
        """
        point_prompts: list[dict[int, torch.Tensor]] = []
        similarities_per_image: list[dict[int, torch.Tensor]] = []

        # this basically makes a vertical stack + flatten
        flattened_ref_embeds = ref_embeddings.reshape(-1, ref_embeddings.shape[-1])

        for target_embed, original_size in zip(target_embeddings, original_sizes, strict=False):
            class_point_prompts: dict[int, torch.Tensor] = {}
            similarities: dict[int, list[torch.Tensor]] = defaultdict(list)
            similarity_map = flattened_ref_embeds @ target_embed.T
            h, w = original_size

            for class_id, flatten_ref_mask in flatten_ref_masks.items():
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
                foreground_indices, foreground_scores, soft_sim_map = self._perform_soft_matching(
                    mask=flatten_ref_mask,
                    similarity_map=similarity_map,
                    use_rff=self.approximate_matching,
                    ref_features=flattened_ref_embeds,
                    target_features=target_embed,
                    score_threshold=self.softmatching_score_threshold,
                    bidirectional=self.softmatching_bidirectional,
                    use_sampling=self.use_sampling,
                    use_spatial_sampling=self.use_spatial_sampling,
                    num_samples=self.num_foreground_points,
                )

                # add the soft map to the similarities
                # this increases the mask filtering technique based on average similarity.
                soft_sim_map = functional.interpolate(
                    soft_sim_map,
                    size=similarities[class_id][0].shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                soft_sim_map = (soft_sim_map - soft_sim_map.min()) / (soft_sim_map.max() - soft_sim_map.min() + 1e-6)
                similarities[class_id].append(soft_sim_map)

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

            # Concatenate all tensors once per class
            concatenated_similarities = {
                class_id: torch.cat(tensor_list, dim=0) for class_id, tensor_list in similarities.items()
            }
            similarities_per_image.append(concatenated_similarities)
        return point_prompts, similarities_per_image
