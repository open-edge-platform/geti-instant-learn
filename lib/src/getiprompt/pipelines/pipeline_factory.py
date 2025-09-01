# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pipeline factory module."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from getiprompt.utils.constants import PipelineName, SAMModelName

if TYPE_CHECKING:
    from argparse import Namespace

    from getiprompt.pipelines.pipeline_base import Pipeline

logger = logging.getLogger("Vision Prompt")


def load_pipeline(sam: SAMModelName, pipeline_name: PipelineName, args: Namespace) -> Pipeline:
    """Instantiate and return the requested pipeline.

    Args:
        sam: The name of the SAM model.
        pipeline_name: The name of the pipeline.
        args: The arguments to the pipeline.

    Returns:
        The instantiated pipeline.
    """
    # Lazy import to avoid circular dependencies during module import time.
    from getiprompt.pipelines import GroundingDinoSAM, Matcher, PerDino, PerSam, PerSamMAPI, SoftMatcher

    logger.info("Constructing pipeline: %s", pipeline_name.value)

    match pipeline_name:
        case PipelineName.PER_SAM:
            return PerSam(
                sam=sam,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                num_grid_cells=args.num_grid_cells,
                similarity_threshold=args.similarity_threshold,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
                device=args.device,
            )
        case PipelineName.PER_DINO:
            return PerDino(
                sam=sam,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                num_grid_cells=args.num_grid_cells,
                similarity_threshold=args.similarity_threshold,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
                device=args.device,
            )
        case PipelineName.MATCHER:
            return Matcher(
                sam=sam,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
                device=args.device,
            )
        case PipelineName.PER_SAM_MAPI:
            return PerSamMAPI()
        case PipelineName.SOFT_MATCHER:
            return SoftMatcher(
                sam=sam,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                apply_mask_refinement=args.apply_mask_refinement,
                skip_points_in_existing_masks=args.skip_points_in_existing_masks,
                mask_similarity_threshold=args.mask_similarity_threshold,
                use_sampling=args.use_sampling,
                use_spatial_sampling=args.use_spatial_sampling,
                approximate_matching=args.approximate_matching,
                softmatching_score_threshold=args.softmatching_score_threshold,
                softmatching_bidirectional=args.softmatching_bidirectional,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
                device=args.device,
            )
        case PipelineName.GROUNDING_DINO_SAM:
            return GroundingDinoSAM(
                sam=sam,
                apply_mask_refinement=args.apply_mask_refinement,
                precision=args.precision,
                compile_models=args.compile_models,
                verbose=args.verbose,
                image_size=args.image_size,
                device=args.device,
            )
        case _:
            msg = f"Algorithm {pipeline_name.value} not implemented yet"
            raise NotImplementedError(msg)
