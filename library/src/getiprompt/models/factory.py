# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Model factory module."""

import logging
from argparse import Namespace

from getiprompt.components.prompt_generators import GroundingModel
from getiprompt.utils.constants import ModelName, SAMModelName

# Lazy import to avoid circular dependencies during module import time.
from .base import Model
from .grounded_sam import GroundedSAM
from .matcher import Matcher
from .per_dino import PerDino
from .soft_matcher import SoftMatcher

logger = logging.getLogger("Geti Prompt")


def load_model(sam: SAMModelName, model_name: ModelName, args: Namespace) -> Model:
    """Instantiate and return the requested model.

    Args:
        sam: The name of the SAM model.
        model_name: The name of the model.
        args: The arguments to the model.

    Returns:
        The instantiated model.
    """
    logger.info("Constructing model: %s", model_name.value)

    match model_name:
        case ModelName.PER_DINO:
            return PerDino(
                sam=sam,
                encoder_model=args.encoder_model,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                num_grid_cells=args.num_grid_cells,
                similarity_threshold=args.similarity_threshold,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case ModelName.MATCHER:
            return Matcher(
                sam=sam,
                encoder_model=args.encoder_model,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                mask_similarity_threshold=args.mask_similarity_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case ModelName.SOFT_MATCHER:
            return SoftMatcher(
                sam=sam,
                encoder_model=args.encoder_model,
                num_foreground_points=args.num_foreground_points,
                num_background_points=args.num_background_points,
                mask_similarity_threshold=args.mask_similarity_threshold,
                use_sampling=args.use_sampling,
                use_spatial_sampling=args.use_spatial_sampling,
                approximate_matching=args.approximate_matching,
                softmatching_score_threshold=args.softmatching_score_threshold,
                softmatching_bidirectional=args.softmatching_bidirectional,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case ModelName.GROUNDED_SAM:
            return GroundedSAM(
                sam=sam,
                grounding_model=GroundingModel(args.grounding_model),
                box_threshold=args.box_threshold,
                text_threshold=args.text_threshold,
                precision=args.precision,
                compile_models=args.compile_models,
                device=args.device,
            )
        case _:
            msg = f"Algorithm {model_name.value} not implemented yet"
            raise NotImplementedError(msg)
