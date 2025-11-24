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
from .sam3 import SAM3
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
                confidence_threshold=args.confidence_threshold,
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
                confidence_threshold=args.confidence_threshold,
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
                confidence_threshold=args.confidence_threshold,
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
        case ModelName.SAM3:
            # SAM3 doesn't use the SAM backend parameter - it has its own architecture
            return SAM3(
                bpe_path=getattr(args, "bpe_path", None),
                device=args.device,
                confidence_threshold=getattr(args, "confidence_threshold", 0.5),
                resolution=getattr(args, "resolution", 1008),
                precision=args.precision,
                checkpoint_path=getattr(args, "checkpoint_path", None),
                load_from_HF=getattr(args, "load_from_HF", True),
                enable_segmentation=getattr(args, "enable_segmentation", True),
                enable_inst_interactivity=getattr(args, "enable_inst_interactivity", False),
                compile_models=args.compile_models,
            )
        case _:
            msg = f"Algorithm {model_name.value} not implemented yet"
            raise NotImplementedError(msg)
