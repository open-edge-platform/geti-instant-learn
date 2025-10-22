# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SoftMatcher model."""

from typing import TYPE_CHECKING

from getiprompt.components.prompt_generators import SoftmatcherPromptGenerator
from getiprompt.models import Matcher
from getiprompt.utils.constants import SAMModelName

if TYPE_CHECKING:
    from getiprompt.components.prompt_generators import PromptGenerator


class SoftMatcher(Matcher):
    """This is the SoftMatcher model.

    Instead of using a linear sum assignment, this model uses a soft matching algorithm to generate prompts
    for the segmenter.

    This model is based on the paper:
    "Probabilistic Feature Matching for Fast Scalable Visual Prompting"
    https://www.ijcai.org/proceedings/2024/1000.pdf

    Main novelties:
    - Replaces the bidirectional prompt generator with a soft matching algorithm, for very fast computation
    - Can use Random Fourier Features to approximate the similarity map to increase prompt generation speed

    We have added several sampling techniques to increase the performance of the model.

    Examples:
        >>> from getiprompt.models.softmatcher import SoftMatcher
        >>> from getiprompt.types import Image, Priors, Results
        >>> import torch
        >>> import numpy as np
        >>>
        >>> soft_matcher = SoftMatcher()
        >>>
        >>> # Create mock inputs
        >>> ref_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> target_image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        >>> ref_priors = Priors()
        >>> ref_priors.masks.add(torch.ones(30, 30, dtype=torch.bool), class_id=1)
        >>>
        >>> # Run learn and infer
        >>> learn_results = soft_matcher.learn([Image(ref_image)], [ref_priors])
        >>> infer_results = soft_matcher.infer([Image(target_image)])
        >>>
        >>> isinstance(learn_results, Results) and isinstance(infer_results, Results)
        True
        >>> infer_results.masks is not None
        True
        >>> infer_results.annotations is not None
        True
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        mask_similarity_threshold: float | None = 0.42,
        use_sampling: bool = False,
        use_spatial_sampling: bool = False,
        approximate_matching: bool = False,
        softmatching_score_threshold: float = 0.4,
        softmatching_bidirectional: bool = False,
        encoder_model: str = "dinov3_large",
        precision: str = "bf16",
        compile_models: bool = False,
        benchmark_inference_speed: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize the SoftMatcher model.

        Args:
            sam: The name of the SAM model to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            mask_similarity_threshold: The similarity threshold for the mask.
            use_sampling: Whether to use sampling.
            use_spatial_sampling: Whether to use spatial sampling.
            approximate_matching: Whether to use approximate matching.
            softmatching_score_threshold: The score threshold for the soft matching.
            softmatching_bidirectional: Whether to use bidirectional soft matching.
            encoder_model: The encoder model to use.
            precision: The precision to use for the model.
            compile_models: Whether to compile the models.
            benchmark_inference_speed: Whether to benchmark the inference speed.
            device: The device to use for the model.
        """
        super().__init__(
            sam=sam,
            num_foreground_points=num_foreground_points,
            num_background_points=num_background_points,
            mask_similarity_threshold=mask_similarity_threshold,
            encoder_model=encoder_model,
            precision=precision,
            compile_models=compile_models,
            benchmark_inference_speed=benchmark_inference_speed,
            device=device,
        )
        self.prompt_generator: PromptGenerator = SoftmatcherPromptGenerator(
            encoder_input_size=self.encoder.input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
            num_foreground_points=num_foreground_points,
            use_sampling=use_sampling,
            use_spatial_sampling=use_spatial_sampling,
            approximate_matching=approximate_matching,
            softmatching_score_threshold=softmatching_score_threshold,
            softmatching_bidirectional=softmatching_bidirectional,
        )
