# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM MAPI decoder."""

import numpy as np
from model_api.models import SAMLearnableVisualPrompter
from model_api.models.visual_prompting import VisualPromptingFeatures

from getiprompt.processes.segmenters.segmenter_base import Segmenter
from getiprompt.types import Features, Image, Masks, Points


class SamMAPIDecoder(Segmenter):
    """This is a wrapper around the ModelAPI SAM decoder.

    Examples:
        >>> from model_api.models import SAMLearnableVisualPrompter
        >>> from model_api.models.model import Model
        >>> from getiprompt.utils.constants import MAPI_ENCODER_PATH, MAPI_DECODER_PATH
        >>> from getiprompt.processes.segmenters import SamMAPIDecoder
        >>> from getiprompt.types import Features, Image, Masks, Points
        >>> import torch
        >>> import numpy as np
        >>>
        >>> encoder = Model.create_model(MAPI_ENCODER_PATH)
        >>> decoder = Model.create_model(MAPI_DECODER_PATH)
        >>> model = SAMLearnableVisualPrompter(encoder, decoder)
        >>> segmenter = SamMAPIDecoder(model=model)
        >>> image = Image(np.zeros((20, 20, 3), dtype=np.uint8))
        >>> features = Features(global_features=torch.randn(1, 256))
        >>> masks, points = segmenter(images=[image], reference_features=[features])
        >>> isinstance(masks, list) and isinstance(masks[0], Masks) and masks[0].get(0) is not None
        True
        >>> isinstance(points, list) and isinstance(points[0], Points) and points[0].get(0) is not None
        True
    """

    def __init__(
        self,
        model: SAMLearnableVisualPrompter,
    ) -> None:
        super().__init__()
        self.model = model

    def __call__(
        self,
        images: list[Image],
        reference_features: list[Features] | None = None,
    ) -> tuple[list[Masks], list[Points]]:
        """Create masks from priors using SAM.

        Args:
            images: List of target images.
            reference_features: Features from the reference images.

        Returns:
            A tuple of a list of masks, one for each class in each target image,
            and a list of points, one for each class in each target image.
        """
        # Recreate feature object from the _state
        if len(reference_features) != 1:
            msg = "MAPISamDecoder only supports one set of reference features"
            raise ValueError(
                msg,
            )
        reference = VisualPromptingFeatures(
            reference_features[0].global_features.numpy(),
            used_indices=np.array([0]),
        )

        masks_per_image: list[Masks] = []
        points_per_image: list[Points] = []

        for image in images:
            # Get results an stack into a single np array
            result = self.model.infer(
                image.to_numpy(),
                reference_features=reference,
                apply_masks_refinement=False,
            )
            masks = Masks()
            points = Points()

            if 0 in result.data:
                mask = result.get_mask(0).mask
                mask = np.stack(mask)
                for m in mask:
                    masks.add(m, class_id=0)

                # Convert output into Points and Priors
                ps = np.stack(result.data[0].points)
                scores = result.data[0].scores
                # Generate x, y, score, label
                # Note that Model API does not return the used background points
                points_scores = np.ones([len(ps), 4])
                points_scores[:, 0] = ps[:, 0]
                points_scores[:, 1] = ps[:, 1]
                points_scores[:, 2] = scores
                points.add(points_scores, class_id=0)

            # Add to return list
            points_per_image.append(points)
            masks_per_image.append(masks)

        return masks_per_image, points_per_image
