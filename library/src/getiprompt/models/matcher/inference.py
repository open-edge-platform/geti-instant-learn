# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO-based Matcher model for inference with exported models."""

from pathlib import Path

import torch

from getiprompt.components import SamDecoder
from getiprompt.components.encoders import OpenVINOImageEncoder
from getiprompt.components.feature_extractors import MaskedFeatureExtractor
from getiprompt.components.filters import PointPromptFilter
from getiprompt.components.prompt_generators import BidirectionalPromptGenerator
from getiprompt.components.sam.openvino import OpenVINOSAMPredictor
from getiprompt.utils.constants import Backend, SAMModelName

from .matcher import Matcher


class InferenceModel(Matcher):
    """OpenVINO-based Matcher model for efficient inference.

    This class inherits from Matcher but uses OpenVINO backend for all components,
    loading from pre-exported model files. This provides faster inference on Intel
    hardware (CPU, GPU, VPU) compared to PyTorch backend.

    The model folder should contain exported models:
        - image_encoder.xml (and .bin): The DINO image encoder
        - exported_sam.xml (and .bin): The SAM model files

    Examples:
        >>> from getiprompt.models import OVMatcher
        >>> from getiprompt.data.base import Batch
        >>> from getiprompt.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> # First export a regular Matcher model
        >>> from getiprompt.models import Matcher
        >>> matcher = Matcher()
        >>> matcher.export(export_dir="./exports/matcher", backend=Backend.OPENVINO)

        >>> # Then load with OVMatcher for inference
        >>> ov_matcher = OVMatcher(
        ...     model_folder="./exports/matcher",
        ...     sam=SAMModelName.SAM_HQ_TINY,
        ...     device="CPU"
        ... )

        >>> # Create mock inputs
        >>> ref_image = torch.zeros((3, 1024, 1024))
        >>> target_image = torch.zeros((3, 1024, 1024))
        >>> ref_mask = torch.ones(30, 30, dtype=torch.bool)

        >>> # Create reference sample
        >>> ref_sample = Sample(
        ...     image=ref_image,
        ...     masks=ref_mask.unsqueeze(0),
        ...     category_ids=np.array([1]),
        ...     is_reference=[True],
        ...     categories=["object"],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=target_image,
        ...     is_reference=[False],
        ...     categories=["object"],
        ... )
        >>> target_batch = Batch.collate([target_sample])

        >>> # Run fit and predict
        >>> ov_matcher.fit(ref_batch)
        >>> infer_results = ov_matcher.predict(target_batch)
    """

    def __init__(
        self,
        model_folder: str | Path,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        mask_similarity_threshold: float | None = 0.38,
        device: str = "cpu",
        precision: str = "fp32",
    ) -> None:
        """Initialize the OVMatcher model with OpenVINO backend.

        Args:
            model_folder: Path to the folder containing exported models.
                Should contain:
                - image_encoder.xml: Exported DINO encoder
                - exported_sam.xml: Exported SAM model
            sam: The name of the SAM model to use.
            num_foreground_points: The number of foreground points to use.
            num_background_points: The number of background points to use.
            mask_similarity_threshold: The similarity threshold for the mask.
            device: The OpenVINO device to use ("CPU", "GPU", "AUTO").

        Raises:
            FileNotFoundError: If required model files are not found in model_folder.
        """
        # Don't call parent __init__ as we need to replace components
        # Instead, initialize from torch.nn.Module directly
        torch.nn.Module.__init__(self)

        model_folder = Path(model_folder)
        if not model_folder.exists():
            msg = f"Model folder not found: {model_folder}"
            raise FileNotFoundError(msg)

        # Find encoder model file
        encoder_path = model_folder / "image_encoder.xml"
        if not encoder_path.exists():
            msg = (
                f"Image encoder model not found at {encoder_path}. "
                f"Please export the Matcher model first using matcher.export()"
            )
            raise FileNotFoundError(msg)

        # Find SAM model file
        sam_path = model_folder / "exported_sam.xml"
        if not sam_path.exists():
            msg = f"SAM model not found at {sam_path}. Please export the Matcher model first using matcher.export()"
            raise FileNotFoundError(msg)

        # Initialize OpenVINO components
        self.encoder = OpenVINOImageEncoder(
            model_path=encoder_path,
            device=device,
            precision=precision,
        )

        self.sam_predictor = OpenVINOSAMPredictor(
            sam_model_name=sam,
            device=device,
            precision=precision,
            model_path=sam_path,
        )

        # Initialize other components (these don't need backend changes)
        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
        )

        self.prompt_generator = BidirectionalPromptGenerator(
            encoder_input_size=self.encoder.input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_background_points=num_background_points,
        )

        self.prompt_filter = PointPromptFilter(num_foreground_points=num_foreground_points)

        self.segmenter: SamDecoder = SamDecoder(
            sam_predictor=self.sam_predictor,
            target_length=1024,
            mask_similarity_threshold=mask_similarity_threshold,
        )

        # State variables
        self.masked_ref_embeddings = None
        self.ref_masks = None

    def export(
        self,
        export_dir: str | Path = Path("./exports/matcher"),
        backend: Backend = Backend.ONNX,
    ) -> Path:
        """Export is not supported for OVMatcher.

        OVMatcher is designed to load already-exported models. Use the regular
        Matcher class to export models.

        Raises:
            NotImplementedError: Always raised.
        """
        msg = (
            "OVMatcher does not support export as it already uses exported models. "
            "Please use the regular Matcher class to export:\n"
            "  from getiprompt.models import Matcher\n"
            "  matcher = Matcher()\n"
            "  matcher.export(export_dir, backend=Backend.OPENVINO)"
        )
        raise NotImplementedError(msg)
