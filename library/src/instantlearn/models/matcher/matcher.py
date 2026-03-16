# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Matcher model, based on the paper 'Segment Anything with One Shot Using All-Purpose Feature Matching'."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional

from instantlearn.components.encoders import ImageEncoder
from instantlearn.components.feature_extractors import MaskedFeatureExtractor, ReferenceFeatures
from instantlearn.components.sam import SamDecoder, load_sam_model
from instantlearn.data.base.batch import Batch, Collatable
from instantlearn.data.base.sample import Sample
from instantlearn.models.base import Model
from instantlearn.utils.constants import Backend, SAMModelName

from .prompt_generators import BidirectionalPromptGenerator


class EncoderForwardFeaturesWrapper(nn.Module):
    """Wrapper for image encoder to expose forward_features method for export."""

    IMAGENET_DEFAULT_MEAN = torch.tensor((0.485, 0.456, 0.406))
    IMAGENET_DEFAULT_STD = torch.tensor((0.229, 0.224, 0.225))

    def __init__(
        self,
        encoder: nn.Module,
        ignore_token_length: int,
        input_size: int = 512,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.ignore_token_length = ignore_token_length
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to get encoder features."""
        imagenet_mean = self.IMAGENET_DEFAULT_MEAN.to(device=x.device, dtype=x.dtype)
        imagenet_std = self.IMAGENET_DEFAULT_STD.to(device=x.device, dtype=x.dtype)
        x = x.float() / 255.0
        x = functional.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear")
        x = (x - imagenet_mean[None, :, None, None]) / imagenet_std[None, :, None, None]
        features = self.encoder.forward_features(x)
        features = features[:, self.ignore_token_length :, :]  # ignore CLS and other tokens
        return functional.normalize(features, p=2, dim=-1)


class MatcherInferenceGraph(nn.Module):
    """Traceable inference graph with frozen reference features for ONNX export."""

    def __init__(
        self,
        encoder: nn.Module,
        prompt_generator: BidirectionalPromptGenerator,
        sam_decoder: SamDecoder,
        ref_features: ReferenceFeatures,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.prompt_generator = prompt_generator
        self.sam_decoder = sam_decoder

        # Freeze reference features as model constants
        self.register_buffer("ref_embeddings", ref_features.ref_embeddings)
        self.register_buffer("masked_ref_embeddings", ref_features.masked_ref_embeddings)
        self.register_buffer("flatten_ref_masks", ref_features.flatten_ref_masks)
        self.register_buffer("category_ids", torch.tensor(ref_features.category_ids, device=ref_features.device))

    def forward(self, target_image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single image forward pass for export: target_image [1, 3, H, W] → (masks, scores, labels)."""
        # Encode target [1, num_patches, embed_dim]
        target_embeddings = self.encoder(target_image)
        feature_device = target_embeddings.device

        # Align frozen reference tensors to target embedding device for trace-time safety.
        # This prevents mixed-device matmul when model buffers and encoder output diverge.
        ref_embeddings = self.ref_embeddings.to(feature_device)
        masked_ref_embeddings = self.masked_ref_embeddings.to(feature_device)
        flatten_ref_masks = self.flatten_ref_masks.to(feature_device)
        category_ids = self.category_ids.to(feature_device)

        # Get original size from input tensor [1, 3, H, W] using public APIs only.
        # scalar_tensor preserves dynamic shape in export without relying on private/legacy ONNX helpers.
        height = torch.scalar_tensor(target_image.shape[2], dtype=torch.long, device=feature_device)
        width = torch.scalar_tensor(target_image.shape[3], dtype=torch.long, device=feature_device)
        original_sizes = torch.stack([height, width], dim=0).unsqueeze(0)

        # Generate prompts using frozen ref_features
        # point_prompts: [1, C, max_points, 4], num_points: [1, C], similarities: [1, C, feat_size, feat_size]
        point_prompts, similarities = self.prompt_generator.forward(
            ref_embeddings,
            masked_ref_embeddings,
            flatten_ref_masks,
            category_ids,
            target_embeddings,
            original_sizes,
        )

        # Decode using export-friendly method (single image, returns tensors)
        return self.sam_decoder.forward_export(
            target_image[0],  # Single image [3, H, W]
            category_ids,
            point_prompts[0],  # [C, max_points, 4]
            similarities[0],  # [C, feat_size, feat_size]
        )


class Matcher(Model):
    """Matcher model for one-shot segmentation.

    Based on "[ICLR'24] Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching"
    https://arxiv.org/abs/2305.13310.

    The pipeline is fully traceable (ONNX/TorchScript compatible):
    - Encoder → MaskedFeatureExtractor → BidirectionalPromptGenerator → TraceableSamDecoder

    Examples:
        >>> from instantlearn.models import Matcher
        >>> from instantlearn.data.base import Batch
        >>> from instantlearn.data.base.sample import Sample
        >>> import torch
        >>> import numpy as np

        >>> matcher = Matcher()

        >>> # Create reference sample
        >>> ref_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     masks=torch.ones(30, 30, dtype=torch.bool).unsqueeze(0),
        ...     category_ids=np.array([1]),
        ...     is_reference=[True],
        ...     categories=["object"],
        ... )
        >>> ref_batch = Batch.collate([ref_sample])

        >>> # Create target sample
        >>> target_sample = Sample(
        ...     image=torch.zeros((3, 1024, 1024)),
        ...     is_reference=[False],
        ...     categories=["object"],
        ... )
        >>> target_batch = Batch.collate([target_sample])

        >>> # Run fit and predict
        >>> matcher.fit(ref_batch)
        >>> predict_results = matcher.predict(target_batch)

        >>> isinstance(predict_results, Results)
        True

        >>> predict_results.masks is not None
        True
    """

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        encoder_model: str = "dinov3_large",
        confidence_threshold: float | None = 0.38,
        use_mask_refinement: bool = True,
        use_nms: bool = True,
        precision: str = "bf16",
        compile_models: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize the Matcher model.

        Args:
            sam: SAM model variant to use.
            num_foreground_points: Maximum foreground points per category.
            num_background_points: Background points per category.
            encoder_model: Image encoder model ID.
            confidence_threshold: Minimum confidence score for keeping predicted masks
                                 in the final output. Higher values = stricter filtering, fewer masks.
            use_mask_refinement: Whether to use 2-stage mask refinement with box prompts.
            use_nms: Whether to use NMS in SamDecoder.
            precision: Model precision ("bf16", "fp32").
            compile_models: Whether to compile models with torch.compile.
            device: Device for inference.
        """
        super().__init__()
        # SAM predictor
        self.sam_predictor = load_sam_model(
            sam,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        # Image encoder
        self.encoder = ImageEncoder(
            model_id=encoder_model,
            backend=Backend.TIMM,
            device=device,
            precision=precision,
            compile_models=compile_models,
        )

        # Feature extractor
        self.masked_feature_extractor = MaskedFeatureExtractor(
            input_size=self.encoder.input_size,
            patch_size=self.encoder.patch_size,
            device=device,
        )

        # Prompt generator (includes filtering)
        self.prompt_generator = BidirectionalPromptGenerator(
            encoder_input_size=self.encoder.input_size,
            encoder_patch_size=self.encoder.patch_size,
            encoder_feature_size=self.encoder.feature_size,
            num_foreground_points=num_foreground_points,
            num_background_points=num_background_points,
        )

        # SAM decoder
        self.segmenter = SamDecoder(
            sam_predictor=self.sam_predictor,
            confidence_threshold=confidence_threshold,
            use_mask_refinement=use_mask_refinement,
            use_nms=use_nms,
        )

        # Reference features (set during fit)
        self.ref_features: ReferenceFeatures | None = None

    def fit(self, reference: Sample | list[Sample] | Batch) -> ReferenceFeatures:
        """Learn from reference images.

        Args:
            reference: Reference data to learn from. Accepts:
                - Sample: A single reference sample
                - list[Sample]: A list of reference samples
                - Batch: A batch of reference samples
        """
        reference_batch = Batch.collate(reference)
        ref_embeddings = self.encoder(images=reference_batch.images)
        self.ref_features = self.masked_feature_extractor(
            ref_embeddings,
            reference_batch.masks,
            reference_batch.category_ids,
        )
        return self.ref_features

    def predict(self, target: Collatable) -> list[dict[str, torch.Tensor]]:
        """Predict masks for target images.

        Args:
            target: Target data to infer. Accepts:
                - Sample: A single target sample
                - list[Sample]: A list of target samples
                - Batch: A batch of target samples
                - str | Path: A single image path
                - list[str] | list[Path]: Multiple image paths

        Returns:
            List of predictions per image, each containing:
                "pred_masks": [num_masks, H, W]
                "pred_scores": [num_masks]
                "pred_labels": [num_masks] - category IDs

        Raises:
            RuntimeError: If fit() has not been called before predict().
        """
        target_batch = Batch.collate(target)
        if self.ref_features is None:
            msg = "No reference features. Call fit() first."
            raise RuntimeError(msg)

        # Get original sizes [T, 2]
        original_sizes = torch.tensor(
            [image.size()[-2:] for image in target_batch.images],
            device=self.ref_features.device,
        )

        # Encode all targets [T, num_patches, embed_dim]
        target_embeddings = self.encoder(images=target_batch.images)

        # Generate prompts [T, C, max_points, 4], [T, C], [T, C, feat_size, feat_size]
        point_prompts, similarities = self.prompt_generator(
            self.ref_features.ref_embeddings,
            self.ref_features.masked_ref_embeddings,
            self.ref_features.flatten_ref_masks,
            self.ref_features.category_ids,
            target_embeddings,
            original_sizes,
        )

        # Decode masks for all images
        return self.segmenter(
            target_batch.images,
            self.ref_features.category_ids,
            point_prompts=point_prompts,
            similarities=similarities,
        )

    @staticmethod
    def _fix_onnx_output_names(onnx_path: Path, expected_names: list[str]) -> None:
        """Ensure ONNX graph outputs have the expected names.

        Registered buffers returned as outputs often get auto-generated names
        (e.g. '39982') because the ONNX tracer treats them as graph constants.
        Renames outputs in-place using the ONNX protobuf, also updating all
        internal node references and initializers so the graph stays valid.
        """
        if not onnx_path.exists():
            return

        import onnx

        model = onnx.load(str(onnx_path))
        rename_map: dict[str, str] = {}
        for output, expected in zip(model.graph.output, expected_names, strict=False):
            if output.name != expected:
                rename_map[output.name] = expected
        if not rename_map:
            return
        # Update node outputs that feed into graph outputs.
        for node in model.graph.node:
            for i, name in enumerate(node.output):
                if name in rename_map:
                    node.output[i] = rename_map[name]
        # Update initializers (registered buffers appear here).
        for initializer in model.graph.initializer:
            if initializer.name in rename_map:
                initializer.name = rename_map[initializer.name]
        # Update the graph output names.
        for output in model.graph.output:
            if output.name in rename_map:
                output.name = rename_map[output.name]
        onnx.save(model, str(onnx_path))

    @torch.no_grad()
    def export(
        self,
        export_dir: str | Path = Path("./exports/matcher"),
        backend: str | Backend = Backend.ONNX,
        compress_to_fp16: bool = False,
    ) -> Path:
        """Export model components.

        Args:
            export_dir: Directory to save exported models.
            backend: Export backend (ONNX, OpenVINO).
            compress_to_fp16: Whether to compress OpenVINO model to FP16.

        Returns:
            Path to export directory.

        Raises:
            ImportError: If OpenVINO is selected but not installed.
            RuntimeError: If fit() has not been called before predict().
        """
        if self.ref_features is None:
            msg = "No reference features. Call fit() first."
            raise RuntimeError(msg)

        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        export_device = self.ref_features.device
        if backend == Backend.OPENVINO:
            export_device = torch.device("cpu")
        first_encoder_param = next(iter(self.encoder._model.model.parameters()), None)
        if backend != Backend.OPENVINO and isinstance(first_encoder_param, torch.Tensor):
            export_device = first_encoder_param.device

        self.sam_predictor.sync_device(export_device, dtype=torch.float32)
        self.segmenter.device = self.sam_predictor.device
        ref_features = self.ref_features.to(export_device)

        matcher = (
            MatcherInferenceGraph(
                encoder=EncoderForwardFeaturesWrapper(
                    self.encoder._model.model,
                    ignore_token_length=self.encoder._model.ignore_token_length,
                ),
                prompt_generator=self.prompt_generator,
                sam_decoder=self.segmenter,
                ref_features=ref_features,
            )
            .to(export_device)
            .float()
        )  # Force FP32 for stable CPU tracing

        input_size = self.encoder.input_size
        target_image = torch.randn(1, 3, input_size, input_size, device=export_device)
        if backend == Backend.ONNX:
            onnx_path = export_path / "matcher.onnx"
            torch.onnx.export(
                matcher,
                args=(target_image,),
                f=onnx_path,
                input_names=["target_image"],
                output_names=["masks", "scores", "labels"],
                dynamic_axes={
                    "target_image": {2: "height", 3: "width"},
                    "masks": {0: "num_masks", 1: "height", 2: "width"},
                    "scores": {0: "num_masks"},
                    "labels": {0: "num_masks"},
                },
                dynamo=False,
            )
            self._fix_onnx_output_names(onnx_path, ["masks", "scores", "labels"])
            return onnx_path

        if backend == Backend.OPENVINO:
            try:
                import openvino

                # Export to ONNX first, then convert to OpenVINO.
                # Direct PyTorch → OpenVINO conversion fails on many ops (aten::pad, aten::unbind, etc.)
                # ONNX → OpenVINO conversion has much better support.
                onnx_path = export_path / "matcher.onnx"
                torch.onnx.export(
                    matcher,
                    args=(target_image,),
                    f=onnx_path,
                    input_names=["target_image"],
                    output_names=["masks", "scores", "labels"],
                    # Keep OpenVINO export graph static for stable GPU shape inference.
                    # Dynamic axes here can lead to infer-time broadcast mismatches.
                    dynamo=False,
                )
                # Prefer ONNX frontend path for better operator coverage.
                # Fall back to direct conversion when ONNX export output is unavailable.
                core = openvino.Core()
                if onnx_path.exists():
                    try:
                        ov_model = core.read_model(str(onnx_path))
                    except RuntimeError:
                        ov_model = openvino.convert_model(matcher, example_input=target_image)
                else:
                    ov_model = openvino.convert_model(matcher, example_input=target_image)

                # Fix output names: registered buffers returned as model outputs
                # get auto-generated names (e.g. '39982') from the ONNX tracer.
                expected_names = ["masks", "scores", "labels"]
                for output, name in zip(ov_model.outputs, expected_names, strict=False):
                    output.tensor.set_names({name})

                # Reshape to static input for optimal GPU kernel compilation.
                input_name = ov_model.inputs[0].get_any_name()
                ov_model.reshape({input_name: [1, 3, input_size, input_size]})

                openvino.save_model(
                    ov_model,
                    export_path / "matcher.xml",
                    compress_to_fp16=compress_to_fp16,
                )
                return export_path / "matcher.xml"
            except ImportError as e:
                msg = "OpenVINO is not installed. Please install it to use OpenVINO export."
                raise ImportError(msg) from e

        return export_path
