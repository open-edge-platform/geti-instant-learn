# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM exportable model for ONNX/OpenVINO export.

This module provides utilities to export SAM models from PyTorch to
ONNX and OpenVINO IR formats. It handles ONNX-incompatible operations
and ensures the exported models can be used with OpenVINOSAMPredictor.
"""

import logging

import openvino
import torch
from segment_anything_hq.modeling.prompt_encoder import PromptEncoder
from segment_anything_hq.predictor import SamPredictor
from torch import nn

logger = logging.getLogger("Geti Prompt")


class ExportableSAMPredictor(nn.Module):
    """Wrapper for exporting SAM models to ONNX and OpenVINO IR formats.

    This class prepares a SAM predictor for export by:
    1. Replacing ONNX-incompatible operations (e.g., boolean indexing in prompt encoder)
    2. Freezing all parameters
    3. Providing a traced forward pass
    4. Handling ONNX export with proper input/output naming
    5. Converting to OpenVINO IR format

    The exported model performs end-to-end inference from preprocessed image
    to segmentation masks, including image encoding, prompt encoding, and mask decoding.

    Args:
        sam_predictor: PyTorch SAM predictor (segment_anything_hq.SamPredictor)
            to export. This should be the internal predictor from PyTorchSAMPredictor,
            or a standalone SamPredictor instance.

    Note:
        This class is typically used via PyTorchSAMPredictor.export() convenience method,
        but can also be instantiated directly for advanced use cases.

    Example:
        >>> # Method 1: Via convenience method (recommended)
        >>> predictor = load_sam_model(SAMModelName.SAM_HQ_TINY, backend="pytorch")
        >>> ov_path = predictor.export(Path("./exported"))

        >>> # Method 2: Direct instantiation (advanced)
        >>> from getiprompt.components.inference import ExportableSAMPredictor
        >>> exportable = ExportableSAMPredictor(predictor._predictor)
        >>> exportable.export(Path("./exported"))
    """

    class _ExportablePromptEncoder(PromptEncoder):
        def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
            points += 0.5  # Shift to center of pixel
            if pad:
                padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
                padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
                points = torch.cat([points, padding_point], dim=1)
                labels = torch.cat([labels, padding_label], dim=1)

            point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
            # Use ONNX-compatible operations instead of boolean indexing
            # Create masks for each label type
            mask_neg1 = (labels == -1).float().unsqueeze(-1)  # [B, N, 1]
            mask_0 = (labels == 0).float().unsqueeze(-1)  # [B, N, 1]
            mask_1 = (labels == 1).float().unsqueeze(-1)  # [B, N, 1]

            # Apply embeddings using element-wise multiplication
            point_embedding *= 1 - mask_neg1  # Zero out -1 labels
            point_embedding += mask_neg1 * self.not_a_point_embed.weight
            point_embedding += mask_0 * self.point_embeddings[0].weight
            point_embedding += mask_1 * self.point_embeddings[1].weight
            return point_embedding

    def __init__(self, sam_predictor: SamPredictor):
        super().__init__()
        self.sam_predictor = sam_predictor
        self.input_size = sam_predictor.model.image_encoder.img_size
        self._patch_prompt_encoder()

    def _patch_prompt_encoder(self) -> PromptEncoder:
        """Set the prompt encoder to the exportable prompt encoder."""
        prompt_encoder = self._ExportablePromptEncoder(
            embed_dim=self.sam_predictor.model.prompt_encoder.embed_dim,
            image_embedding_size=self.sam_predictor.model.prompt_encoder.image_embedding_size,
            input_image_size=self.sam_predictor.model.prompt_encoder.input_image_size,
            mask_in_chans=16,  # It's always 16
        )

        # need to load patched prompt encoder with the original prompt encoder weights
        prompt_encoder.load_state_dict(self.sam_predictor.model.prompt_encoder.state_dict(), strict=True)
        self.sam_predictor.model.prompt_encoder = prompt_encoder

    def _freeze_modules(self, modules: list[nn.Module]) -> None:
        """Freeze the modules."""
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(False)

    def _validate_and_set_names(
        self,
        model_ports: list,
        expected_names: list[str],
        port_type: str,
        arg_name: str,
    ) -> None:
        """Validate and set names for model inputs or outputs.

        Args:
            model_ports: List of model input or output ports from OpenVINO model.
            expected_names: List of expected names to validate against.
            port_type: Type of port ("input" or "output") for error messages.
            arg_name: Name of the argument ("input_names" or "output_names") for error messages.

        Raises:
            ValueError: If a name is not found in the traced names.
        """
        for i, name in enumerate(expected_names):
            traced_names = model_ports[i].get_names()
            name_found = False
            for traced_name in traced_names:
                if name in traced_name:
                    name_found = True
                    break
            name_found = name_found and bool(len(traced_names))

            if not name_found:
                msg = (
                    f"{name} is not matched with the converted model's traced {port_type} names: {traced_names}."
                    f" Please check {arg_name} argument of the exporter's constructor."
                )
                raise ValueError(msg)
            model_ports[i].tensor.set_names({name})

    @torch.no_grad()
    def forward(
        self,
        transformed_image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        original_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = original_size
        original_size = (h, w)
        self.sam_predictor.set_torch_image(transformed_image, original_size)
        return self.sam_predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
            return_logits=False,
        )

    def export(self, output_path: str) -> None:
        """Export the model to ONNX and OpenVINO IR format.

        Args:
            output_path: The path to export the model to.

        TODO: Add support for boxes and mask_input in prompt encoder.
        """
        self._freeze_modules([
            self.sam_predictor.model.mask_decoder,
            self.sam_predictor.model.prompt_encoder,
            self.sam_predictor.model.image_encoder,
        ])

        dummy_image = torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.float32)
        dummpy_coords = torch.rand((10, 1, 2), dtype=torch.float32) * self.input_size
        dummy_labels = torch.ones((10, 1), dtype=torch.float32)
        original_size = torch.tensor((self.input_size, self.input_size), dtype=torch.int32)

        # Define input and output names
        input_names = [
            "transformed_image",
            "point_coords",
            "point_labels",
            "original_size",
        ]
        output_names = ["masks", "iou_predictions", "low_res_logits"]

        # Define dynamic axes
        dynamic_axes = {
            "transformed_image": {0: "batch_size", 2: "height", 3: "width"},
            "point_coords": {0: "num_masks", 1: "num_points"},
            "point_labels": {0: "num_masks", 1: "num_points"},
        }

        with torch.no_grad():
            try:
                # Export to ONNX
                torch.onnx.export(
                    self,
                    (dummy_image, dummpy_coords, dummy_labels, original_size),
                    output_path / "exported_sam.onnx",
                    opset_version=20,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=True,
                )
            except Exception as e:
                msg = f"Error exporting to ONNX: {e}"
                logger.exception(msg)
                raise e

        exported_model = openvino.convert_model(output_path / "exported_sam.onnx")
        self._validate_and_set_names(
            exported_model.outputs,
            output_names,
            "output",
            "output_names",
        )
        self._validate_and_set_names(
            exported_model.inputs,
            input_names,
            "input",
            "input_names",
        )
        openvino.save_model(exported_model, output_path / "exported_sam.xml")
