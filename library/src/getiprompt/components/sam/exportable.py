# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM exportable model for ONNX/OpenVINO export.

This module provides utilities to export SAM models from PyTorch to
ONNX and OpenVINO IR formats. It handles ONNX-incompatible operations
and ensures the exported models can be used with OpenVINOSAMPredictor.
"""

import openvino
import torch
from segment_anything_hq.modeling.prompt_encoder import PromptEncoder
from segment_anything_hq.predictor import SamPredictor
from torch import nn



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
        self.set_prompt_encoder()

        for p in self.sam_predictor.model.mask_decoder.parameters():
            p.requires_grad_(False)

        for p in self.sam_predictor.model.prompt_encoder.parameters():
            p.requires_grad_(False)

        for p in self.sam_predictor.model.image_encoder.parameters():
            p.requires_grad_(False)

    def set_prompt_encoder(self) -> PromptEncoder:
        """Set the prompt encoder to the exportable prompt encoder."""
        prompt_encoder = self._ExportablePromptEncoder(
            embed_dim=self.sam_predictor.model.prompt_encoder.embed_dim,
            image_embedding_size=self.sam_predictor.model.prompt_encoder.image_embedding_size,
            input_image_size=self.sam_predictor.model.prompt_encoder.input_image_size,
            mask_in_chans=16,  # It's always 16
        )
        self.sam_predictor.model.prompt_encoder = prompt_encoder

    @torch.no_grad()
    def forward(
        self,
        transformed_image: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        boxes: torch.Tensor,
        mask_input: torch.Tensor,
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(original_size, torch.Tensor):
            original_size = tuple(original_size)

        input_size = tuple(transformed_image.shape[-2:])
        input_image = self.sam_predictor.model.preprocess(transformed_image)
        features, interm_features = self.sam_predictor.model.image_encoder(input_image)
        points = (point_coords, point_labels)

        # Embed prompts
        sparse_embeddings, dense_embeddings = self.sam_predictor.model.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=mask_input,
        )

        low_res_masks, iou_predictions = self.sam_predictor.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.sam_predictor.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            hq_token_only=False,
            interm_embeddings=interm_features,
        )

        # Upscale the masks to the original image resolution
        masks = self.sam_predictor.model.postprocess_masks(low_res_masks, input_size, original_size)
        masks = masks > self.sam_predictor.model.mask_threshold
        return masks, iou_predictions, low_res_masks

    def export(self, output_path: str) -> None:
        transformed_image = torch.zeros((1, 3, 1024, 1024), dtype=torch.float32)
        point_coords = torch.rand((1, 3, 2), dtype=torch.float32)
        point_labels = torch.randint(0, 2, (1, 3), dtype=torch.int32)
        boxes = torch.rand((1, 4), dtype=torch.float32)
        mask_input = torch.rand((1, 1, 256, 256), dtype=torch.float32)
        original_size = torch.tensor((1024, 1024), dtype=torch.int32)

        # Define input and output names
        input_names = [
            "transformed_image",
            "point_coords",
            "point_labels",
            "boxes",
            "mask_input",
            "original_size",
        ]
        output_names = ["masks", "iou_predictions", "low_res_logits"]

        # Define dynamic axes
        dynamic_axes = {
            "transformed_image": {0: "batch_size"},
            "point_coords": {0: "num_masks"},
            "point_labels": {0: "num_masks"},
            "boxes": {0: "num_masks"},
            "mask_input": {0: "num_masks"},
        }

        with torch.no_grad():
            try:
                # Export to ONNX
                torch.onnx.export(
                    self,
                    (transformed_image, point_coords, point_labels, boxes, mask_input, original_size),
                    output_path / "exported_sam.onnx",
                    opset_version=20,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    verbose=True,
                )
            except Exception as e:
                print(f"Error exporting to ONNX: {e}")
                raise e

        exported_model = openvino.convert_model(output_path / "exported_sam.onnx")
        for i, name in enumerate(output_names):
            traced_names = exported_model.outputs[i].get_names()
            name_found = False
            for traced_name in traced_names:
                if name in traced_name:
                    name_found = True
                    break
            name_found = name_found and bool(len(traced_names))

            if not name_found:
                msg = (
                    f"{name} is not matched with the converted model's traced output names: {traced_names}."
                    " Please check output_names argument of the exporter's constructor."
                )
                raise ValueError(msg)
            exported_model.outputs[i].tensor.set_names({name})

        for i, name in enumerate(input_names):
            traced_names = exported_model.inputs[i].get_names()
            name_found = False
            for traced_name in traced_names:
                if name in traced_name:
                    name_found = True
                    break
            name_found = name_found and bool(len(traced_names))

            if not name_found:
                msg = (
                    f"{name} is not matched with the converted model's traced input names: {traced_names}."
                    " Please check input_names argument of the exporter's constructor."
                )
                raise ValueError(msg)

            exported_model.inputs[i].tensor.set_names({name})
        openvino.save_model(exported_model, output_path / "exported_sam.xml")
