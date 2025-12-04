# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM exportable model for ONNX/OpenVINO export.

This module provides utilities to export SAM models from PyTorch to
ONNX and OpenVINO IR formats. It handles ONNX-incompatible operations
and ensures the exported models can be used with OpenVINOSAMPredictor.
"""

import logging
from pathlib import Path

import openvino
import torch
from segment_anything_hq.modeling.prompt_encoder import PromptEncoder
from segment_anything_hq.predictor import SamPredictor
from torch import nn

from getiprompt.utils.constants import Backend

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

        **Optional Prompts**: The exported model requires boxes and mask_input as inputs
        for ONNX compatibility, but supports "not provided" scenarios using sentinel values:

        - **Boxes**: Pass all-zero boxes (e.g., [[0, 0, 0, 0]]) to indicate "no boxes".
          The prompt encoder detects these and zeros out box embeddings.

        - **Mask Input**: Pass all-zero masks (e.g., zeros((B, 1, 256, 256))) to indicate
          "no mask input". The prompt encoder detects these and uses the default no_mask_embed.

        This allows the same traced model to handle all prompt combinations:
        - Stage 1: Points only (boxes and masks are zeros)
        - Stage 2: Points + boxes + mask refinement (all non-zero)

    Example:
        >>> # Method 1: Via convenience method (recommended)
        >>> predictor = load_sam_model(SAMModelName.SAM_HQ_TINY, backend=Backend.PYTORCH)
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

        def forward(
            self,
            points: tuple[torch.Tensor, torch.Tensor] | None,
            boxes: torch.Tensor | None,
            masks: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """ONNX-traceable forward pass with optional box and mask prompts.

            Uses sentinel values (all-zero tensors) to handle optional inputs:
            - All-zero boxes → skip box embedding (zero out embeddings)
            - All-zero masks → use default no_mask_embed (blend to default)

            All operations are pure tensor operations (no .item(), no Python
            conditionals on tensor values) to ensure ONNX traceability.
            """
            bs = self._get_batch_size(points, boxes, masks)
            sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())

            if points is not None:
                coords, labels = points
                # Always pad points when boxes input exists (even if dummy)
                # The box embeddings will be masked out later if they're dummy
                point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
                sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

            if boxes is not None:
                # Embed boxes first
                box_embeddings = self._embed_boxes(boxes)

                # Detect dummy boxes: check if all coordinates are zero
                # Shape: boxes is typically [B, 4] or [B, 1, 4]
                boxes_flat = boxes.reshape(boxes.shape[0], -1)  # [B, 4] or [B, num_boxes*4]
                boxes_sum = boxes_flat.abs().sum(dim=1, keepdim=True)  # [B, 1]

                # Create mask: 1.0 if boxes are valid (non-zero), 0.0 if dummy (all zeros)
                has_valid_boxes = (boxes_sum > 0).float()  # [B, 1]
                has_valid_boxes = has_valid_boxes.unsqueeze(-1)  # [B, 1, 1]
                has_valid_boxes = has_valid_boxes.expand(-1, box_embeddings.shape[1], -1)  # [B, num_boxes, 1]
                has_valid_boxes = has_valid_boxes.expand_as(box_embeddings)  # [B, num_boxes, embed_dim]

                # Zero out box embeddings for dummy boxes (element-wise multiplication)
                box_embeddings = box_embeddings * has_valid_boxes

                # Concatenate (zeros will be concatenated if boxes were dummy)
                sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

            if masks is not None:
                # Compute mask embeddings
                mask_embeddings = self._embed_masks(masks)

                # Detect dummy masks: check if all values are zero
                # Shape: masks is [B, 1, H, W]
                masks_flat = masks.reshape(masks.shape[0], -1)  # [B, H*W]
                masks_sum = masks_flat.abs().sum(dim=1, keepdim=True)  # [B, 1]

                # Create mask: 1.0 if masks are valid (non-zero), 0.0 if dummy (all zeros)
                has_valid_masks = (masks_sum > 0).float()  # [B, 1]

                # Get default "no mask" embedding
                no_mask_embed = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                    bs,
                    -1,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                )

                # Blend between mask embeddings and no_mask embeddings
                # has_valid_masks: [B, 1] -> [B, 1, 1, 1] -> [B, embed_dim, H, W]
                has_valid_masks = has_valid_masks.view(bs, 1, 1, 1)  # [B, 1, 1, 1]
                has_valid_masks = has_valid_masks.expand_as(mask_embeddings)  # [B, embed_dim, H, W]

                # If masks are valid, use mask_embeddings; otherwise use no_mask_embed
                dense_embeddings = has_valid_masks * mask_embeddings + (1 - has_valid_masks) * no_mask_embed
            else:
                dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                    bs,
                    -1,
                    self.image_embedding_size[0],
                    self.image_embedding_size[1],
                )

            return sparse_embeddings, dense_embeddings

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
        boxes: torch.Tensor,
        mask_input: torch.Tensor,
        original_size: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h, w = original_size
        original_size = (h, w)
        self.sam_predictor.set_torch_image(transformed_image, original_size)
        return self.sam_predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=boxes,
            mask_input=mask_input,
            multimask_output=True,
            return_logits=False,
        )

    def export(self, output_path: str, backend: Backend = Backend.ONNX) -> Path:
        """Export the model to the specified format.

        Args:
            output_path: The path to export the model to.
            backend: The backend to export to (Backend.ONNX or Backend.OPENVINO).
        """
        # Move all model components to CPU for export compatibility
        logger.info("Moving SAM model to CPU for export...")
        self.sam_predictor.model.to("cpu")

        self._freeze_modules([
            self.sam_predictor.model.mask_decoder,
            self.sam_predictor.model.prompt_encoder,
            self.sam_predictor.model.image_encoder,
        ])

        # Create dummy inputs on CPU to match model device
        dummy_image = torch.zeros((1, 3, self.input_size, self.input_size), dtype=torch.float32, device="cpu")
        dummpy_coords = torch.rand((10, 1, 2), dtype=torch.float32, device="cpu") * self.input_size
        dummy_labels = torch.ones((10, 1), dtype=torch.float32, device="cpu")
        dummy_boxes = torch.rand((10, 1, 4), dtype=torch.float32, device="cpu") * self.input_size
        dummy_mask_input = torch.rand((10, 1, 256, 256), dtype=torch.float32, device="cpu")
        original_size = torch.tensor((self.input_size, self.input_size), dtype=torch.int32, device="cpu")

        model_inputs = (
            dummy_image,
            dummpy_coords,
            dummy_labels,
            dummy_boxes,
            dummy_mask_input,
            original_size,
        )

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

        if backend == Backend.ONNX:
            # Define dynamic axes
            dynamic_axes = {
                "transformed_image": {0: "batch_size", 2: "height", 3: "width"},
                "point_coords": {0: "num_masks", 1: "num_points"},
                "point_labels": {0: "num_masks", 1: "num_points"},
                "boxes": {0: "num_masks", 1: "num_boxes"},
                "mask_input": {0: "num_masks"},
            }
            with torch.no_grad():
                try:
                    # Export to ONNX
                    torch.onnx.export(
                        self,
                        model_inputs,
                        output_path / "exported_sam.onnx",
                        opset_version=20,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=dynamic_axes,
                        verbose=True,
                    )
                    return output_path / "exported_sam.onnx"
                except Exception as e:
                    msg = f"Error exporting to ONNX: {e}"
                    logger.exception(msg)
                    raise e

        if backend == Backend.OPENVINO:
            dynamic_shapes = {
                "transformed_image": openvino.PartialShape([-1, 3, -1, -1]),
                "point_coords": openvino.PartialShape([-1, -1, 2]),
                "point_labels": openvino.PartialShape([-1, -1]),
                "boxes": openvino.PartialShape([-1, 1, 4]),
                "mask_input": openvino.PartialShape([-1, 1, 256, 256]),
                "original_size": openvino.PartialShape([2]),
            }
            try:
                ov_model = openvino.convert_model(
                    input_model=self,
                    example_input=model_inputs,
                    input=dynamic_shapes,
                )
                for i, ov_output in enumerate(ov_model.outputs):
                    ov_output.get_tensor().set_names({output_names[i]})
                openvino.save_model(ov_model, output_path / "exported_sam.xml")
                return output_path / "exported_sam.xml"
            except Exception as e:
                msg = f"Error exporting to OpenVINO IR: {e}"
                logger.exception(msg)
                raise e
