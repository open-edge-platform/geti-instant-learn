# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM exportable model."""

import torch
from torch import nn

import openvino

from segment_anything_hq.predictor import SamPredictor


class SamExportableModel(nn.Module):
    def __init__(self, sam_predictor: SamPredictor):
        super().__init__()
        self.sam_predictor = sam_predictor
        for p in self.sam_predictor.model.mask_decoder.parameters():
            p.requires_grad_(False)

        for p in self.sam_predictor.model.prompt_encoder.parameters():
            p.requires_grad_(False)

        for p in self.sam_predictor.model.image_encoder.parameters():
            p.requires_grad_(False)

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
        # point_coords shapes: (N, pos+neg, 3), dim=3: x, y, score
        # point labels shapes: (N, pos+neg)
        # boxes shapes: (N, 4)
        # low_res_logits.shape: (N, 1, 256, 256)


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
            "original_size"
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
        save_path = output_path.replace(".onnx", ".xml")
        openvino.save_model(exported_model, save_path)
