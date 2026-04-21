# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 inference graph for ONNX/OpenVINO export.

Provides a traceable nn.Module that wraps the SAM3 forward pass with
frozen text features, suitable for ONNX export and OpenVINO conversion.

The canvas-based visual exemplar mode is inherently export-friendly because:
- Canvas construction (image stitching) is pure Python preprocessing
- The model runs in CLASSIC mode (single forward pass per canvas)
- Text features are fixed ("visual") and baked as model buffers
- Coordinate remapping is pure Python postprocessing

Export flow: PyTorch → ONNX → OpenVINO IR (same path as Matcher).
"""

from __future__ import annotations

import torch
from torch import nn


class Sam3CanvasInferenceGraph(nn.Module):
    """Traceable SAM3 inference graph with frozen text features for export.

    Wraps the SAM3 model forward pass for canvas-based visual exemplar mode.
    Text features are baked as buffers; the graph accepts a preprocessed
    canvas image and bbox prompts as dynamic inputs.

    Canvas construction (stitching reference + target) and prediction
    remapping (extracting target-region boxes) stay in Python and are
    NOT part of this graph.

    Args:
        sam3_model: The Sam3Model instance (detector).
        preprocessor: Sam3Preprocessor for image normalization.
        postprocessor: Sam3Postprocessor for score/box/mask extraction.
        text_input_ids: Frozen CLIP token IDs for the text prompt [1, seq_len].
        text_attention_mask: Frozen CLIP attention mask [1, seq_len].

    Example:
        >>> graph = Sam3CanvasInferenceGraph(model, preproc, postproc, ids, mask)
        >>> # Export to ONNX
        >>> dummy_image = torch.randn(1, 3, 1008, 1008)
        >>> dummy_boxes = torch.tensor([[[100, 200, 300, 400]]], dtype=torch.float32)
        >>> torch.onnx.export(graph, (dummy_image, dummy_boxes), "sam3_canvas.onnx")
    """

    def __init__(
        self,
        sam3_model: nn.Module,
        preprocessor: nn.Module,
        postprocessor: nn.Module,
        text_input_ids: torch.LongTensor,
        text_attention_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.sam3_model = sam3_model
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor

        # Freeze text features — always the same for a given fit()
        self.register_buffer("text_input_ids", text_input_ids)
        self.register_buffer("text_attention_mask", text_attention_mask)

    def forward(
        self,
        canvas_image: torch.Tensor,
        input_boxes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run SAM3 on a preprocessed canvas image with bbox prompts.

        Args:
            canvas_image: Raw canvas image [1, 3, H, W] (uint8 or float).
            input_boxes: Normalized bounding boxes [1, N, 4] in xyxy format,
                already scaled to [0, 1] relative to the preprocessed image size.

        Returns:
            Tuple of (scores, boxes, masks) where:
            - scores: [num_detections] confidence scores
            - boxes: [num_detections, 4] boxes in pixel coords (xyxy)
            - masks: [num_detections, H, W] binary masks
        """
        # Preprocess: resize to target_size, normalize to [-1, 1]
        pixel_values, original_sizes = self.preprocessor(canvas_image)

        # Box labels: all positive (foreground) prompts
        input_boxes_labels = torch.ones(
            input_boxes.shape[:2], dtype=torch.long, device=canvas_image.device,
        )

        # Run SAM3 model with frozen text and dynamic boxes
        outputs = self.sam3_model(
            pixel_values=pixel_values,
            input_ids=self.text_input_ids,
            attention_mask=self.text_attention_mask,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
        )

        # Postprocess — the Sam3Postprocessor already has
        # torch.onnx.is_in_onnx_export() guards for export-safe output format.
        target_sizes = [pixel_values.shape[-2:]]
        results = self.postprocessor(outputs, target_sizes=target_sizes)

        # During ONNX export, postprocessor returns tuples (scores, boxes, masks)
        # During eager mode, it returns list[dict] — unpack for consistent interface
        if isinstance(results[0], dict):
            r = results[0]
            return r["scores"], r["boxes"], r["masks"]
        return results[0]  # Already a tuple during export
