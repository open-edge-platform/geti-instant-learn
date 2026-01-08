# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientSAM3 Image model wrapper.

This module provides an extended Sam3Image class with SAM1-style instance
segmentation prediction methods (predict_inst, predict_inst_batch).

These methods are specific to EfficientSAM3's use case of point/box prompting
for instance segmentation, matching the API from the original efficientsam3 repo.
"""

import torch

from getiprompt.models.foundation.sam3.sam3_image import Sam3Image


class EfficientSAM3Image(Sam3Image):
    """Extended Sam3Image with SAM1-style instance prediction methods.

    This class inherits from Sam3Image and adds predict_inst and predict_inst_batch
    methods for SAM1-style instance segmentation using point/box prompts.

    These methods are designed to match the API from:
    https://github.com/SimonZeng7108/efficientsam3

    The key difference from Sam3Image's forward_grounding:
    - predict_inst uses SAM1-style point/box prompts
    - Works with inst_interactive_predictor (SAM3InteractiveImagePredictor)
    - Returns (masks, scores, logits) tuples like original SAM
    """

    def predict_inst(
        self,
        inference_state=None,
        point_coords=None,
        point_labels=None,
        **kwargs,
    ):
        """SAM1-style instance segmentation prediction with point/box prompts.

        This method supports two modes:
        1. With inference_state (recommended): Pass the state from Sam3Processor.set_image()
           and the method will extract backbone features automatically.
        2. Without inference_state: You must call inst_interactive_predictor.set_image()
           before calling this method.

        Args:
            inference_state: State dict from Sam3Processor.set_image() containing backbone_out.
                If provided, features are extracted automatically.
            point_coords: Point coordinates [N, 2] as numpy array
            point_labels: Point labels [N] as numpy array
            **kwargs: Additional arguments for predictor.predict()

        Returns:
            Tuple of (masks_np, iou_predictions_np, low_res_masks_np)
        """
        if not hasattr(self, "inst_interactive_predictor"):
            raise RuntimeError(
                "Model not initialized with instance interactivity. "
                "Set enable_inst_interactivity=True when building the model."
            )

        # If inference_state is provided, extract features from backbone output
        if inference_state is not None:
            orig_h, orig_w = (
                inference_state["original_height"],
                inference_state["original_width"],
            )

            # Get backbone output - check for sam2_backbone_out or regular backbone_out
            backbone_out = inference_state.get("backbone_out", {})
            if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                backbone_out = backbone_out["sam2_backbone_out"]

            # Extract vision features using the model's prepare method
            (_, vision_feats, _, _) = self._prepare_backbone_features(backbone_out)

            # Add no_mem_embed to lowest resolution feature map (as done during training)
            if hasattr(self, "no_mem_embed"):
                vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

            # Get feature sizes from model attribute or compute from features
            if hasattr(self.inst_interactive_predictor, "_bb_feat_sizes"):
                bb_feat_sizes = self.inst_interactive_predictor._bb_feat_sizes
            elif hasattr(self, "_bb_feat_sizes"):
                bb_feat_sizes = self._bb_feat_sizes
            else:
                # Default feature sizes for 1008x1008 input with stride 14
                bb_feat_sizes = [(288, 288), (144, 144), (72, 72)]

            # Reshape features: (L, B, C) -> (B, C, H, W)
            feats = [
                feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
            ][::-1]

            # Set up the predictor's features
            self.inst_interactive_predictor._features = {
                "image_embed": feats[-1],
                "high_res_feats": feats[:-1],
            }
            self.inst_interactive_predictor._is_image_set = True
            self.inst_interactive_predictor._orig_hw = [(orig_h, orig_w)]

            # Run prediction
            result = self.inst_interactive_predictor.predict(
                point_coords=point_coords, point_labels=point_labels, **kwargs
            )

            # Clean up predictor state
            self.inst_interactive_predictor._features = None
            self.inst_interactive_predictor._is_image_set = False

            return result

        # Legacy mode: expect predictor to already have image set
        return self.inst_interactive_predictor.predict(
            point_coords=point_coords, point_labels=point_labels, **kwargs
        )

    def predict_inst_batch(
        self,
        inference_state=None,
        point_coords_batch=None,
        point_labels_batch=None,
        **kwargs,
    ):
        """SAM1-style instance segmentation prediction for batch.

        Args:
            inference_state: State dict from Sam3Processor.set_image_batch().
            point_coords_batch: List of point coordinates
            point_labels_batch: List of point labels
            **kwargs: Additional arguments for predictor.predict_batch()

        Returns:
            Lists of (masks_batch, iou_predictions_batch, low_res_masks_batch)
        """
        if not hasattr(self, "inst_interactive_predictor"):
            raise RuntimeError(
                "Model not initialized with instance interactivity. "
                "Set enable_inst_interactivity=True when building the model."
            )

        # If inference_state is provided, extract features from backbone output
        if inference_state is not None:
            # Get backbone output
            backbone_out = inference_state.get("backbone_out", {})
            if "sam2_backbone_out" in backbone_out and backbone_out["sam2_backbone_out"] is not None:
                backbone_out = backbone_out["sam2_backbone_out"]

            # Extract vision features
            (_, vision_feats, _, _) = self._prepare_backbone_features(backbone_out)

            # Add no_mem_embed
            if hasattr(self, "no_mem_embed"):
                vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

            batch_size = vision_feats[-1].shape[1]
            orig_heights = inference_state["original_heights"]
            orig_widths = inference_state["original_widths"]

            # Get feature sizes
            if hasattr(self.inst_interactive_predictor, "_bb_feat_sizes"):
                bb_feat_sizes = self.inst_interactive_predictor._bb_feat_sizes
            elif hasattr(self, "_bb_feat_sizes"):
                bb_feat_sizes = self._bb_feat_sizes
            else:
                bb_feat_sizes = [(288, 288), (144, 144), (72, 72)]

            # Reshape features for batch
            feats = [
                feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], bb_feat_sizes[::-1])
            ][::-1]

            # Set up predictor
            self.inst_interactive_predictor._features = {
                "image_embed": feats[-1],
                "high_res_feats": feats[:-1],
            }
            self.inst_interactive_predictor._is_image_set = True
            self.inst_interactive_predictor._is_batch = True
            self.inst_interactive_predictor._orig_hw = [
                (h, w) for h, w in zip(orig_heights, orig_widths)
            ]

            # Run batch prediction
            result = self.inst_interactive_predictor.predict_batch(
                point_coords_batch=point_coords_batch,
                point_labels_batch=point_labels_batch,
                **kwargs,
            )

            # Clean up
            self.inst_interactive_predictor._features = None
            self.inst_interactive_predictor._is_image_set = False
            self.inst_interactive_predictor._is_batch = False

            return result

        # Legacy mode
        return self.inst_interactive_predictor.predict_batch(
            point_coords_batch=point_coords_batch,
            point_labels_batch=point_labels_batch,
            **kwargs,
        )
