# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Sam3Model (nn.Module).

Uses a tiny random-weight configuration to verify forward pass shapes and
component interactions without downloading pretrained weights.
"""

import pytest
import torch

from instantlearn.models.sam3.model import (
    DotProductScoring,
    GeometryEncoder,
    MaskDecoder,
    MaskEmbedder,
    PixelDecoder,
    Sam3Model,
)

# ---------------------------------------------------------------------------
# Tiny model config for fast unit tests
# ---------------------------------------------------------------------------
TINY_CONFIG = {
    # Vision
    "vision_hidden_size": 64,
    "vision_intermediate_size": 128,
    "vision_num_hidden_layers": 2,
    "vision_num_attention_heads": 4,
    "num_channels": 3,
    "image_size": 112,
    "patch_size": 14,
    "vision_hidden_act": "gelu",
    "vision_layer_norm_eps": 1e-6,
    "vision_attention_dropout": 0.0,
    "rope_theta": 10000.0,
    "window_size": 4,
    "global_attn_indexes": [1],
    "pretrain_image_size": 112,
    "vision_hidden_dropout": 0.0,
    "fpn_hidden_size": 32,
    "scale_factors": [4.0, 2.0, 1.0, 0.5],
    # Text
    "text_vocab_size": 256,
    "text_hidden_size": 64,
    "text_intermediate_size": 128,
    "text_projection_dim": 32,
    "text_num_hidden_layers": 2,
    "text_num_attention_heads": 4,
    "text_max_position_embeddings": 16,
    "text_hidden_act": "gelu",
    # Geometry
    "geometry_hidden_size": 32,
    "geometry_num_layers": 1,
    "geometry_num_attention_heads": 4,
    "geometry_intermediate_size": 64,
    "geometry_dropout": 0.0,
    "geometry_hidden_act": "relu",
    "geometry_roi_size": 3,
    # DETR encoder
    "detr_encoder_hidden_size": 32,
    "detr_encoder_num_layers": 1,
    "detr_encoder_num_attention_heads": 4,
    "detr_encoder_intermediate_size": 64,
    "detr_encoder_dropout": 0.0,
    "detr_encoder_hidden_act": "relu",
    # DETR decoder
    "detr_decoder_hidden_size": 32,
    "detr_decoder_num_layers": 1,
    "detr_decoder_num_queries": 10,
    "detr_decoder_num_attention_heads": 4,
    "detr_decoder_intermediate_size": 64,
    "detr_decoder_dropout": 0.0,
    "detr_decoder_hidden_act": "relu",
    # Mask decoder
    "mask_decoder_hidden_size": 32,
    "mask_decoder_num_upsampling_stages": 3,
    "mask_decoder_num_attention_heads": 4,
    "mask_decoder_dropout": 0.0,
}


@pytest.fixture
def tiny_model() -> Sam3Model:
    """Create a tiny Sam3Model with random weights."""
    return Sam3Model(**TINY_CONFIG).eval()


# ---------------------------------------------------------------------------
# Sam3Model forward pass
# ---------------------------------------------------------------------------


class TestSam3ModelForward:
    """Test Sam3Model forward pass shapes."""

    def test_forward_text_only(self, tiny_model: Sam3Model) -> None:
        """Test forward with text prompts only (no geometry)."""
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 112, 112)
        input_ids = torch.randint(0, 256, (batch_size, 16))
        attention_mask = torch.ones(batch_size, 16, dtype=torch.long)

        with torch.no_grad():
            outputs = tiny_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        assert "pred_masks" in outputs
        assert "pred_boxes" in outputs
        assert "pred_logits" in outputs
        assert outputs["pred_logits"].shape[0] == batch_size
        assert outputs["pred_boxes"].shape[0] == batch_size
        assert outputs["pred_masks"].shape[0] == batch_size

    def test_forward_with_boxes(self, tiny_model: Sam3Model) -> None:
        """Test forward with text + box prompts."""
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 112, 112)
        input_ids = torch.randint(0, 256, (batch_size, 16))
        attention_mask = torch.ones(batch_size, 16, dtype=torch.long)
        input_boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])  # cxcywh
        input_boxes_labels = torch.ones(batch_size, 1, dtype=torch.long)

        with torch.no_grad():
            outputs = tiny_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_boxes=input_boxes,
                input_boxes_labels=input_boxes_labels,
            )

        assert outputs["pred_logits"].shape[0] == batch_size
        num_queries = TINY_CONFIG["detr_decoder_num_queries"]
        assert outputs["pred_logits"].shape[1] == num_queries

    def test_forward_with_points(self, tiny_model: Sam3Model) -> None:
        """Test forward with text + point prompts."""
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 112, 112)
        input_ids = torch.randint(0, 256, (batch_size, 16))
        attention_mask = torch.ones(batch_size, 16, dtype=torch.long)
        input_points = torch.tensor([[[0.5, 0.5]]])  # xy normalized
        input_points_labels = torch.ones(batch_size, 1, dtype=torch.long)

        with torch.no_grad():
            outputs = tiny_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_points=input_points,
                input_points_labels=input_points_labels,
            )

        assert "pred_masks" in outputs
        assert "pred_boxes" in outputs

    def test_forward_precomputed_vision(self, tiny_model: Sam3Model) -> None:
        """Test forward with pre-computed vision embeddings."""
        batch_size = 1
        pixel_values = torch.randn(batch_size, 3, 112, 112)

        with torch.no_grad():
            vision_embeds = tiny_model.get_vision_features(pixel_values)

        input_ids = torch.randint(0, 256, (batch_size, 16))
        attention_mask = torch.ones(batch_size, 16, dtype=torch.long)

        with torch.no_grad():
            outputs = tiny_model(
                vision_embeds=vision_embeds,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        assert "pred_masks" in outputs

    def test_forward_precomputed_geometry(self, tiny_model: Sam3Model) -> None:
        """Test forward with pre-computed geometry features (visual exemplar path)."""
        batch_size = 1
        hidden = TINY_CONFIG["geometry_hidden_size"]
        pixel_values = torch.randn(batch_size, 3, 112, 112)
        input_ids = torch.randint(0, 256, (batch_size, 16))
        attention_mask = torch.ones(batch_size, 16, dtype=torch.long)

        precomputed_geo = torch.randn(batch_size, 2, hidden)
        precomputed_mask = torch.ones(batch_size, 2, dtype=torch.bool)

        with torch.no_grad():
            outputs = tiny_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                precomputed_geometry_features=precomputed_geo,
                precomputed_geometry_mask=precomputed_mask,
            )

        assert "pred_masks" in outputs
        assert outputs["pred_logits"].shape[0] == batch_size

    def test_forward_raises_no_vision(self, tiny_model: Sam3Model) -> None:
        """Test forward raises when neither pixel_values nor vision_embeds given."""
        input_ids = torch.randint(0, 256, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.long)

        with pytest.raises(ValueError):
            tiny_model(input_ids=input_ids, attention_mask=attention_mask)

    def test_forward_raises_no_text(self, tiny_model: Sam3Model) -> None:
        """Test forward raises when neither input_ids nor text_embeds given."""
        pixel_values = torch.randn(1, 3, 112, 112)

        with pytest.raises(ValueError):
            tiny_model(pixel_values=pixel_values)

    def test_output_keys(self, tiny_model: Sam3Model) -> None:
        """Test all expected output keys are present."""
        pixel_values = torch.randn(1, 3, 112, 112)
        input_ids = torch.randint(0, 256, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.long)

        with torch.no_grad():
            outputs = tiny_model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        expected_keys = {
            "pred_masks",
            "pred_boxes",
            "pred_logits",
            "presence_logits",
            "semantic_seg",
        }
        assert expected_keys.issubset(set(outputs.keys()))


# ---------------------------------------------------------------------------
# Vision features
# ---------------------------------------------------------------------------


class TestSam3ModelVisionFeatures:
    """Test get_vision_features method."""

    def test_vision_features_structure(self, tiny_model: Sam3Model) -> None:
        """Test vision feature output structure."""
        pixel_values = torch.randn(1, 3, 112, 112)

        with torch.no_grad():
            vision = tiny_model.get_vision_features(pixel_values)

        assert "fpn_hidden_states" in vision
        assert "fpn_position_encoding" in vision
        assert isinstance(vision["fpn_hidden_states"], (list, tuple))
        assert len(vision["fpn_hidden_states"]) > 0

    def test_vision_features_batch(self, tiny_model: Sam3Model) -> None:
        """Test vision features with batch > 1."""
        pixel_values = torch.randn(2, 3, 112, 112)

        with torch.no_grad():
            vision = tiny_model.get_vision_features(pixel_values)

        assert vision["fpn_hidden_states"][0].shape[0] == 2


# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------


class TestSam3ModelTextFeatures:
    """Test get_text_features method."""

    def test_text_features_output(self, tiny_model: Sam3Model) -> None:
        """Test text features have pooler_output attribute."""
        input_ids = torch.randint(0, 256, (1, 16))
        attention_mask = torch.ones(1, 16, dtype=torch.long)

        with torch.no_grad():
            text = tiny_model.get_text_features(input_ids, attention_mask)

        assert hasattr(text, "pooler_output")
        assert text.pooler_output.shape[0] == 1


# ---------------------------------------------------------------------------
# GeometryEncoder
# ---------------------------------------------------------------------------


class TestGeometryEncoder:
    """Test GeometryEncoder in isolation."""

    @pytest.fixture
    def geometry_encoder(self) -> GeometryEncoder:
        """Create a small geometry encoder."""
        return GeometryEncoder(
            hidden_size=32,
            num_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            roi_size=3,
        ).eval()

    def test_point_encoding(self, geometry_encoder: GeometryEncoder) -> None:
        """Test encoding point prompts."""
        batch_size = 1
        h, w = 8, 8
        img_feats = (torch.randn(batch_size, 32, h, w),)
        img_pos = (torch.randn(batch_size, 32, h, w),)

        with torch.no_grad():
            outputs = geometry_encoder(
                point_embeddings=torch.rand(batch_size, 2, 2),
                point_mask=torch.ones(batch_size, 2, dtype=torch.bool),
                point_labels=torch.ones(batch_size, 2, dtype=torch.long),
                img_feats=img_feats,
                img_pos_embeds=img_pos,
            )

        assert "last_hidden_state" in outputs
        assert "attention_mask" in outputs
        # Output includes points + cls token
        assert outputs["last_hidden_state"].shape[0] == batch_size

    def test_box_encoding(self, geometry_encoder: GeometryEncoder) -> None:
        """Test encoding box prompts."""
        batch_size = 1
        h, w = 8, 8
        img_feats = (torch.randn(batch_size, 32, h, w),)
        img_pos = (torch.randn(batch_size, 32, h, w),)

        with torch.no_grad():
            outputs = geometry_encoder(
                box_embeddings=torch.rand(batch_size, 1, 4),
                box_mask=torch.ones(batch_size, 1, dtype=torch.bool),
                box_labels=torch.ones(batch_size, 1, dtype=torch.long),
                img_feats=img_feats,
                img_pos_embeds=img_pos,
            )

        assert outputs["last_hidden_state"].shape[0] == batch_size

    def test_raises_without_inputs(self, geometry_encoder: GeometryEncoder) -> None:
        """Test raises without any box or point inputs."""
        h, w = 8, 8
        img_feats = (torch.randn(1, 32, h, w),)
        img_pos = (torch.randn(1, 32, h, w),)

        with pytest.raises(ValueError, match="At least one"):
            geometry_encoder(img_feats=img_feats, img_pos_embeds=img_pos)

    def test_drop_spatial_bias(self, geometry_encoder: GeometryEncoder) -> None:
        """Test drop_spatial_bias produces different features."""
        batch_size = 1
        h, w = 8, 8
        img_feats = (torch.randn(batch_size, 32, h, w),)
        img_pos = (torch.randn(batch_size, 32, h, w),)
        points = torch.rand(batch_size, 1, 2)

        with torch.no_grad():
            out_normal = geometry_encoder(
                point_embeddings=points,
                point_mask=torch.ones(batch_size, 1, dtype=torch.bool),
                point_labels=torch.ones(batch_size, 1, dtype=torch.long),
                img_feats=img_feats,
                img_pos_embeds=img_pos,
                drop_spatial_bias=False,
            )
            out_dropped = geometry_encoder(
                point_embeddings=points,
                point_mask=torch.ones(batch_size, 1, dtype=torch.bool),
                point_labels=torch.ones(batch_size, 1, dtype=torch.long),
                img_feats=img_feats,
                img_pos_embeds=img_pos,
                drop_spatial_bias=True,
            )

        # Features should differ when spatial bias is dropped
        assert not torch.allclose(
            out_normal["last_hidden_state"],
            out_dropped["last_hidden_state"],
        )


# ---------------------------------------------------------------------------
# Subcomponents
# ---------------------------------------------------------------------------


class TestPixelDecoder:
    """Test PixelDecoder FPN."""

    def test_forward_shape(self) -> None:
        """Test pixel decoder output shape matches finest input."""
        decoder = PixelDecoder(hidden_size=32, num_upsampling_stages=2).eval()
        features = [
            torch.randn(1, 32, 16, 16),
            torch.randn(1, 32, 8, 8),
            torch.randn(1, 32, 4, 4),
        ]

        with torch.no_grad():
            out = decoder(features)

        assert out.shape == (1, 32, 16, 16)


class TestMaskEmbedder:
    """Test MaskEmbedder MLP."""

    def test_forward_shape(self) -> None:
        """Test mask embedder preserves shape."""
        embedder = MaskEmbedder(hidden_size=32).eval()
        queries = torch.randn(1, 10, 32)

        with torch.no_grad():
            out = embedder(queries)

        assert out.shape == (1, 10, 32)


class TestDotProductScoring:
    """Test DotProductScoring."""

    def test_forward_shape(self) -> None:
        """Test scoring output shape."""
        scorer = DotProductScoring(hidden_size=32, intermediate_size=64).eval()
        decoder_states = torch.randn(1, 1, 10, 32)  # [layers, batch, queries, dim]
        text_features = torch.randn(1, 16, 32)

        with torch.no_grad():
            scores = scorer(decoder_states, text_features)

        assert scores.shape == (1, 1, 10, 1)

    def test_scoring_with_text_mask(self) -> None:
        """Test scoring works with text mask."""
        scorer = DotProductScoring(hidden_size=32, intermediate_size=64).eval()
        decoder_states = torch.randn(1, 1, 5, 32)
        text_features = torch.randn(1, 8, 32)
        text_mask = torch.ones(1, 8, dtype=torch.bool)
        text_mask[0, 4:] = False

        with torch.no_grad():
            scores = scorer(decoder_states, text_features, text_mask)

        assert scores.shape == (1, 1, 5, 1)
