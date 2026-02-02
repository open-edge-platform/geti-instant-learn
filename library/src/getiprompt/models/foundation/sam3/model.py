# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SAM3 model and related components (GeometryEncoder, MaskDecoder, scoring, etc.)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch import Tensor

from transformers import CLIPTextModelWithProjection

from .common import (
    Attention,
    MLP,
    PreTrainedModelBase,
    SinePositionEmbedding,
    box_cxcywh_to_xyxy,
    concat_padded_sequences,
    expand_attention_mask,
    inverse_sigmoid,
)
from .configuration import Config, GeometryEncoderConfig, MaskDecoderConfig
from .detr import DecoderMLP, DetrDecoder, DetrEncoder
from .vit import VisionModel


class GeometryEncoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int = 256,
        num_attention_heads: int = 8,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.dropout = nn.Dropout(dropout)

        self.cross_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size)

        self.mlp = MLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout=hidden_dropout,
        )
        self.layer_norm3 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        prompt_feats: Tensor,
        vision_feats: Tensor,
        vision_pos_encoding: Tensor,
        prompt_mask: Tensor,
        **kwargs,
    ):
        residual = prompt_feats
        hidden_states = self.layer_norm1(prompt_feats)
        hidden_states, _ = self.self_attn(
            query=hidden_states, key=hidden_states, value=hidden_states, attention_mask=prompt_mask, **kwargs
        )
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        key = vision_feats + vision_pos_encoding
        hidden_states, _ = self.cross_attn(query=hidden_states, key=key, value=vision_feats, **kwargs)
        hidden_states = self.dropout(hidden_states) + residual
        residual = hidden_states
        hidden_states = self.layer_norm3(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states) + residual

        return hidden_states


class GeometryEncoder(nn.Module):
    """
    Encoder for geometric prompts (boxes).

    Boxes are encoded using three approaches:
     - Direct projection: linear projection from coordinate space to hidden_size
     - Pooling: pool features from the backbone at the specified location (ROI align for boxes)
     - Position encoding: use position encoding of the box center

    These encodings are combined additively and further processed with transformer layers.
    """

    def __init__(self, config: GeometryEncoderConfig):
        super().__init__()
        hidden_size = config.hidden_size
        roi_size = config.roi_size
        num_layers = config.num_layers
        num_attention_heads = config.num_attention_heads
        intermediate_size = config.intermediate_size
        dropout = config.dropout
        hidden_act = config.hidden_act
        hidden_dropout = config.hidden_dropout

        self.hidden_size = hidden_size
        self.roi_size = roi_size

        self.position_encoding = SinePositionEmbedding(num_pos_feats=hidden_size // 2, normalize=True)
        self.label_embed = nn.Embedding(2, hidden_size)
        self.cls_embed = nn.Embedding(1, hidden_size)

        self.boxes_direct_project = nn.Linear(4, hidden_size)
        self.boxes_pool_project = nn.Conv2d(hidden_size, hidden_size, roi_size)
        self.boxes_pos_enc_project = nn.Linear(hidden_size + 2, hidden_size)

        self.vision_layer_norm = nn.LayerNorm(hidden_size)

        self.final_proj = nn.Linear(hidden_size, hidden_size)
        self.prompt_layer_norm = nn.LayerNorm(hidden_size)

        self.layers = nn.ModuleList([
            GeometryEncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                hidden_act=hidden_act,
                hidden_dropout=hidden_dropout,
            )
            for _ in range(num_layers)
        ])
        self.output_layer_norm = nn.LayerNorm(hidden_size)

    def _encode_box_coordinates(
        self, center_x: torch.Tensor, center_y: torch.Tensor, width: torch.Tensor, height: torch.Tensor
    ) -> torch.Tensor:
        """
        Encode box coordinates by combining position-encoded centers with raw width/height.

        Args:
            center_x: 1D tensor of box center x coordinates
            center_y: 1D tensor of box center y coordinates
            width: 1D tensor of box widths
            height: 1D tensor of box heights

        Returns:
            Encoded box coordinates [N, embedding_dim]
        """
        pos_x, pos_y = self.position_encoding.encode_1d_positions(center_x, center_y)
        pos = torch.cat((pos_y, pos_x, height[:, None], width[:, None]), dim=1)
        return pos

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, vision_features):
        """Encode box prompts. Mask convention: True=valid, False=padding."""
        batch_size, num_boxes = boxes.shape[:2]
        height, width = vision_features.shape[-2:]
        boxes_embed = self.boxes_direct_project(boxes)

        # Pool features using ROI align
        # Convert boxes from CxCyWH to xyxy format and denormalize
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)
        scale = torch.tensor([width, height, width, height], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
        scale = scale.view(1, 1, 4)
        boxes_xyxy = boxes_xyxy * scale
        # ROI align expects list of boxes per batch element,
        # convert from bfloat16 to float16 as roi_align only supports float16 and float32
        dtype = torch.float16 if vision_features.dtype == torch.bfloat16 else vision_features.dtype
        sampled_features = torchvision.ops.roi_align(
            vision_features.to(dtype), boxes_xyxy.to(dtype).unbind(0), self.roi_size
        ).to(vision_features.dtype)

        pooled_projection = self.boxes_pool_project(sampled_features)
        pooled_projection = pooled_projection.view(batch_size, num_boxes, self.hidden_size)
        boxes_embed = boxes_embed + pooled_projection

        # Add position encoding
        center_x, center_y, box_width, box_height = boxes.unbind(-1)
        pos_enc = self._encode_box_coordinates(
            center_x.flatten(), center_y.flatten(), box_width.flatten(), box_height.flatten()
        )
        pos_enc = pos_enc.view(batch_size, num_boxes, pos_enc.shape[-1])
        pos_projection = self.boxes_pos_enc_project(pos_enc)
        boxes_embed = boxes_embed + pos_projection

        # Add label embeddings (positive/negative)
        label_embed = self.label_embed(boxes_labels.long())
        return label_embed + boxes_embed, boxes_mask

    def forward(
        self,
        box_embeddings: torch.Tensor,
        box_mask: torch.Tensor,
        box_labels: torch.Tensor,
        img_feats: tuple[torch.Tensor, ...],
        img_pos_embeds: tuple[torch.Tensor, ...] | None = None,
    ):
        """
        Forward pass for encoding geometric prompts.

        Args:
            box_embeddings: Box coordinates in CxCyWH format [batch_size, num_boxes, 4]
            box_mask: Attention mask for boxes [batch_size, num_boxes]
            box_labels: Labels for boxes (positive/negative) [batch_size, num_boxes]
            img_feats: Image features from vision encoder
            img_pos_embeds: Optional position embeddings for image features

        Returns:
            GeometryEncoderOutput containing encoded geometry features and attention mask.
        """
        batch_size = box_embeddings.shape[0]

        # Prepare vision features for cross-attention: flatten spatial dimensions
        vision_feats = img_feats[-1]  # [B, C, H, W]
        vision_pos_embeds = img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(vision_feats)
        vision_feats_flat = vision_feats.flatten(2).transpose(1, 2)  # [B, H*W, C]
        vision_pos_embeds_flat = vision_pos_embeds.flatten(2).transpose(1, 2)  # [B, H*W, C]

        # Normalize image features for pooling operations
        img_feats_last = img_feats[-1]  # [B, C, H, W]
        img_feats_last = img_feats_last.permute(0, 2, 3, 1)  # [B, H, W, C]
        normalized_img_feats = self.vision_layer_norm(img_feats_last)
        normalized_img_feats = normalized_img_feats.permute(0, 3, 1, 2)  # [B, C, H, W]

        prompt_embeds, prompt_mask = self._encode_boxes(box_embeddings, box_mask, box_labels, normalized_img_feats)

        # Add CLS token (always valid)
        cls_embed = self.cls_embed.weight.view(1, self.hidden_size).unsqueeze(0).expand(batch_size, -1, -1)
        cls_mask = torch.ones(batch_size, 1, dtype=prompt_mask.dtype, device=prompt_mask.device)
        prompt_embeds, prompt_mask = concat_padded_sequences(prompt_embeds, prompt_mask, cls_embed, cls_mask)

        prompt_embeds = self.prompt_layer_norm(self.final_proj(prompt_embeds))

        # Create bidirectional attention mask for transformer layers
        prompt_attention_mask = None
        if prompt_mask is not None:
            prompt_attention_mask = expand_attention_mask(prompt_mask)

        # Apply transformer layers with cross-attention to vision features
        for layer in self.layers:
            prompt_embeds = layer(
                prompt_feats=prompt_embeds,
                vision_feats=vision_feats_flat,
                vision_pos_encoding=vision_pos_embeds_flat,
                prompt_mask=prompt_attention_mask,
            )

        # Final output normalization
        prompt_embeds = self.output_layer_norm(prompt_embeds)

        return {
            "last_hidden_state": prompt_embeds,
            "attention_mask": prompt_mask,
        }


class DotProductScoring(nn.Module):
    """
    Computes classification scores by computing dot product between projected decoder queries and pooled text features.
    This is used to determine confidence/presence scores for each query.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        intermediate_size: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        projection_dim = hidden_size

        self.text_mlp = DecoderMLP(
            input_dim=hidden_size,
            hidden_dim=intermediate_size,
            output_dim=hidden_size,
            num_layers=2,
        )
        self.text_mlp_dropout = nn.Dropout(dropout)
        self.text_mlp_out_norm = nn.LayerNorm(hidden_size)

        self.text_proj = nn.Linear(hidden_size, projection_dim)
        self.query_proj = nn.Linear(hidden_size, projection_dim)

        self.scale = float(1.0 / np.sqrt(projection_dim))

        # Clamping to avoid numerical issues
        self.clamp_logits = True
        self.clamp_max_val = 12.0

    def _pool_text_features(self, text_features: torch.Tensor, text_mask: torch.Tensor | None) -> torch.Tensor:
        """
        Mean pool text features, accounting for padding.

        Args:
            text_features: [batch_size, seq_len, hidden_size]
            text_mask: [batch_size, seq_len] where True indicates valid tokens, False indicates padding

        Returns:
            pooled_text: [batch_size, hidden_size]
        """
        if text_mask is None:
            # No padding, simple mean
            return text_features.mean(dim=1)

        is_valid = text_mask.to(text_features.dtype).unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Count valid tokens per batch
        num_valid = is_valid.sum(dim=1).clamp(min=1.0)  # [batch_size, 1]

        # Mean pool only over valid tokens
        pooled_text = (text_features * is_valid).sum(dim=1) / num_valid  # [batch_size, hidden_size]

        return pooled_text

    def forward(
        self,
        decoder_hidden_states: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute classification scores via dot product.

        Args:
            decoder_hidden_states: [num_layers, batch_size, num_queries, hidden_size]
            text_features: [batch_size, seq_len, hidden_size]
            text_mask: [batch_size, seq_len] where True=valid, False=padding

        Returns:
            scores: [num_layers, batch_size, num_queries, 1]
        """
        orig_text_features = text_features
        text_features = self.text_mlp(text_features)
        text_features = self.text_mlp_dropout(text_features)
        text_features = text_features + orig_text_features
        text_features = self.text_mlp_out_norm(text_features)

        pooled_text = self._pool_text_features(text_features, text_mask)

        proj_text = self.text_proj(pooled_text)
        proj_queries = self.query_proj(decoder_hidden_states)

        proj_text = proj_text.unsqueeze(-1)
        scores = torch.matmul(proj_queries, proj_text.unsqueeze(0))
        scores = scores * self.scale
        if self.clamp_logits:
            scores = scores.clamp(min=-self.clamp_max_val, max=self.clamp_max_val)

        return scores


class MaskEmbedder(nn.Module):
    """
    MLP that embeds object queries for mask prediction.
    Similar to MaskFormer's mask embedder.
    """

    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.Linear(hidden_size, hidden_size),
            ]
        )
        self.activation = nn.ReLU()

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: Query embeddings [batch_size, num_queries, hidden_size]

        Returns:
            Mask embeddings [batch_size, num_queries, hidden_size]
        """
        hidden_states = queries
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i < len(self.layers) - 1:
                hidden_states = self.activation(hidden_states)
        return hidden_states


class PixelDecoder(nn.Module):
    """
    Feature Pyramid Network (FPN) decoder that generates pixel-level features.
    Inspired by MaskFormer's pixel decoder.
    """

    def __init__(
        self,
        hidden_size: int = 256,
        num_upsampling_stages: int = 3,
    ):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(hidden_size, hidden_size, kernel_size=3, stride=1, padding=1)
                for _ in range(num_upsampling_stages)
            ]
        )
        self.norms = nn.ModuleList([nn.GroupNorm(8, hidden_size) for _ in range(num_upsampling_stages)])

        self.out_channels = hidden_size

    def forward(self, backbone_features: list[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            backbone_features: List of backbone features [batch_size, hidden_size, H_i, W_i]
                              from low to high resolution (assumes already projected to hidden_size)

        Returns:
            Pixel embeddings [batch_size, hidden_size, H, W] at the finest resolution
        """
        # Start from the coarsest feature (last in list)
        prev_fpn = backbone_features[-1]
        # Iterate through features from coarse to fine (excluding the last which we started with)
        for layer_idx, backbone_feat in enumerate(reversed(backbone_features[:-1])):
            # Upsample previous FPN output to match current backbone feature size
            prev_fpn = F.interpolate(prev_fpn, size=backbone_feat.shape[-2:], mode="nearest")

            # Add skip connection
            prev_fpn = prev_fpn + backbone_feat

            # Apply conv and norm
            prev_fpn = self.conv_layers[layer_idx](prev_fpn)
            prev_fpn = self.norms[layer_idx](prev_fpn)
            prev_fpn = F.relu(prev_fpn)

        return prev_fpn


class MaskDecoder(PreTrainedModelBase):
    """
    Mask decoder that combines object queries with pixel-level features to predict instance masks.
    Also produces a semantic segmentation output and supports cross-attention to prompts.
    """

    def __init__(self, config: MaskDecoderConfig):
        super().__init__(config)
        self.config = config
        hidden_size = config.hidden_size
        num_upsampling_stages = config.num_upsampling_stages
        num_attention_heads = config.num_attention_heads
        dropout = config.dropout

        self.pixel_decoder = PixelDecoder(
            hidden_size=hidden_size,
            num_upsampling_stages=num_upsampling_stages,
        )

        self.mask_embedder = MaskEmbedder(hidden_size=hidden_size)

        self.instance_projection = nn.Conv2d(self.pixel_decoder.out_channels, hidden_size, kernel_size=1)

        self.semantic_projection = nn.Conv2d(self.pixel_decoder.out_channels, 1, kernel_size=1)

        self.prompt_cross_attn = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
        )
        self.prompt_cross_attn_norm = nn.LayerNorm(hidden_size)
        self.prompt_cross_attn_dropout = nn.Dropout(dropout)

        self.post_init()

    def forward(
        self,
        decoder_queries: torch.Tensor,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
        prompt_features: torch.Tensor | None = None,
        prompt_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        """
        Args:
            decoder_queries: Decoder output queries [batch_size, num_queries, hidden_size]
            backbone_features: List of backbone features to process through FPN
            encoder_hidden_states: Encoder outputs [batch_size, seq_len, hidden_size]
            prompt_features: Prompt features (text + geometry) for cross-attention [batch_size, prompt_len, hidden_size]
            prompt_mask: Padding mask [batch_size, prompt_len] where True=valid, False=padding

        Returns:
            MaskDecoderOutput containing predicted masks and semantic segmentation.
        """
        if prompt_features is not None:
            # Cross-attention: encoder features attend to prompt features
            residual = encoder_hidden_states
            normed_hidden_states = self.prompt_cross_attn_norm(encoder_hidden_states)

            cross_attn_mask = None
            if prompt_mask is not None:
                cross_attn_mask = expand_attention_mask(prompt_mask)

            attn_output, _ = self.prompt_cross_attn(
                query=normed_hidden_states,
                key=prompt_features,
                value=prompt_features,
                attention_mask=cross_attn_mask,
                **kwargs,
            )
            encoder_hidden_states = residual + self.prompt_cross_attn_dropout(attn_output)

        # Process backbone features through FPN to get pixel embeddings
        pixel_embed = self._embed_pixels(
            backbone_features=backbone_features,
            encoder_hidden_states=encoder_hidden_states,
        )

        # Predict instance masks via dot product between query embeddings and pixel embeddings
        instance_embeds = self.instance_projection(pixel_embed)
        mask_embeddings = self.mask_embedder(decoder_queries)
        pred_masks = torch.einsum("bqc,bchw->bqhw", mask_embeddings, instance_embeds)

        # Generate semantic segmentation
        semantic_seg = self.semantic_projection(pixel_embed)

        return {
            "pred_masks": pred_masks,
            "semantic_seg": semantic_seg,
            "attentions": None,
        }

    def _embed_pixels(
        self,
        backbone_features: list[torch.Tensor],
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed pixels by combining backbone FPN features with encoder vision features.
        The encoder vision features replace the finest-resolution backbone feature.

        Args:
            backbone_features: List of backbone features [batch_size, C, H_i, W_i]
            encoder_hidden_states: Encoder outputs [batch_size, seq_len, hidden_size]

        Returns:
            Pixel embeddings [batch_size, hidden_size, H, W]
        """
        backbone_visual_feats = [feat.clone() for feat in backbone_features]

        # Extract vision features from encoder output and reshape to spatial format
        spatial_dim = backbone_features[-1].shape[-2] * backbone_features[-1].shape[-1]
        encoder_visual_embed = encoder_hidden_states[:, :spatial_dim, :]
        batch_size, _, hidden_size = encoder_visual_embed.shape
        height, width = backbone_features[-1].shape[-2:]
        encoder_visual_embed = encoder_visual_embed.transpose(1, 2).reshape(batch_size, hidden_size, height, width)

        # Replace finest backbone feature with encoder vision features
        backbone_visual_feats[-1] = encoder_visual_embed

        # Process through FPN decoder
        pixel_embed = self.pixel_decoder(backbone_visual_feats)

        return pixel_embed


class Sam3Model(PreTrainedModelBase):
    input_modalities = ["image", "text"]
    _checkpoint_conversion_mapping = {
        r"detector_model.(.+)": r"\1"  # the regex allows to remove the prefix, and add it back in revert mode
    }
    _keys_to_ignore_on_load_unexpected = [
        r"^tracker_model.",
        r"^tracker_neck.",
    ]

    def __init__(self, config: Config):
        # loading from a sam3_video config
        if hasattr(config, "detector_config") and config.detector_config is not None:
            detector_config = config.detector_config
            if isinstance(detector_config, dict):
                detector_config = Config(**detector_config)
            config = detector_config
        super().__init__(config)
        self.vision_encoder = VisionModel(config.vision_config)
        self.text_encoder = CLIPTextModelWithProjection(config.text_config)
        self.vocab_size = config.text_config.vocab_size

        self.text_projection = nn.Linear(config.text_config.hidden_size, config.detr_encoder_config.hidden_size)

        self.geometry_encoder = GeometryEncoder(config.geometry_encoder_config)
        self.detr_encoder = DetrEncoder(config.detr_encoder_config)
        self.detr_decoder = DetrDecoder(config.detr_decoder_config)
        self.mask_decoder = MaskDecoder(config.mask_decoder_config)

        self.dot_product_scoring = DotProductScoring(
            hidden_size=config.detr_decoder_config.hidden_size,
            intermediate_size=config.detr_decoder_config.intermediate_size,
            dropout=config.detr_decoder_config.dropout,
        )

        self.post_init()

    def get_text_features(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ):
        r"""
        Get text features from the text encoder.
        
        Returns the CLIP text model output with `pooler_output` attribute containing
        the projected text embeddings.
        """
        text_outputs = self.text_encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True, **kwargs
        )
        last_hidden_state = text_outputs.last_hidden_state
        text_outputs.pooler_output = self.text_projection(last_hidden_state)

        return text_outputs

    def get_vision_features(
        self,
        pixel_values: torch.FloatTensor,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        r"""
        Example:

        ```python
        >>> from transformers import Model, Processor
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO

        >>> model = Model.from_pretrained("facebook/sam3")
        >>> processor = Processor.from_pretrained("facebook/sam3")

        >>> # Pre-compute vision embeddings
        >>> url = "http://images.cocodataset.org/val2017/000000077595.jpg"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read()))
        >>> img_inputs = processor(images=image, return_tensors="pt")
        >>> vision_embeds = model.get_vision_features(pixel_values=img_inputs.pixel_values)

        >>> # Reuse vision embeddings for multiple text prompts
        >>> text_inputs = processor(text="cat", return_tensors="pt")
        >>> outputs = model(vision_embeds=vision_embeds, input_ids=text_inputs.input_ids)
        ```
        """
        vision_outputs = self.vision_encoder(pixel_values, **kwargs)
        return vision_outputs

    def forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        vision_embeds: dict[str, torch.Tensor] | None = None,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        text_embeds: torch.FloatTensor | None = None,
        input_boxes: torch.FloatTensor | None = None,
        input_boxes_labels: torch.LongTensor | None = None,
        **kwargs,
    ) -> dict[str, torch.Tensor | None]:
        r"""
        vision_embeds (`dict`, *optional*):
            Pre-computed vision embeddings. Can be used to easily reuse vision embeddings. If provided, `pixel_values`
            should not be passed. Mutually exclusive with `pixel_values`.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Pre-computed text embeddings. Can be used to easily reuse text embeddings. If provided, `input_ids`
            should not be passed. Mutually exclusive with `input_ids`.
        input_boxes (`torch.FloatTensor` of shape `(batch_size, num_boxes, 4)`, *optional*):
            Normalized box coordinates in [0, 1] range, in (cx, cy, w, h) format.
        input_boxes_labels (`torch.LongTensor` of shape `(batch_size, num_boxes)`, *optional*):
            Labels for boxes: 1 (positive), 0 (negative).

        Example:

        ```python
        >>> from PIL import Image
        >>> import httpx
        >>> from io import BytesIO
        >>> from transformers import AutoModel, AutoProcessor

        >>> model = AutoModel.from_pretrained("facebook/sam3")
        >>> processor = AutoProcessor.from_pretrained("facebook/sam3")

        >>> url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/model_doc/sam-car.png"
        >>> with httpx.stream("GET", url) as response:
        ...     image = Image.open(BytesIO(response.read())).convert("RGB")
        >>> text = "car"
        >>> inputs = processor(images=image, text=text, return_tensors="pt")

        >>> # Get segmentation output
        >>> outputs = model(**inputs)
        >>> pred_masks = outputs["pred_masks"]
        >>> pred_boxes = outputs["pred_boxes"]
        ```
        """
        if (pixel_values is None) == (vision_embeds is None):
            raise ValueError("You must specify exactly one of pixel_values or vision_embeds")

        if (input_ids is None) == (text_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or text_embeds")

        if pixel_values is not None:
            batch_size = pixel_values.shape[0]
            device = pixel_values.device
        else:
            batch_size = vision_embeds["fpn_hidden_states"][0].shape[0]
            device = vision_embeds["fpn_hidden_states"][0].device

        if vision_embeds is None:
            vision_outputs = self.vision_encoder(pixel_values, **kwargs)
        else:
            vision_outputs = vision_embeds

        fpn_hidden_states = vision_outputs["fpn_hidden_states"][:-1]
        fpn_position_encoding = vision_outputs["fpn_position_encoding"][:-1]

        if text_embeds is None:
            text_features = self.get_text_features(
                input_ids=input_ids, attention_mask=attention_mask
            ).pooler_output
        else:
            text_features = text_embeds

        text_mask = attention_mask.bool() if attention_mask is not None else None
        has_geometry_prompts = input_boxes is not None and input_boxes.numel() > 0

        geometry_prompt_features = None
        geometry_prompt_mask = None

        if has_geometry_prompts:
            if input_boxes is not None and input_boxes.numel() > 0:
                box_embeddings = input_boxes  # [batch_size, num_boxes, 4]
                box_labels = (
                    input_boxes_labels
                    if input_boxes_labels is not None
                    else torch.ones_like(box_embeddings[..., 0], dtype=torch.long)
                )
                box_mask = (
                    (input_boxes_labels != -10)
                    if input_boxes_labels is not None
                    else torch.ones(batch_size, input_boxes.shape[1], dtype=torch.bool, device=device)
                )
                box_labels = torch.where(box_labels == -10, 0, box_labels)
            else:
                box_embeddings = torch.zeros(batch_size, 0, 4, dtype=text_features.dtype, device=device)
                box_labels = torch.zeros(batch_size, 0, dtype=torch.long, device=device)
                box_mask = torch.zeros(batch_size, 0, dtype=torch.bool, device=device)

            geometry_outputs = self.geometry_encoder(
                box_embeddings=box_embeddings,
                box_mask=box_mask,
                box_labels=box_labels,
                img_feats=fpn_hidden_states,
                img_pos_embeds=fpn_position_encoding,
            )

            geometry_prompt_features = geometry_outputs["last_hidden_state"]
            geometry_prompt_mask = geometry_outputs["attention_mask"]

        if geometry_prompt_features is not None:
            # Repeat text_features for all geometry prompts
            if text_features.shape[0] == 1 and geometry_prompt_features.shape[0] > 1:
                text_features = text_features.repeat(geometry_prompt_features.shape[0], 1, 1)
            combined_prompt_features = torch.cat([text_features, geometry_prompt_features], dim=1)
            if text_mask is not None and text_mask.shape[0] == 1 and geometry_prompt_mask.shape[0] > 1:
                text_mask = text_mask.repeat(geometry_prompt_mask.shape[0], 1)

            if text_mask is not None and geometry_prompt_mask is not None:
                combined_prompt_mask = torch.cat([text_mask, geometry_prompt_mask], dim=1)
            elif text_mask is not None:
                geo_valid_mask = torch.ones(
                    batch_size, geometry_prompt_features.shape[1], dtype=torch.bool, device=device
                )
                combined_prompt_mask = torch.cat([text_mask, geo_valid_mask], dim=1)
            elif geometry_prompt_mask is not None:
                text_valid_mask = torch.ones(batch_size, text_features.shape[1], dtype=torch.bool, device=device)
                combined_prompt_mask = torch.cat([text_valid_mask, geometry_prompt_mask], dim=1)
            else:
                combined_prompt_mask = None
        else:
            combined_prompt_features = text_features
            combined_prompt_mask = text_mask

        encoder_outputs = self.detr_encoder(
            vision_features=[fpn_hidden_states[-1]],
            text_features=combined_prompt_features,
            vision_pos_embeds=[fpn_position_encoding[-1]],
            text_mask=combined_prompt_mask,
            **kwargs,
        )

        decoder_outputs = self.detr_decoder(
            vision_features=encoder_outputs["last_hidden_state"],
            text_features=encoder_outputs["text_features"],
            vision_pos_encoding=encoder_outputs["pos_embeds_flattened"],
            text_mask=combined_prompt_mask,
            spatial_shapes=encoder_outputs["spatial_shapes"],
            **kwargs,
        )

        # Refine boxes from decoder
        all_box_offsets = self.detr_decoder.box_head(decoder_outputs["intermediate_hidden_states"])
        reference_boxes_inv_sig = inverse_sigmoid(decoder_outputs["reference_boxes"])
        all_pred_boxes_cxcywh = (reference_boxes_inv_sig + all_box_offsets).sigmoid()
        all_pred_boxes = box_cxcywh_to_xyxy(all_pred_boxes_cxcywh)

        all_pred_logits = self.dot_product_scoring(
            decoder_hidden_states=decoder_outputs["intermediate_hidden_states"],
            text_features=encoder_outputs["text_features"],
            text_mask=combined_prompt_mask,
        ).squeeze(-1)

        pred_logits = all_pred_logits[-1]
        pred_boxes = all_pred_boxes[-1]
        decoder_hidden_states = decoder_outputs["intermediate_hidden_states"][-1]
        presence_logits = decoder_outputs["presence_logits"][-1]

        mask_outputs = self.mask_decoder(
            decoder_queries=decoder_hidden_states,
            backbone_features=list(fpn_hidden_states),
            encoder_hidden_states=encoder_outputs["last_hidden_state"],
            prompt_features=combined_prompt_features,
            prompt_mask=combined_prompt_mask,
            **kwargs,
        )

        return {
            "pred_masks": mask_outputs["pred_masks"],
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
            "presence_logits": presence_logits,
            "semantic_seg": mask_outputs["semantic_seg"],
            "decoder_hidden_states": decoder_outputs.get("hidden_states"),
            "decoder_reference_boxes": decoder_outputs["reference_boxes"],
            "encoder_hidden_states": encoder_outputs.get("hidden_states"),
            "vision_hidden_states": vision_outputs.get("hidden_states"),
            "vision_attentions": vision_outputs.get("attentions"),
            "detr_encoder_attentions": encoder_outputs.get("attentions"),
            "detr_decoder_attentions": decoder_outputs.get("attentions"),
            "mask_decoder_attentions": mask_outputs.get("attentions"),
        }


__all__ = [
    "GeometryEncoderLayer",
    "GeometryEncoder",
    "DotProductScoring",
    "MaskEmbedder",
    "PixelDecoder",
    "MaskDecoder",
    "Sam3Model",
]
