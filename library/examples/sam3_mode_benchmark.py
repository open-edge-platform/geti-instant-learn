#!/usr/bin/env python3
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM3 mode comparison benchmark.

Tests whether visual exemplar bounding boxes add value beyond text alone,
and experiments with ways to reduce text dependency.

Baseline modes:
  1. Text-Only (Classic)  — fit() with category name, no image/bbox
  2. VE + "visual"        — fit() with reference image+bbox, text="visual"
  3. VE + real name       — fit() with reference image+bbox, real category name

Phase 1 experiments (inference-only patches, no model changes):
  4. VE + mask-text       — Zero out text attention mask so geometry is the only signal
  5. VE + scale-text-0.1  — Scale text features by 0.1
  6. VE + scale-text-0.01 — Scale text features by 0.01
  7. VE + empty-str       — Use "" instead of "visual" as text prompt

Phase 2 experiments (architecture-level, still inference-only):
  8. VE + tile-geo        — Repeat geometry tokens to 32 to balance token ratio
  9. VE + tile+mask       — Tile geometry AND mask text (geometry-only, more tokens)
  10. VE + clip-crop      — Replace text with CLIP ViT-B crop (random projection)
  11. VE + clip-L-aligned — Replace text with CLIP ViT-L crop (SAM3's text_projection)

Phase 3 experiments (community-sourced, drop_spatial_bias=True):
  12. DSB+"visual"         — drop_spatial_bias=True with text="visual"
  13. DSB+real             — drop_spatial_bias=True with real category name
  14. DSB+tile-geo         — drop_spatial_bias + tile geometry 32×
  15. DSB+tile+mask        — drop_spatial_bias + tile geometry + mask text
  16. DSB+mask-text        — drop_spatial_bias + mask text (pure ROI-pooled features)

Phase 4 experiments (richer geometry representations + feature matching):
  Method C — Multi-point from mask (sample N points instead of bbox center):
  17. MP16+"visual"        — 16 mask-sampled points, text="visual"
  18. MP32+"visual"        — 32 mask-sampled points, text="visual"
  19. MP16+mask-text       — 16 mask-sampled points, text masked out
  Method D — Mask-pooled backbone features (FPN features averaged within mask):
  20. MaskPool             — mask-averaged FPN feature as geometry token
  21. MaskPool+tile        — mask-averaged FPN feature tiled to 32 tokens
  22. MaskPool+mask        — mask-averaged FPN, text masked out
  Method E — Backbone feature matching (auto-detect via cosine similarity):
  23. FeatMatch            — no VE pipeline; match reference mask features in target

Phase 5 experiments (FSS-SAM3 unified canvas approach, arxiv:2604.05433):
  24. Canvas+"visual"      — stitch ref+target into 1 image, bbox prompt, text="visual"
  25. Canvas+real          — stitch ref+target, bbox prompt, real category name
  26. Canvas+text-only     — stitch ref+target, text prompt only (no bbox)

Run:
    python sam3_mode_benchmark.py --dataset perseg
    python sam3_mode_benchmark.py --dataset lvis --categories cupcake sheep pastry doughnut
    python sam3_mode_benchmark.py --dataset both
    python sam3_mode_benchmark.py --dataset both --phase1  # Run Phase 1 experiments only
    python sam3_mode_benchmark.py --dataset both --phase2  # Run Phase 2 experiments only
    python sam3_mode_benchmark.py --dataset both --phase3  # Run Phase 3 (drop_spatial_bias)
    python sam3_mode_benchmark.py --dataset both --phase4  # Run Phase 4 (multi-point, mask-pool, feat match)
    python sam3_mode_benchmark.py --dataset both --phase5  # Run Phase 5 (FSS-SAM3 canvas)
"""

from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

from instantlearn.data import LVISDataset, PerSegDataset
from instantlearn.data.base.sample import Sample
from instantlearn.data.lvis import LVISAnnotationMode
from instantlearn.models.sam3 import SAM3, Sam3PromptMode


# ── Defaults ──

PERSEG_ROOT = Path("/home/rgangire/workspace/data/prompt/PerSeg")
LVIS_ROOT = Path("/home/rgangire/workspace/data/prompt/lvis")

PERSEG_CATEGORIES = [
    "dog", "cat", "backpack", "clock", "teddybear",
    "duck_toy", "candle", "chair",
]
LVIS_CATEGORIES = ["cupcake", "sheep", "pastry", "doughnut"]


# ── Helpers ──

def bbox_from_mask(mask: torch.Tensor | np.ndarray) -> np.ndarray:
    """Derive a tight bounding box [x1, y1, x2, y2] from a binary mask."""
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.squeeze()
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.array([0, 0, 1, 1], dtype=np.float32)
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def ensure_bboxes(sample: Sample) -> Sample:
    """If sample has masks but no bboxes, derive bboxes from masks."""
    if sample.bboxes is not None:
        return sample
    if sample.masks is None:
        return sample
    masks = sample.masks
    if isinstance(masks, torch.Tensor):
        n = masks.shape[0]
    else:
        n = len(masks)
    bboxes = np.stack([bbox_from_mask(masks[i]) for i in range(n)])
    sample.bboxes = bboxes
    return sample


def get_reference_and_targets(
    dataset,
    category_name: str,
    max_targets: int = 5,
    shuffle: bool = False,
    seed: int = 42,
    ref_index: int = 0,
):
    """Get one reference sample and target samples for a category."""
    ref_ds = dataset.get_reference_dataset(category=category_name)
    tgt_ds = dataset.get_target_dataset(category=category_name)

    ref_samples = []
    for sample in ref_ds:
        filtered = sample.filter_by_category(category_name)
        if filtered is not None:
            ref_samples.append(ensure_bboxes(filtered))

    if shuffle:
        random.Random(seed).shuffle(ref_samples)

    if ref_index < len(ref_samples):
        ref_samples = [ref_samples[ref_index]]
    elif ref_samples:
        ref_samples = [ref_samples[0]]

    n_tgt = len(tgt_ds)
    if shuffle:
        indices = list(range(n_tgt))
        random.Random(seed + 1).shuffle(indices)
        tgt_samples = [ensure_bboxes(tgt_ds[i]) for i in indices[:max_targets]]
    else:
        tgt_samples = [ensure_bboxes(tgt_ds[i]) for i in range(min(max_targets, n_tgt))]

    return ref_samples, tgt_samples


def strip_annotations(samples: list[Sample]) -> list[Sample]:
    """Image-only copies so GT isn't mistaken for prompts."""
    return [Sample(image=s.image, image_path=s.image_path) for s in samples]


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """IoU between [N,4] and [M,4] boxes in xyxy format."""
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def compute_tp_fp(pred: dict, gt_sample: Sample, category_name: str, iou_threshold: float = 0.5):
    """Count TP/FP by matching predicted boxes to GT via IoU.

    Returns: (tp, fp, n_gt, mean_iou)
    """
    pred_boxes = pred["pred_boxes"][:, :4].cpu().numpy()
    n_pred = len(pred_boxes)

    gt_bboxes = gt_sample.bboxes
    gt_cats = gt_sample.categories
    if gt_bboxes is not None and gt_cats is not None:
        indices = [i for i, c in enumerate(gt_cats) if c == category_name]
        if indices:
            gt_boxes = np.array(gt_bboxes)[indices][:, :4]
            if isinstance(gt_boxes, torch.Tensor):
                gt_boxes = gt_boxes.cpu().numpy()
        else:
            gt_boxes = np.empty((0, 4))
    else:
        gt_boxes = np.empty((0, 4))

    n_gt = len(gt_boxes)
    if n_pred == 0:
        return 0, 0, n_gt, 0.0
    if n_gt == 0:
        return 0, n_pred, 0, 0.0

    iou_matrix = box_iou(pred_boxes, gt_boxes)
    tp = 0
    matched_ious = []
    matched_gt = set()
    for pred_idx in range(n_pred):
        best_gt = iou_matrix[pred_idx].argmax()
        if iou_matrix[pred_idx, best_gt] >= iou_threshold and best_gt not in matched_gt:
            tp += 1
            matched_gt.add(best_gt)
            matched_ious.append(iou_matrix[pred_idx, best_gt])
    fp = n_pred - tp
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    return tp, fp, n_gt, mean_iou


# ── Benchmark runner ──

def _patch_text_mask_zero(model: SAM3) -> None:
    """Exp 1: Zero out all text masks so text tokens are excluded from attention."""
    model.exemplar_text_mask = [
        torch.zeros_like(m) for m in model.exemplar_text_mask
    ]


def _patch_text_scale(model: SAM3, alpha: float) -> None:
    """Exp 2: Scale text feature tensors by alpha."""
    model.exemplar_text_features = [
        f * alpha for f in model.exemplar_text_features
    ]


def _patch_tile_geometry(model: SAM3, target_tokens: int = 32) -> None:
    """Exp 4a: Repeat geometry features along seq dim to match text token count.

    Each geometry feature [1, N, 256] is repeated ceil(target_tokens/N) times
    then trimmed to [1, target_tokens, 256]. Mask is expanded accordingly.
    """
    new_feats = []
    new_masks = []
    for feats, mask in zip(model.exemplar_geometry_features, model.exemplar_geometry_mask, strict=True):
        n = feats.shape[1]
        if n >= target_tokens:
            new_feats.append(feats)
            new_masks.append(mask)
            continue
        repeats = (target_tokens + n - 1) // n
        tiled = feats.repeat(1, repeats, 1)[:, :target_tokens, :]
        tiled_mask = mask.repeat(1, repeats)[:, :target_tokens]
        new_feats.append(tiled)
        new_masks.append(tiled_mask)
    model.exemplar_geometry_features = new_feats
    model.exemplar_geometry_mask = new_masks


def _patch_tile_geo_mask_text(model: SAM3, target_tokens: int = 32) -> None:
    """Exp 4b: Tile geometry AND mask out text — geometry-only with more tokens."""
    _patch_tile_geometry(model, target_tokens)
    _patch_text_mask_zero(model)


class CLIPCropEncoder:
    """Encode image crops via a standalone CLIP vision model.

    Produces [1, K, 256] features from an image crop that can replace
    text features in the SAM3 prompt pipeline.
    """

    def __init__(self, device: str = "cuda", target_dim: int = 256) -> None:
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.target_dim = target_dim
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
        clip_dim = self.clip.config.vision_config.hidden_size  # 768 for ViT-B
        self.proj = torch.nn.Linear(clip_dim, target_dim).to(device)
        # Initialize projection to preserve scale
        torch.nn.init.xavier_uniform_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    @torch.no_grad()
    def encode_crop(self, image: torch.Tensor | np.ndarray, bbox: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a cropped region of an image.

        Args:
            image: CHW tensor (uint8 or float) or HWC numpy array.
            bbox: [x1, y1, x2, y2] bounding box.

        Returns:
            (features, mask) where features is [1, K, target_dim] and mask is [1, K] bool.
        """
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().permute(1, 2, 0).numpy()
            if img_np.dtype != np.uint8:
                img_np = (img_np.clip(0, 255)).astype(np.uint8)
        else:
            img_np = image

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_np.shape[1], max(x2, x1 + 1))
        y2 = min(img_np.shape[0], max(y2, y1 + 1))
        crop = img_np[y1:y2, x1:x2]

        inputs = self.processor(images=crop, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.clip.vision_model(pixel_values=pixel_values)
        # last_hidden_state: [1, num_patches+1, 768] (includes CLS token)
        hidden = outputs.last_hidden_state  # [1, 197, 768] for ViT-B/16 at 224x224

        features = self.proj(hidden)  # [1, 197, 256]
        mask = torch.ones(1, features.shape[1], dtype=torch.bool, device=self.device)
        return features, mask


class CLIPCropEncoderAligned:
    """Encode image crops via CLIP ViT-L/14 projected through SAM3's text_projection.

    CLIP ViT-L has 1024-dim hidden states — same as SAM3's CLIP text encoder —
    so we reuse SAM3's trained text_projection (Linear 1024->256) for proper alignment.
    """

    def __init__(self, device: str = "cuda", sam3_model: SAM3 | None = None) -> None:
        from transformers import CLIPProcessor, CLIPVisionModel

        self.device = device
        self.clip_vision = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        # Reuse SAM3's trained text_projection for alignment (Linear 1024->256)
        if sam3_model is not None:
            self.text_projection = sam3_model.model.text_projection
        else:
            raise ValueError("sam3_model is required for aligned projection")

    @torch.no_grad()
    def encode_crop(self, image: torch.Tensor | np.ndarray, bbox: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode crop via CLIP ViT-L + SAM3's text_projection(1024->256)."""
        if isinstance(image, torch.Tensor):
            img_np = image.cpu().permute(1, 2, 0).numpy()
            if img_np.dtype != np.uint8:
                img_np = (img_np.clip(0, 255)).astype(np.uint8)
        else:
            img_np = image

        x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(img_np.shape[1], max(x2, x1 + 1))
        y2 = min(img_np.shape[0], max(y2, y1 + 1))
        crop = img_np[y1:y2, x1:x2]

        inputs = self.processor(images=crop, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        outputs = self.clip_vision(pixel_values=pixel_values)
        # last_hidden_state: [1, 257, 1024] for ViT-L/14 at 224x224
        hidden = outputs.last_hidden_state  # [1, 257, 1024]
        # Project through SAM3's trained text_projection
        features = self.text_projection(hidden)  # [1, 257, 256]
        mask = torch.ones(1, features.shape[1], dtype=torch.bool, device=self.device)
        return features, mask


def _patch_clip_crop(
    model: SAM3,
    ref_sample: Sample,
    clip_encoder: CLIPCropEncoder | CLIPCropEncoderAligned,
) -> None:
    """Exp 5/5b: Replace text features with CLIP visual crop features.

    For each exemplar category, encode the reference image crop(s) and
    substitute them for the cached text features.
    """
    new_text_feats = []
    new_text_masks = []

    for geo_feats, cat_id in zip(model.exemplar_geometry_features, model.exemplar_category_ids, strict=True):
        # Find bboxes for this category
        cat_bboxes = []
        if ref_sample.bboxes is not None and ref_sample.category_ids is not None:
            for i, cid in enumerate(ref_sample.category_ids):
                if cid == cat_id:
                    cat_bboxes.append(ref_sample.bboxes[i])

        if not cat_bboxes:
            # Fallback: keep original text features
            idx = model.exemplar_category_ids.index(cat_id)
            new_text_feats.append(model.exemplar_text_features[idx])
            new_text_masks.append(model.exemplar_text_mask[idx])
            continue

        # Encode each crop and concatenate
        crop_feats_list = []
        crop_mask_list = []
        for bbox in cat_bboxes:
            feats, mask = clip_encoder.encode_crop(ref_sample.image, bbox)
            crop_feats_list.append(feats)
            crop_mask_list.append(mask)

        all_feats = torch.cat(crop_feats_list, dim=1)  # [1, K*num_crops, 256]
        all_mask = torch.cat(crop_mask_list, dim=1)
        new_text_feats.append(all_feats)
        new_text_masks.append(all_mask)

    model.exemplar_text_features = new_text_feats
    model.exemplar_text_mask = new_text_masks


# ── Phase 4: Multi-point from mask (Method C) ──


def _sample_points_from_mask(
    mask: np.ndarray | torch.Tensor, n_points: int = 16, seed: int = 42,
) -> np.ndarray:
    """Sample N points uniformly from inside a binary mask.

    Returns:
        [N, 2] array of (x, y) coordinates in pixel space.
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    mask = mask.squeeze()
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        h, w = mask.shape
        return np.array([[w // 2, h // 2]], dtype=np.float32)
    rng = np.random.RandomState(seed)
    n = min(n_points, len(ys))
    indices = rng.choice(len(ys), n, replace=False)
    return np.stack([xs[indices], ys[indices]], axis=1).astype(np.float32)


def _create_multipoint_sample(
    ref_sample: Sample, n_points: int = 16, seed: int = 42,
) -> Sample:
    """Create a Sample with N mask-sampled points instead of a bounding box.

    Requires ref_sample to have masks. Falls back to original if no masks.
    """
    new_sample = copy.deepcopy(ref_sample)
    if ref_sample.masks is None:
        return new_sample
    masks = ref_sample.masks
    if isinstance(masks, (np.ndarray, torch.Tensor)) and masks.shape[0] == 0:
        return new_sample

    mask = masks[0]  # First instance mask
    points = _sample_points_from_mask(mask, n_points=n_points, seed=seed)

    new_sample.points = points
    new_sample.bboxes = None  # Remove bboxes so fit() uses points path

    cat = ref_sample.categories[0] if ref_sample.categories else "visual"
    cat_id = int(ref_sample.category_ids[0]) if ref_sample.category_ids is not None else 0
    new_sample.categories = [cat] * len(points)
    new_sample.category_ids = np.array([cat_id] * len(points))

    return new_sample


# ── Phase 4: Mask-pooled backbone features (Method D) ──


def _patch_mask_pooled_features(model: SAM3, ref_sample: Sample) -> None:
    """Replace cached geometry features with mask-averaged FPN features.

    Pools backbone features within the reference mask, producing a richer
    representation than a single grid-sampled point at the bbox center.
    """
    if ref_sample.masks is None:
        return
    mask = ref_sample.masks[0]
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy().squeeze()
    else:
        mask_np = np.asarray(mask).squeeze()

    image_tensor = ref_sample.image.unsqueeze(0) if ref_sample.image.ndim == 3 else ref_sample.image
    with torch.no_grad():
        pixel_values, _ = model.image_preprocessor(image_tensor.to(model.device))
        vision_embeds = model.model.get_vision_features(pixel_values)

    # Finest available FPN scale (72×72 for 1008 input)
    fpn_feat = vision_embeds["fpn_hidden_states"][2]  # [1, 256, H, W]
    _, _C, H, W = fpn_feat.shape

    mask_resized = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    mask_resized = F.interpolate(mask_resized, size=(H, W), mode="bilinear", align_corners=False)
    mask_resized = (mask_resized > 0.5).to(fpn_feat.device)  # [1, 1, H, W]

    masked_feat = fpn_feat * mask_resized
    n_pixels = mask_resized.sum().clamp(min=1)
    pooled = masked_feat.sum(dim=(2, 3)) / n_pixels  # [1, 256]
    pooled = pooled.unsqueeze(1)  # [1, 1, 256]

    dtype = model.exemplar_geometry_features[0].dtype
    model.exemplar_geometry_features = [pooled.to(dtype)]
    model.exemplar_geometry_mask = [torch.ones(1, 1, dtype=torch.bool, device=model.device)]


# ── Phase 4: Backbone feature matching (Method E) ──


def _feature_match_predict(
    model: SAM3,
    ref_sample: Sample,
    target_samples: list[Sample],
    sim_threshold: float = 0.5,
    min_area: int = 100,
) -> list[dict]:
    """Predict objects via backbone feature matching (SAMv1 D.6 approach).

    1. Encode reference image → mask-pool FPN features → reference vector
    2. For each target: cosine similarity map → threshold → connected components → bboxes
    """
    from scipy import ndimage

    empty_pred = {"pred_boxes": torch.empty(0, 5)}
    if ref_sample.masks is None:
        return [empty_pred for _ in target_samples]

    mask = ref_sample.masks[0]
    mask_np = mask.cpu().numpy().squeeze() if isinstance(mask, torch.Tensor) else np.asarray(mask).squeeze()

    # Encode reference and pool at mask
    ref_img = ref_sample.image.unsqueeze(0) if ref_sample.image.ndim == 3 else ref_sample.image
    with torch.no_grad():
        ref_pv, _ = model.image_preprocessor(ref_img.to(model.device))
        ref_vision = model.model.get_vision_features(ref_pv)

    ref_feat = ref_vision["fpn_hidden_states"][2]  # [1, 256, H, W]
    _, _C, H, W = ref_feat.shape

    mask_r = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
    mask_r = F.interpolate(mask_r, size=(H, W), mode="bilinear", align_corners=False)
    mask_r = (mask_r > 0.5).to(ref_feat.device)

    masked_feat = ref_feat * mask_r
    n_pix = mask_r.sum().clamp(min=1)
    ref_vector = F.normalize(masked_feat.sum(dim=(2, 3)) / n_pix, dim=-1)  # [1, 256]

    predictions = []
    for tgt in target_samples:
        tgt_img = tgt.image.unsqueeze(0) if tgt.image.ndim == 3 else tgt.image
        with torch.no_grad():
            tgt_pv, _ = model.image_preprocessor(tgt_img.to(model.device))
            tgt_vision = model.model.get_vision_features(tgt_pv)

        tgt_feat = tgt_vision["fpn_hidden_states"][2]  # [1, 256, H_t, W_t]
        tgt_norm = F.normalize(tgt_feat, dim=1)
        sim_map = torch.einsum("bc,bchw->bhw", ref_vector, tgt_norm).squeeze(0)  # [H_t, W_t]

        # Upsample to original image size
        orig_h, orig_w = tgt.image.shape[-2:]
        sim_up = F.interpolate(
            sim_map.unsqueeze(0).unsqueeze(0), size=(orig_h, orig_w),
            mode="bilinear", align_corners=False,
        ).squeeze().cpu().numpy()

        binary = (sim_up > sim_threshold).astype(np.uint8)
        labeled, n_comp = ndimage.label(binary)

        boxes = []
        for comp_id in range(1, n_comp + 1):
            comp_mask = labeled == comp_id
            if comp_mask.sum() < min_area:
                continue
            ys, xs = np.where(comp_mask)
            boxes.append([xs.min(), ys.min(), xs.max(), ys.max(), 1.0])

        pred_boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty(0, 5)
        predictions.append({"pred_boxes": pred_boxes})

    return predictions


# ── Phase 5: FSS-SAM3 Unified Canvas (arxiv:2604.05433) ──


def _create_canvas(
    ref_image: torch.Tensor,
    tgt_image: torch.Tensor,
    ref_bbox: np.ndarray,
    split_ratio: float = 0.6,
) -> tuple[torch.Tensor, np.ndarray, tuple[int, int, int, int]]:
    """Create a unified canvas with reference on top and target on bottom.

    The FSS-SAM3 paper found vertical layout with support at top (ratio 0.6)
    gives best results. Both images are force-resized to fill their regions.

    Args:
        ref_image: Reference image tensor (C, H_ref, W_ref).
        tgt_image: Target image tensor (C, H_tgt, W_tgt).
        ref_bbox: Reference bounding box [x1, y1, x2, y2] in pixel coords.
        split_ratio: Fraction of canvas height for reference (default 0.6).

    Returns:
        (canvas, canvas_bbox, tgt_region) where:
        - canvas: (C, canvas_H, canvas_W) tensor
        - canvas_bbox: [x1, y1, x2, y2] mapped to canvas coordinates
        - tgt_region: (tx, ty, tw, th) target region on canvas
    """
    C = ref_image.shape[0]
    ref_h, ref_w = ref_image.shape[1], ref_image.shape[2]
    tgt_h, tgt_w = tgt_image.shape[1], tgt_image.shape[2]

    # Canvas size = max of both widths, sum of scaled heights
    canvas_w = max(ref_w, tgt_w)
    canvas_h = canvas_w  # Square canvas for consistency

    # Reference gets top portion, target gets bottom
    ref_canvas_h = int(canvas_h * split_ratio)
    tgt_canvas_h = canvas_h - ref_canvas_h

    # Resize both images to fill their canvas regions
    ref_resized = F.interpolate(
        ref_image.unsqueeze(0).float(), size=(ref_canvas_h, canvas_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)
    tgt_resized = F.interpolate(
        tgt_image.unsqueeze(0).float(), size=(tgt_canvas_h, canvas_w),
        mode="bilinear", align_corners=False,
    ).squeeze(0)

    # Assemble canvas: target on top, reference on bottom (FSS-SAM3 best config)
    canvas = torch.zeros(C, canvas_h, canvas_w, dtype=ref_resized.dtype)
    canvas[:, :tgt_canvas_h, :canvas_w] = tgt_resized
    canvas[:, tgt_canvas_h:, :canvas_w] = ref_resized

    # Map reference bbox to canvas coordinates
    sx = canvas_w / ref_w
    sy = ref_canvas_h / ref_h
    x1, y1, x2, y2 = ref_bbox[:4]
    canvas_bbox = np.array([
        x1 * sx,
        y1 * sy + tgt_canvas_h,  # Offset by target region height
        x2 * sx,
        y2 * sy + tgt_canvas_h,
    ], dtype=np.float32)

    tgt_region = (0, 0, canvas_w, tgt_canvas_h)  # (x, y, w, h)
    return canvas, canvas_bbox, tgt_region


def _canvas_predict(
    model: SAM3,
    ref_sample: Sample,
    target_samples: list[Sample],
    text: str | None = None,
    use_bbox: bool = True,
    split_ratio: float = 0.6,
) -> list[dict]:
    """Predict using the FSS-SAM3 unified canvas approach.

    Stitches ref+target into a single image, runs SAM3 in CLASSIC mode
    with the support bbox (mapped to canvas coords) and optional text.
    Extracts predictions from the target region of the canvas.

    Args:
        model: SAM3 model instance.
        ref_sample: Reference sample with image, bboxes, categories.
        target_samples: List of target samples (images only).
        text: Text prompt. None = no text prompt used.
        use_bbox: Whether to include the support bbox as geometric prompt.
        split_ratio: Fraction of canvas for reference image.
    """
    ref_image = ref_sample.image
    ref_bbox = ref_sample.bboxes[0] if ref_sample.bboxes is not None else None
    if ref_bbox is None and use_bbox:
        use_bbox = False

    original_mode = model.prompt_mode
    model.prompt_mode = Sam3PromptMode.CLASSIC

    predictions = []
    for tgt in target_samples:
        tgt_image = tgt.image
        tgt_h, tgt_w = tgt_image.shape[-2:]

        canvas, canvas_bbox, tgt_region = _create_canvas(
            ref_image, tgt_image, ref_bbox if ref_bbox is not None else np.zeros(4),
            split_ratio=split_ratio,
        )

        # Build the canvas sample with appropriate prompts
        canvas_sample = Sample(image=canvas)
        if use_bbox:
            canvas_sample.bboxes = np.array([canvas_bbox])
            canvas_sample.categories = [text or "visual"]
            canvas_sample.category_ids = np.array([0])
        elif text:
            canvas_sample.categories = [text]
            canvas_sample.category_ids = np.array([0])

        # Fit with category for CLASSIC mode (enables text-only)
        if not use_bbox and text:
            model.fit(Sample(categories=[text], category_ids=[0]))
        else:
            model.category_mapping = None  # Clear any prior fit

        preds = model.predict([canvas_sample])
        pred = preds[0]

        # Extract boxes that fall within the target region
        tx, ty, tw, th = tgt_region
        pred_boxes = pred["pred_boxes"][:, :4].cpu()
        if pred_boxes.shape[0] > 0:
            # Box centers
            cx = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
            cy = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
            in_target = (cx >= tx) & (cx < tx + tw) & (cy >= ty) & (cy < ty + th)
            scores = pred["pred_boxes"][:, 4].cpu() if pred["pred_boxes"].shape[1] > 4 else torch.ones(len(pred_boxes))
            target_boxes = pred_boxes[in_target]
            target_scores = scores[in_target]

            if target_boxes.shape[0] > 0:
                # Remap boxes from canvas coords to original target image coords
                scale_x = tgt_w / tw
                scale_y = tgt_h / th
                remapped = target_boxes.clone()
                remapped[:, 0] = (target_boxes[:, 0] - tx) * scale_x
                remapped[:, 1] = (target_boxes[:, 1] - ty) * scale_y
                remapped[:, 2] = (target_boxes[:, 2] - tx) * scale_x
                remapped[:, 3] = (target_boxes[:, 3] - ty) * scale_y
                # Clamp to image bounds
                remapped[:, 0].clamp_(min=0)
                remapped[:, 1].clamp_(min=0)
                remapped[:, 2].clamp_(max=tgt_w)
                remapped[:, 3].clamp_(max=tgt_h)
                pred_with_scores = torch.cat([remapped, target_scores.unsqueeze(1)], dim=1)
                predictions.append({"pred_boxes": pred_with_scores})
            else:
                predictions.append({"pred_boxes": torch.empty(0, 5)})
        else:
            predictions.append({"pred_boxes": torch.empty(0, 5)})

    model.prompt_mode = original_mode
    return predictions


def run_benchmark(
    dataset,
    dataset_name: str,
    categories: list[str],
    model: SAM3,
    max_targets: int = 5,
    shuffle: bool = False,
    seed: int = 42,
    phase1: bool = False,
    phase2: bool = False,
    phase3: bool = False,
    phase4: bool = False,
    phase5: bool = False,
    clip_encoder: CLIPCropEncoder | None = None,
    clip_encoder_aligned: CLIPCropEncoderAligned | None = None,
    model_dsb: SAM3 | None = None,
):
    """Run mode comparison and return summary rows.

    Args:
        phase1: If True, also run Phase 1 experiments (mask/scale/empty text).
        phase2: If True, also run Phase 2 experiments (tile geometry, CLIP crop).
        phase3: If True, also run Phase 3 experiments (drop_spatial_bias).
        phase4: If True, also run Phase 4 experiments (multi-point, mask-pool, feat match).
        phase5: If True, also run Phase 5 experiments (FSS-SAM3 canvas).
        clip_encoder: Pre-loaded CLIP ViT-B encoder for Exp 5.
        clip_encoder_aligned: Pre-loaded CLIP ViT-L aligned encoder for Exp 5b.
        model_dsb: SAM3 model with drop_spatial_bias=True for Phase 3.
    """
    rows = []

    for cat in categories:
        ref_samples, tgt_samples = get_reference_and_targets(
            dataset, cat, max_targets=max_targets, shuffle=shuffle, seed=seed,
        )
        if not ref_samples:
            print(f"  [{dataset_name}] {cat}: no reference samples, skipping")
            continue
        if not tgt_samples:
            print(f"  [{dataset_name}] {cat}: no target samples, skipping")
            continue

        ref_sample = ref_samples[0]
        n_bboxes = len(ref_sample.bboxes) if ref_sample.bboxes is not None else 0
        print(f"  [{dataset_name}] {cat}: ref has {n_bboxes} bbox(es), {len(tgt_samples)} targets")

        tgt_clean = strip_annotations(tgt_samples)

        # Define modes: (mode_name, prompt_mode, override_text, post_fit_patch, use_dsb)
        # post_fit_patch is a callable(model) applied after fit() to manipulate cached features
        # use_dsb: if True, use model_dsb (drop_spatial_bias=True) instead of default model
        modes: list[tuple[str, Sam3PromptMode | None, str | None, str | None, bool]] = [
            ("Text-Only", Sam3PromptMode.CLASSIC, None, None, False),
            ('VE+"visual"', Sam3PromptMode.VISUAL_EXEMPLAR, "visual", None, False),
            ("VE+real", Sam3PromptMode.VISUAL_EXEMPLAR, None, None, False),
        ]

        if phase1:
            modes.extend([
                ("VE+mask-text", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mask_text", False),
                ("VE+scale0.1", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "scale_0.1", False),
                ("VE+scale0.01", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "scale_0.01", False),
                ('VE+""', Sam3PromptMode.VISUAL_EXEMPLAR, "", None, False),
            ])

        if phase2:
            modes.extend([
                ("VE+tile-geo", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "tile_geo", False),
                ("VE+tile+mask", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "tile_geo_mask_text", False),
                ("VE+clip-crop", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "clip_crop", False),
                ("VE+clipL-aln", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "clip_l_aligned", False),
            ])

        if phase3 and model_dsb is not None:
            modes.extend([
                ('DSB+"visual"', Sam3PromptMode.VISUAL_EXEMPLAR, "visual", None, True),
                ("DSB+real", Sam3PromptMode.VISUAL_EXEMPLAR, None, None, True),
                ("DSB+tile-geo", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "tile_geo", True),
                ("DSB+tile+mask", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "tile_geo_mask_text", True),
                ("DSB+mask-text", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mask_text", True),
            ])

        if phase4:
            modes.extend([
                # Method C: Multi-point from mask
                ('MP16+"visual"', Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mp16", False),
                ('MP32+"visual"', Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mp32", False),
                ("MP16+mask-text", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mp16_mask", False),
                # Method D: Mask-pooled backbone features
                ("MaskPool", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mask_pool", False),
                ("MaskPool+tile", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mask_pool_tile", False),
                ("MaskPool+mask", Sam3PromptMode.VISUAL_EXEMPLAR, "visual", "mask_pool_mask", False),
                # Method E: Feature matching
                ("FeatMatch", None, None, "feat_match", False),
            ])

        if phase5:
            modes.extend([
                ('Canvas+"visual"', None, "visual", "canvas_visual", False),
                ("Canvas+real", None, None, "canvas_real", False),
                ("Canvas+text-only", None, None, "canvas_text_only", False),
            ])

        for mode_name, prompt_mode, override_text, patch, use_dsb in modes:
            active_model = model_dsb if use_dsb else model

            # Method E: feature matching uses a completely separate code path
            if patch == "feat_match":
                preds = _feature_match_predict(active_model, ref_sample, tgt_clean)
            elif patch and patch.startswith("canvas"):
                # Phase 5: FSS-SAM3 canvas approach
                if patch == "canvas_visual":
                    canvas_text = "visual"
                    canvas_bbox = True
                elif patch == "canvas_real":
                    canvas_text = cat
                    canvas_bbox = True
                else:  # canvas_text_only
                    canvas_text = cat
                    canvas_bbox = False
                preds = _canvas_predict(
                    active_model, ref_sample, tgt_clean,
                    text=canvas_text, use_bbox=canvas_bbox,
                )
            else:
                active_model.prompt_mode = prompt_mode

                if prompt_mode == Sam3PromptMode.CLASSIC:
                    active_model.fit(Sample(categories=[cat], category_ids=[0]))
                else:
                    ref = copy.deepcopy(ref_sample)
                    if override_text is not None:
                        ref.categories = [override_text] * len(ref.categories)

                    # Pre-fit patches (Method C: multi-point from mask)
                    if patch and patch.startswith("mp"):
                        n_pts = 32 if "32" in patch else 16
                        ref = _create_multipoint_sample(ref, n_points=n_pts)

                    active_model.fit(ref)

                # Apply post-fit patches
                if patch == "mask_text":
                    _patch_text_mask_zero(active_model)
                elif patch == "scale_0.1":
                    _patch_text_scale(active_model, 0.1)
                elif patch == "scale_0.01":
                    _patch_text_scale(active_model, 0.01)
                elif patch == "tile_geo":
                    _patch_tile_geometry(active_model)
                elif patch == "tile_geo_mask_text":
                    _patch_tile_geo_mask_text(active_model)
                elif patch == "clip_crop" and clip_encoder is not None:
                    _patch_clip_crop(active_model, ref_sample, clip_encoder)
                elif patch == "clip_l_aligned" and clip_encoder_aligned is not None:
                    _patch_clip_crop(active_model, ref_sample, clip_encoder_aligned)
                elif patch == "mp16_mask":
                    _patch_text_mask_zero(active_model)
                elif patch == "mask_pool":
                    _patch_mask_pooled_features(active_model, ref_sample)
                elif patch == "mask_pool_tile":
                    _patch_mask_pooled_features(active_model, ref_sample)
                    _patch_tile_geometry(active_model)
                elif patch == "mask_pool_mask":
                    _patch_mask_pooled_features(active_model, ref_sample)
                    _patch_text_mask_zero(active_model)

                preds = active_model.predict(tgt_clean)

            total_tp, total_fp, total_gt = 0, 0, 0
            all_ious = []
            for pred, sample in zip(preds, tgt_samples, strict=True):
                tp, fp, n_gt, miou = compute_tp_fp(pred, sample, cat)
                total_tp += tp
                total_fp += fp
                total_gt += n_gt
                if miou > 0:
                    all_ious.append(miou)

            total_det = total_tp + total_fp
            prec = total_tp / max(total_det, 1)
            rec = total_tp / max(total_gt, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-6)
            avg_iou = float(np.mean(all_ious)) if all_ious else 0.0

            rows.append({
                "dataset": dataset_name,
                "category": cat,
                "mode": mode_name,
                "tp": total_tp,
                "fp": total_fp,
                "gt": total_gt,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "iou": avg_iou,
                "n_images": len(tgt_samples),
            })

    return rows


def print_summary(rows: list[dict], title: str = "") -> None:
    """Print a formatted summary table."""
    if title:
        print(f"\n{'=' * 90}")
        print(f"  {title}")
        print(f"{'=' * 90}")

    header = (
        f"{'Dataset':<10} {'Category':<16} {'Mode':<16} "
        f"{'TP':>4} {'FP':>4} {'GT':>4} "
        f"{'Prec':>6} {'Rec':>6} {'F1':>6} {'IoU':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['dataset']:<10} {r['category']:<16} {r['mode']:<16} "
            f"{r['tp']:>4} {r['fp']:>4} {r['gt']:>4} "
            f"{r['precision']:>6.3f} {r['recall']:>6.3f} {r['f1']:>6.3f} {r['iou']:>6.3f}"
        )

    # Per-mode aggregates
    mode_names = list(dict.fromkeys(r["mode"] for r in rows))
    print()
    print(f"{'--- Aggregated (macro avg) ---':^{len(header)}}")
    agg_header = f"{'Mode':<16} {'Avg Prec':>9} {'Avg Rec':>9} {'Avg F1':>9} {'Avg IoU':>9}"
    print(agg_header)
    print("-" * len(agg_header))
    for mode in mode_names:
        mr = [r for r in rows if r["mode"] == mode]
        avg_p = np.mean([r["precision"] for r in mr])
        avg_r = np.mean([r["recall"] for r in mr])
        avg_f = np.mean([r["f1"] for r in mr])
        avg_i = np.mean([r["iou"] for r in mr])
        print(f"{mode:<16} {avg_p:>9.3f} {avg_r:>9.3f} {avg_f:>9.3f} {avg_i:>9.3f}")


def main():
    parser = argparse.ArgumentParser(description="SAM3 mode comparison benchmark")
    parser.add_argument(
        "--dataset", choices=["perseg", "lvis", "both"], default="both",
        help="Which dataset to benchmark",
    )
    parser.add_argument("--categories", nargs="+", default=None, help="LVIS categories (default: 4 defaults)")
    parser.add_argument("--perseg-categories", nargs="+", default=None, help="PerSeg categories")
    parser.add_argument("--max-targets", type=int, default=5, help="Target images per category")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle reference/target selection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffle")
    parser.add_argument("--confidence", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="Device (auto-detect if omitted)")
    parser.add_argument("--perseg-root", type=str, default=str(PERSEG_ROOT))
    parser.add_argument("--lvis-root", type=str, default=str(LVIS_ROOT))
    parser.add_argument("--phase1", action="store_true", help="Run Phase 1 experiments (mask/scale/empty text)")
    parser.add_argument("--phase2", action="store_true", help="Run Phase 2 experiments (tile geometry, CLIP crop)")
    parser.add_argument("--phase3", action="store_true", help="Run Phase 3 experiments (drop_spatial_bias)")
    parser.add_argument("--phase4", action="store_true", help="Run Phase 4 experiments (multi-point, mask-pool, feat match)")
    parser.add_argument("--phase5", action="store_true", help="Run Phase 5 experiments (FSS-SAM3 canvas)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model once
    model = SAM3(device=device, confidence_threshold=args.confidence, precision="fp32")
    print("SAM3 model loaded.")

    clip_encoder = None
    clip_encoder_aligned = None
    if args.phase2:
        print("Loading CLIP ViT-B/16 for crop encoding...")
        clip_encoder = CLIPCropEncoder(device=device)
        print("Loading CLIP ViT-L/14 (aligned) for crop encoding...")
        clip_encoder_aligned = CLIPCropEncoderAligned(device=device, sam3_model=model)
        print("CLIP encoders loaded.")

    model_dsb = None
    if args.phase3:
        print("Loading SAM3 with drop_spatial_bias=True...")
        model_dsb = SAM3(
            device=device, confidence_threshold=args.confidence,
            precision="fp32", drop_spatial_bias=True,
        )
        print("SAM3 (drop_spatial_bias) loaded.")
    print()

    all_rows = []

    # ── PerSeg ──
    if args.dataset in ("perseg", "both"):
        perseg_cats = args.perseg_categories or PERSEG_CATEGORIES
        print(f"Loading PerSeg ({len(perseg_cats)} categories)...")
        perseg_ds = PerSegDataset(root=args.perseg_root, categories=perseg_cats, n_shots=1)
        print(f"  Available: {perseg_ds.categories}, total samples: {len(perseg_ds)}")

        rows = run_benchmark(
            perseg_ds, "PerSeg", perseg_ds.categories, model,
            max_targets=args.max_targets, shuffle=args.shuffle, seed=args.seed,
            phase1=args.phase1, phase2=args.phase2, phase3=args.phase3,
            phase4=args.phase4, phase5=args.phase5,
            clip_encoder=clip_encoder, clip_encoder_aligned=clip_encoder_aligned,
            model_dsb=model_dsb,
        )
        all_rows.extend(rows)
        print_summary(rows, "PerSeg Results")

    # ── LVIS ──
    if args.dataset in ("lvis", "both"):
        lvis_cats = args.categories or LVIS_CATEGORIES
        print(f"\nLoading LVIS ({len(lvis_cats)} categories, INSTANCE mode)...")
        lvis_ds = LVISDataset(
            root=args.lvis_root, categories=lvis_cats, n_shots=1,
            annotation_mode=LVISAnnotationMode.INSTANCE,
        )
        print(f"  Available: {lvis_ds.categories}, total samples: {len(lvis_ds)}")

        rows = run_benchmark(
            lvis_ds, "LVIS", lvis_ds.categories, model,
            max_targets=args.max_targets, shuffle=args.shuffle, seed=args.seed,
            phase1=args.phase1, phase2=args.phase2, phase3=args.phase3,
            phase4=args.phase4, phase5=args.phase5,
            clip_encoder=clip_encoder, clip_encoder_aligned=clip_encoder_aligned,
            model_dsb=model_dsb,
        )
        all_rows.extend(rows)
        print_summary(rows, "LVIS Results")

    # ── Combined ──
    if args.dataset == "both" and all_rows:
        print_summary(all_rows, "Combined Results (PerSeg + LVIS)")


if __name__ == "__main__":
    main()
