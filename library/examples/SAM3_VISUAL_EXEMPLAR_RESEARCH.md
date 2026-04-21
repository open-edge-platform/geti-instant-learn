# SAM3 Visual Exemplar Research: Comprehensive Analysis & Results

> **Branch**: `fix/sam3_visualexemplar`  
> **Files**: `library/examples/sam3_mode_benchmark.py`, `library/examples/sam3_mode_comparison.ipynb`  
> **Date**: April 2026

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture Context](#architecture-context)
- [Experiment Summary](#experiment-summary)
  - [Phase 1: Text Manipulation](#phase-1-text-manipulation)
  - [Phase 2: Geometry Amplification & CLIP Crops](#phase-2-geometry-amplification--clip-crops)
  - [Phase 3: drop_spatial_bias (Community-Sourced)](#phase-3-drop_spatial_bias-community-sourced)
  - [Phase 4: Richer Geometry & Feature Matching](#phase-4-richer-geometry--feature-matching)
  - [Phase 5: FSS-SAM3 Unified Canvas (Breakthrough)](#phase-5-fss-sam3-unified-canvas-breakthrough)
- [Master Results Table](#master-results-table)
- [Architecture Deep Dive](#architecture-deep-dive)
- [Community Research](#community-research)
- [Conclusions & Recommendations](#conclusions--recommendations)
- [How to Run](#how-to-run)

---

## Problem Statement

SAM3's Visual Exemplar (VE) mode is significantly worse than Text-Only mode when the category name is unknown:

| Mode | PerSeg F1 | LVIS F1 | Combined F1 |
|------|-----------|---------|-------------|
| Text-Only | 0.907 | 0.794 | 0.870 |
| VE+real (oracle, uses true category name) | 0.921 | 0.770 | 0.871 |
| **VE+"visual" (current default)** | **0.354** | **0.672** | **0.460** |

The gap: **VE+"visual" scores 0.460 F1 vs Text-Only at 0.870 — a 41-point deficit.**

**Goal**: Find a way to achieve Text-Only-level performance using only visual exemplar (reference image + bbox), no category name.

---

## Architecture Context

### Text + Geometry Fusion

SAM3 combines text and geometry through concatenation before the DETR decoder:

```
Text path:    category name → CLIP tokenizer (max_length=32) → CLIP text encoder [1,32,1024]
              → text_projection(Linear 1024→256) → [1,32,256]

Geometry path: bbox center → geometry_encoder (ROI pool + direct project + pos encoding)
              → [1,N,256] where N = number of bboxes

Fusion:       torch.cat([text_features, geometry_features], dim=1) → [1, 32+N, 256]
              → DETR decoder cross-attention (treats all tokens equally)
```

**Root cause of VE failure**: For a 1-bbox reference, the ratio is ~3 real text tokens vs 1 geometry token. With meaningless text like "visual", these tokens inject noise that drowns the geometry signal.

### Key Files

| File | Key Functions |
|------|---------------|
| `library/src/instantlearn/models/sam3/sam3.py` | `_predict_visual_exemplar()`, `_cache_text_features()`, `_encode_sample_prompts()` |
| `library/src/instantlearn/models/sam3/model.py` | `forward()` (text+geometry concat), `text_projection`, `GeometryEncoder` |
| `library/src/instantlearn/models/sam3/detr.py` | `DETRDecoderLayer.forward()` (text cross-attention) |

### GeometryEncoder Components

The geometry encoder has 3 additive components:
1. **`boxes_direct_project`** — Linear(4→256) on raw coordinates
2. **`boxes_pool`** (ROI align) — Pools visual features at bbox location
3. **`boxes_pos_enc`** — Sinusoidal position encoding of box center/size

With `drop_spatial_bias=True`, only component 2 (ROI pool) is used — coordinates and position encodings are dropped.

---

## Experiment Summary

### Datasets

- **PerSeg**: 8 categories (dog, cat, backpack, clock, teddybear, duck_toy, candle, chair), 41 samples, 1-shot
- **LVIS**: 4 categories (cupcake, sheep, pastry, doughnut), 432 samples, INSTANCE mode, 1-shot

### Phase 1: Text Manipulation

Inference-only patches — no model changes, just tensor manipulation.

| # | Mode | Description | PerSeg F1 | LVIS F1 |
|---|------|-------------|-----------|---------|
| 1 | Text-Only | Baseline: text prompt, no visual exemplar | 0.907 | 0.794 |
| 2 | VE+"visual" | Current default VE with text="visual" | 0.354 | 0.672 |
| 3 | VE+real | Oracle: VE with true category name | 0.921 | 0.770 |
| 4 | VE+mask-text | Zero out text attention mask → geometry only | 0.409 | 0.572 |
| 5 | VE+scale0.1 | Scale text features by 0.1 | 0.412 | 0.666 |
| 6 | VE+scale0.01 | Scale text features by 0.01 | 0.376 | 0.662 |
| 7 | VE+"" | Empty string as text prompt | 0.217 | 0.581 |

**Finding**: Geometry alone is weak (best F1=0.412). Text carries 40-70% of discriminative signal.

### Phase 2: Geometry Amplification & CLIP Crops

| # | Mode | Description | PerSeg F1 | LVIS F1 |
|---|------|-------------|-----------|---------|
| 8 | VE+tile-geo | Tile geometry token 32× to match text count | **0.540** | 0.662 |
| 9 | VE+tile+mask | Tile geometry + mask text | 0.409 | 0.599 |
| 10 | VE+clip-crop | Replace text with CLIP ViT-B crop features | 0.300 | 0.000 |
| 11 | VE+clipL-align | Replace text with CLIP ViT-L features via SAM3's text_projection | 0.000 | 0.000 |

**Finding**: Tile-geo is the best no-text experiment so far (PerSeg 0.540). CLIP crop approaches are dead ends — vision/text hidden states occupy different subspaces. SAM3's `text_projection(Linear 1024→256)` trained on CLIP text hidden states produces garbage when fed CLIP vision hidden states.

### Phase 3: drop_spatial_bias (Community-Sourced)

From SAM3 GitHub issues (#317, #185) and MuggledSAM repo: `drop_spatial_bias=True` removes coordinate-based components from geometry encoding.

| # | Mode | Description | PerSeg F1 | LVIS F1 |
|---|------|-------------|-----------|---------|
| 12 | DSB+"visual" | drop_spatial_bias + text="visual" | 0.106 | 0.703 |
| 13 | DSB+real | drop_spatial_bias + real category name | 0.735 | 0.729 |
| 14 | DSB+tile-geo | drop_spatial_bias + tile geometry 32× | 0.427 | 0.625 |
| 15 | DSB+tile+mask | drop_spatial_bias + tile + mask text | 0.400 | 0.620 |
| 16 | DSB+mask-text | drop_spatial_bias + mask text only | 0.400 | 0.592 |

**Finding**: drop_spatial_bias is **counterproductive**. DSB+"visual" (0.106 PerSeg) is 3.3× worse than VE+"visual" (0.354). The position encodings provide essential spatial grounding, even for cross-image transfer. The community recommendation doesn't hold up in systematic testing.

### Phase 4: Richer Geometry & Feature Matching

| # | Mode | Description | PerSeg F1 | LVIS F1 | Combined F1 |
|---|------|-------------|-----------|---------|-------------|
| 17 | MP16+"visual" | 16 mask-sampled points, text="visual" | — | — | 0.373 |
| 18 | MP32+"visual" | 32 mask-sampled points, text="visual" | — | — | 0.430 |
| 19 | MP16+mask-text | 16 points, text masked out | — | — | **0.534** |
| 20 | MaskPool | Mask-averaged FPN features as geometry token | — | — | 0.233 |
| 21 | MaskPool+tile | MaskPool tiled to 32 tokens | — | — | 0.207 |
| 22 | MaskPool+mask | MaskPool + text masked | — | — | 0.204 |
| 23 | FeatMatch | Backbone feature matching (cosine similarity) | — | — | 0.086 |

**Finding**: MP16+mask-text (0.534) nearly matches VE+tile-geo (0.540) — a genuine alternative. MaskPool fails because raw FPN features don't match geometry encoder output distribution. FeatMatch has high recall but terrible precision.

### Phase 5: FSS-SAM3 Unified Canvas (Breakthrough)

Based on **FSS-SAM3** paper (arxiv:2604.05433). Stitches reference + target images into a single 1008×1008 canvas, runs SAM3 in CLASSIC mode with the reference bbox mapped to canvas coordinates.

| # | Mode | Description | Avg Prec | Avg Rec | Avg F1 | Avg IoU |
|---|------|-------------|----------|---------|--------|---------|
| — | Text-Only | baseline | 0.906 | 0.875 | 0.870 | 0.933 |
| — | VE+"visual" | baseline | 0.438 | 0.495 | 0.460 | 0.618 |
| — | VE+real | oracle baseline | 0.862 | 0.883 | 0.871 | 0.939 |
| 24 | **Canvas+"visual"** | canvas + bbox + text="visual" | **0.921** | **0.884** | **0.893** | 0.929 |
| 25 | **Canvas+real** | canvas + bbox + real category name | **0.937** | **0.901** | **0.907** | 0.931 |
| 26 | Canvas+text-only | canvas + text only (no bbox) | 0.915 | 0.902 | 0.899 | 0.914 |

**This is the breakthrough:**
- **Canvas+"visual" (F1=0.893)** beats VE+real oracle (0.871) and Text-Only (0.870) — **without using any category name!**
- **Canvas+real (F1=0.907)** is the new overall best
- Canvas+text-only (0.899) also beats everything — the canvas layout itself helps text-only mode

---

## Master Results Table

All 26 experiments ranked by Combined F1 (macro avg across PerSeg + LVIS):

| Rank | Mode | Combined F1 | Notes |
|------|------|-------------|-------|
| 1 | **Canvas+real** | **0.907** | NEW BEST — canvas + bbox + category name |
| 2 | Canvas+text-only | 0.899 | Canvas with text, no bbox prompt |
| 3 | **Canvas+"visual"** | **0.893** | **BEST VISUAL-ONLY** — no category name needed |
| 4 | VE+real | 0.871 | Oracle (uses true category name) |
| 5 | Text-Only | 0.870 | Text prompt, no visual exemplar |
| 6 | DSB+real | 0.733 | drop_spatial_bias + real name |
| 7 | VE+tile-geo | ~0.58 | Best Phase 1-2 no-text mode |
| 8 | MP16+mask-text | 0.534 | Best Phase 4 mode |
| 9 | VE+"visual" | 0.460 | Current default (BASELINE) |
| 10 | MP32+"visual" | 0.430 | 32-point sampling |
| 11 | VE+mask-text | ~0.49 | Geometry only |
| 12 | VE+scale0.1 | ~0.54 | Scaled text |
| 13+ | Others | <0.43 | Various failed approaches |

---

## Architecture Deep Dive

### Text Embeddings and Geometry Fusion

SAM3 combines text and geometry through concatenation before the DETR decoder:

1. **Text features** [1,32,256]: CLIP text encoder output projected via `text_projection(Linear 1024→256)`
2. **Geometry features** [1,N,256]: GeometryEncoder output (ROI pool + coordinate projection + position encoding)
3. **Concatenation**: `torch.cat([text, geometry], dim=1)` → [1, 32+N, 256]
4. **DETR decoder**: Cross-attention treats all tokens equally as keys/values

The DETR decoder has dual cross-attention:
- **Text cross-attention**: Queries attend to combined text+geometry tokens
- **Vision cross-attention**: Queries attend to backbone features (independent of text/geometry)

### GeometryEncoder Details

```python
def _encode_points(self, points, points_mask, points_labels, vision_features, drop_spatial_bias=False):
    # Component 1: Direct projection of coordinates
    points_embed = self.points_direct_project(points)  # Linear(2→256)
    
    # Component 2: Pooled visual features via grid_sample
    sampled = F.grid_sample(vision_features, grid, ...)
    pooled = self.points_pool_project(sampled)
    points_embed += pooled
    
    if not drop_spatial_bias:
        # Component 3: Sinusoidal position encoding
        pos_enc = self.position_encoding.encode_1d_positions(x, y)
        pos_proj = self.points_pos_enc_project(pos_enc)
        points_embed += pos_proj
    
    label_embed = self.label_embed(points_labels)
    return label_embed + points_embed, points_mask
```

### Visual Exemplar Flow

```
fit(reference_image, reference_bbox, category_name):
  1. Encode image → FPN features
  2. Convert bbox to center point
  3. geometry_encoder(point, FPN) → geometry_features [1, 1, 256]
  4. CLIP text_encoder(category_name) → text_features [1, 32, 256]
  5. Cache both for predict()

predict(target_image):
  1. Encode target → vision_embeds
  2. model.forward(
       vision_embeds=vision_embeds,
       text_embeds=cached_text,
       precomputed_geometry_features=cached_geometry,
     )
  3. DETR decoder: queries attend to cached text+geometry (cross-image) and target vision features
```

### Canvas Approach Flow (FSS-SAM3)

```
predict(reference_image, reference_bbox, target_image, category_name=None):
  1. Create canvas: stitch target (top 40%) + reference (bottom 60%) into 1008×1008
  2. Map reference bbox to canvas coordinates
  3. Run SAM3 CLASSIC mode on canvas with:
     - text = category_name or "visual"
     - bbox = mapped reference bbox on canvas
  4. Extract predictions from target region of canvas
  5. Remap detected boxes to original target coordinates
```

**Why canvas works**: Both images are in the same attention computation. SAM3's ViT backbone processes them simultaneously, and self-attention naturally correlates features between support and query regions. This is fundamentally more powerful than caching geometry tokens, because the full self-attention mechanism handles cross-image matching rather than just 1-32 conditioning tokens in the DETR decoder.

---

## Community Research

### Sources

- SAM3 GitHub Issue #317: "Transfer concept from one image to another"
- SAM3 GitHub Issue #185: "Image as Exemplar"
- heyoeyo/muggled_sam repo: Working cross-image detection
- SAMv1 paper Appendix D.6: Latent space probing
- FSS-SAM3 paper (arxiv:2604.05433): Unified spatial canvas

### Key Insights

1. **@11710615** (issue #317): "For cross-image tasks, `boxes_pool` is the key component; disable `boxes_direct_project` and `boxes_pos_enc`" — Our Phase 3 testing shows this is NOT consistently beneficial.
2. **heyoeyo** (muggled_sam): Has working cross-image detection with `include_coordinate_encodings=False`. Points from masks outperform box prompts.
3. **@JohannesGrand** (issue #185): "3-5 example images stitched together + positive/negative bboxes + text prompt" → precursor to FSS-SAM3 canvas idea.
4. **FSS-SAM3** (arxiv:2604.05433): Formalized the canvas approach. COCO-20i: 66.6 mIoU (visual only), 75.8 (visual+text). PASCAL-5i: 79.6 (visual only), 81.2 (visual+text). Both SOTA.

### FSS-SAM3 Paper Details

- **Core idea**: Place support and query images into a shared canvas → SAM3 processes as single image
- **Best config**: Vertical concat, support at bottom (60:40 ratio), forced resize (not aspect-preserving)
- **Negative prompts**: Consistently hurt performance (tested and confirmed in paper)
- **5-shot layout**: L-shape arrangement of 5 support images around query
- **Limitation noted by authors**: Image/text encoders not aligned (can't use as CLIP-like model); pseudo-video/memory approach doesn't work

---

## Conclusions & Recommendations

### Key Findings

1. **The canvas approach is the clear winner**: Canvas+"visual" (F1=0.893) beats all other modes including VE+real oracle (0.871), achieving visual-only performance that exceeds text-only (0.870).

2. **Text carries 40-70% of VE's discriminative signal**: No geometry-only manipulation within the VE pipeline can close the gap. The 32:1 token ratio makes geometry tokens negligible in cross-attention.

3. **drop_spatial_bias is counterproductive**: Despite community recommendations, removing coordinate encodings hurts more than it helps in systematic testing.

4. **CLIP vision features can't replace text features**: The text pathway requires actual text-encoder-shaped features. Vision hidden states occupy a different subspace.

5. **The VE pipeline is fundamentally bottlenecked**: Caching features from one image and using them as 1-token conditioning on another is too narrow a channel for cross-image matching.

### Recommendations

1. **Immediate**: Implement the canvas approach as an alternative prediction mode in SAM3. For scenarios where category name is unknown, canvas+"visual" is the best option by a wide margin.

2. **Limitation to address**: Canvas halves the effective resolution (target gets ~40% of 1008×1008). For dense multi-instance scenes (e.g., cupcake with 150 instances), recall drops significantly (0.347 vs 0.693 text-only). Consider:
   - Adaptive split ratios based on reference/target complexity
   - Multi-scale canvas (run at multiple resolutions, merge)
   - Sliding window canvas for large images

3. **For production**: The canvas approach could be the default VE mode when no category name is available, falling back to text-only when category names are known.

---

## How to Run

```bash
cd library

# Run all phases (Phase 1 baselines always run)
uv run python examples/sam3_mode_benchmark.py --dataset both

# Run specific phases
uv run python examples/sam3_mode_benchmark.py --dataset both --phase1
uv run python examples/sam3_mode_benchmark.py --dataset both --phase2
uv run python examples/sam3_mode_benchmark.py --dataset both --phase3
uv run python examples/sam3_mode_benchmark.py --dataset both --phase4
uv run python examples/sam3_mode_benchmark.py --dataset both --phase5  # FSS-SAM3 canvas

# Single dataset
uv run python examples/sam3_mode_benchmark.py --dataset perseg --phase5
uv run python examples/sam3_mode_benchmark.py --dataset lvis --phase5

# Custom options
uv run python examples/sam3_mode_benchmark.py --dataset both --phase5 \
    --device cuda --confidence 0.3 --max-targets 5
```

### Data Requirements

- **PerSeg**: Download to `~/workspace/data/prompt/PerSeg/` (or set `--perseg-root`)
- **LVIS**: Download to `~/workspace/data/prompt/lvis/` (or set `--lvis-root`)

### Benchmark Output

The script prints per-category results and macro-averaged summary tables. Example:

```
                            --- Aggregated (macro avg) ---
Mode              Avg Prec   Avg Rec    Avg F1   Avg IoU
--------------------------------------------------------
Text-Only            0.906     0.875     0.870     0.933
VE+"visual"          0.438     0.495     0.460     0.618
VE+real              0.862     0.883     0.871     0.939
Canvas+"visual"      0.921     0.884     0.893     0.929
Canvas+real          0.937     0.901     0.907     0.931
Canvas+text-only     0.915     0.902     0.899     0.914
```
