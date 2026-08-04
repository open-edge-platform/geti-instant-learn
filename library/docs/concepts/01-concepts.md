# Core Concepts

This page covers the data types every model in the library accepts and returns.
The API is deliberately small: two containers in, one container out.

## Visual Prompting

Traditional detection needs a model trained on thousands of labelled examples
of the thing you care about. Visual prompting removes that step: you show the
model one or a few examples — a mask, a box, a click, or just a word — and it
finds similar objects in new images, so adding a new category is a runtime
operation rather than a training run.

## Reference and Target

Every model follows the same two-phase flow.

| Phase  | Method      | Input               | Meaning                          |
| ------ | ----------- | ------------------- | -------------------------------- |
| Prompt | `fit()`     | reference sample(s) | "here is what I am looking for"  |
| Infer  | `predict()` | target sample(s)    | "find it in these images"        |

```python
model.fit(reference_sample)          # what to look for
predictions = model.predict(target)  # where it is
```

`fit()` is idempotent — calling it again replaces the previous prompt rather
than accumulating. Models that require a prompt raise `ModelNotFittedError` if
you call `predict()` first. Zero-shot models such as `GroundedSAM` and `SAM3`
take a category name instead and can skip `fit()` entirely.

## Sample

`Sample` is the single input type. It is backend-neutral: **every array field is
numpy**, and the module imports no torch at all. That is what lets an
OpenVINO-only deployment use the same data types as a PyTorch one.

```python
from instantlearn.data import Category, Sample

# From arrays
sample = Sample(image=image_hwc, masks=masks_nhw, categories=[Category(0, "cat")])

# Or from paths — loaded on construction
sample = Sample(image_path="cat.jpg", mask_paths="cat_mask.png")

# Text-only prompt, no image needed
sample = Sample(categories=[Category(0, "elephant")])
```

Key fields:

| Field        | Shape            | Notes                    |
| ------------ | ---------------- | ------------------------ |
| `image`      | `(H, W, C)`      | **HWC**, uint8 or float32 |
| `masks`      | `(N, H, W)`      | bool or uint8            |
| `bboxes`     | `(N, 4)`         | float32, xyxy            |
| `points`     | `(N, K, 2)`      | float32                  |
| `categories` | `list[Category]` | id + label per instance  |

The layout matters: `Sample.image` is **HWC**, which is what OpenCV and PIL give
you. Torch models transpose to CHW internally, so you never do it yourself.

`Category` pairs an integer id with a name. It is frozen, so it is hashable and
safe to reuse.

## Prediction

`predict()` returns `list[Prediction]` — one per input sample, in order. Like
`Sample`, it is pure numpy.

```python
for prediction in model.predict([target_a, target_b]):
    prediction.masks        # (N, H, W) bool/uint8
    prediction.scores       # (N,) float32
    prediction.label_ids    # (N,) int32
    prediction.label_names  # (N,) str
    prediction.boxes        # (N, 4) float32 xyxy, or None
```

The arrays are already numpy at frame resolution, so there is nothing to
convert — no `.cpu()`, no `.numpy()`, and no dictionary keys to remember.

## Batch

`Batch` wraps a `list[Sample]` and adds batch-level accessors. It is **not**
required: `fit()` and `predict()` accept a single `Sample`, a `list[Sample]`, or
a `Batch` interchangeably.

```python
model.predict(sample)                  # one
model.predict([sample_a, sample_b])    # several
model.predict(Batch.collate(samples))  # equivalent
```

Prefer a plain list unless you specifically want `Batch`'s accessors — it is the
simpler form and the one the rest of the documentation uses.

## Models

Every model exposes the same four members, whatever its backend:

```python
model.card()        # static capabilities — see Architecture
model.backend       # Backend.TORCH or Backend.OPENVINO
model.fit(...)      # accept a prompt
model.predict(...)  # infer
```

That uniformity is the point of the contract: swapping `SAM3` for
`SAM3OpenVINO` changes the runtime, not your code.

## Next Steps

- [Architecture](02-architecture.md) — the class hierarchy and data flow
- [Custom Models](../how-to-guides/03-custom-models.md) — implement your own
- [Quick Start](../02-quick-start.md) — run all of this end to end
