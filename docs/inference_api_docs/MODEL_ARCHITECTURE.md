# Model Architecture: Modular Design with Extractor/Predictor

> **Related Documentation:**
> - [EXPORT_BEST_PRACTICES.md](EXPORT_BEST_PRACTICES.md) - Multi-backend export guide
> - [INFERENCE_MODEL.md](INFERENCE_MODEL.md) - Production deployment with InferenceModel
> - [WORKFLOW.md](WORKFLOW.md) - Complete workflow from development to production

---

## Design Philosophy

Split each model into **two explicit submodules**:
1. **`ReferenceExtractor`** - Extracts embeddings from reference images + priors
2. **`TargetPredictor`** - Predicts on target images using reference embeddings

This makes the two-stage process explicit and enables clean export to any backend.

---

## Base Classes

```python
# library/src/getiprompt/models/base.py
from abc import ABC, abstractmethod
import torch
from torch import nn
from getiprompt.types import Image, Priors, Results


class ReferenceExtractor(nn.Module, ABC):
    """Extracts reference representation from images and/or priors.

    This is a standalone module that can be exported independently.
    Responsible for the "learning" phase.

    Flexibility:
    - Images can be None for text-only models (DinoTxt, GroundedSAM)
    - Priors can be None for image-only models (future cases)
    - Returns generic tuple to support any reference data type
    """

    @abstractmethod
    def forward(
        self,
        images: list[Image] | None,
        priors: list[Priors] | None
    ) -> tuple[torch.Tensor | Any, torch.Tensor | Any | None]:
        """Extract reference representation.

        Args:
            images: Reference images (optional for text-only models)
            priors: Reference priors (masks, points, boxes, text)

        Returns:
            reference_data: Reference representation (embeddings, text features, etc.)
            auxiliary_data: Optional auxiliary data (masks, metadata, etc.)

        Examples:
            - Matcher: (image_embeddings, reference_masks)
            - DinoTxt: (text_embeddings, None)
            - GroundedSAM: (text_priors, None)
        """
        pass


class TargetPredictor(nn.Module, ABC):
    """Predicts on target images using reference representation.

    This is a standalone module that can be exported independently.
    Responsible for the "inference" phase.

    Flexibility:
    - reference_data can be embeddings, text, or any processed reference
    - auxiliary_data is optional (masks, metadata, etc.)
    - Supports any prediction task (segmentation, detection, classification)
    """

    @abstractmethod
    def forward(
        self,
        target_images: list[Image],
        reference_data: torch.Tensor | Any,
        auxiliary_data: torch.Tensor | Any | None = None
    ) -> Results:
        """Predict on target images using reference data.

        Args:
            target_images: Images to predict on
            reference_data: Pre-computed reference representation
            auxiliary_data: Optional auxiliary data from extractor

        Returns:
            Results with predictions (masks, boxes, classes, etc.)

        Examples:
            - Matcher: uses (embeddings, masks) for matching & segmentation
            - DinoTxt: uses (text_embeddings, None) for classification
            - GroundedSAM: uses (text_priors, None) for detection & segmentation
        """
        pass


class Model(nn.Module):
    """Base model composed of Extractor + Predictor.

    This wrapper provides convenient fit/predict API and manages state.
    The actual work is done by the submodules.

    Public API:
        - fit(): Extract and store reference embeddings
        - predict(): Predict on targets using stored embeddings
        - export(): Export model to ONNX/TensorRT/etc with automatic mode detection
    """

    def __init__(self):
        super().__init__()
        # Subclasses must initialize these
        self.extractor: ReferenceExtractor = None
        self.predictor: TargetPredictor = None

        # Store reference state (optional, for convenience)
        self.register_buffer('_reference_embeddings', None, persistent=True)
        self.register_buffer('_reference_masks', None, persistent=True)

    def forward(self, images: list[Image], priors: list[Priors] | None = None) -> Results:
        """Dispatcher based on training flag."""
        if self.training:
            if priors is None:
                raise ValueError("priors required in training mode")
            return self.fit(images, priors)
        else:
            return self.predict(images)

    def fit(self, reference_images: list[Image], reference_priors: list[Priors]) -> Results:
        """Extract and store reference embeddings using extractor.

        This is the public API method (replaces internal learn()).
        """
        self._reference_embeddings, self._reference_masks = self.extractor(
            reference_images,
            reference_priors
        )

        results = Results()
        results.embeddings = self._reference_embeddings
        return results

    def predict(self, target_images: list[Image]) -> Results:
        """Predict using stored embeddings via predictor.

        This is the public API method (replaces internal infer()).
        """
        if self._reference_embeddings is None:
            raise RuntimeError("Must call fit() before predict()")

        return self.predictor(
            target_images,
            self._reference_embeddings,
            self._reference_masks
        )

    def export(
        self,
        export_dir: str | Path,
        mode: Literal["static", "dynamic", "auto"] = "auto",
        backend: Literal["onnx", "tensorrt", "torchscript", "openvino"] = "onnx",
        **kwargs
    ) -> Path:
        """Export model for production deployment.

        Args:
            export_dir: Directory to save exported model(s)
            mode: Export mode (auto-detects based on fit() state)
                - "static": Bake reference embeddings into model (requires fit() first)
                - "dynamic": Export extractor + predictor separately
                - "auto": Static if fit() was called, else dynamic
            backend: Target backend (onnx, tensorrt, torchscript, openvino)
            **kwargs: Backend-specific options (optimization_level, quantization, etc.)

        Returns:
            Path to export directory with metadata

        Examples:
            >>> # Static export (embeddings baked in)
            >>> model.fit(ref_images, ref_priors)  # Must fit first!
            >>> model.export("./exports", mode="static")

            >>> # Dynamic export (extractor + predictor separate)
            >>> model.export("./exports", mode="dynamic")

            >>> # Auto mode (smart default)
            >>> model.fit(ref_images, ref_priors)
            >>> model.export("./exports")  # Auto-detects: static

            >>> # Multi-backend export
            >>> model.export("./exports", mode="static", backend="tensorrt")
            >>> model.export("./jetson_exports", backend="openvino")
        """
        from getiprompt.export import export_model

        # Auto-detect mode if needed
        if mode == "auto":
            mode = "static" if self._reference_embeddings is not None else "dynamic"

        # Validate static mode requirements
        if mode == "static" and self._reference_embeddings is None:
            raise RuntimeError(
                "Static export requires fit() to be called first. "
                "Either call fit() or use mode='dynamic'."
            )

        # Delegate to export utility function
        return export_model(
            model=self,
            export_dir=export_dir,
            mode=mode,
            backend=backend,
            **kwargs
        )
```

---

## Example Implementations for Special Cases

### GroundedSAM (Text-Only Model)

```python
# library/src/getiprompt/models/matcher.py
from getiprompt.models.base import Model, ReferenceExtractor, TargetPredictor
from getiprompt.components import (
    ImageEncoder, SamDecoder, MaskAdder,
    AllFeaturesSelector, BidirectionalPromptGenerator, MaxPointFilter, MasksToPolygons
)


class MatcherExtractor(ReferenceExtractor):
    """Matcher's reference extractor - extracts embeddings from reference images."""

    def __init__(self, encoder, feature_selector, mask_adder):
        super().__init__()
        self.encoder = encoder
        self.feature_selector = feature_selector
        self.mask_adder = mask_adder

    def forward(self, images, priors):
        # Pre-process priors
        priors = self.mask_adder(images, priors)

        # Extract features
        features, masks = self.encoder(images, priors)

        # Select features
        embeddings = self.feature_selector(features)

        return embeddings, masks


class MatcherPredictor(TargetPredictor):
    """Matcher's target predictor - predicts on targets using reference embeddings."""

    def __init__(self, encoder, prompt_generator, point_filter, segmenter, mask_processor):
        super().__init__()
        self.encoder = encoder
        self.prompt_generator = prompt_generator
        self.point_filter = point_filter
        self.segmenter = segmenter
        self.mask_processor = mask_processor

    def forward(self, target_images, reference_embeddings, reference_masks):
        # Extract target features
        target_features, _ = self.encoder(target_images)

        # Generate prompts from reference
        priors, similarities = self.prompt_generator(
            reference_embeddings,
            target_features,
            reference_masks,
            target_images,
        )

        # Filter and segment
        priors = self.point_filter(priors)
        masks, used_points, _ = self.segmenter(target_images, priors, similarities)
        annotations = self.mask_processor(masks)

        # Return results
        results = Results()
        results.priors = priors
        results.used_points = used_points
        results.masks = masks
        results.annotations = annotations
        results.similarities = similarities
        return results


class Matcher(Model):
    """Matcher model - bidirectional prompt-based matching."""

    def __init__(
        self,
        sam: SAMModelName = SAMModelName.SAM_HQ_TINY,
        num_foreground_points: int = 40,
        num_background_points: int = 2,
        encoder_model: str = "dinov3_large",
        mask_similarity_threshold: float | None = 0.38,
        device: str = "cuda",
        **kwargs
    ):
        super().__init__()

        # Initialize components (shared between extractor and predictor)
        sam_predictor = load_sam_model(sam, device, **kwargs)
        encoder = ImageEncoder(model_id=encoder_model, device=device, **kwargs)
        feature_selector = AllFeaturesSelector()
        prompt_generator = BidirectionalPromptGenerator(
            encoder_input_size=encoder.input_size,
            encoder_patch_size=encoder.patch_size,
            encoder_feature_size=encoder.feature_size,
            num_background_points=num_background_points,
        )
        point_filter = MaxPointFilter(max_num_points=num_foreground_points)
        segmenter = SamDecoder(sam_predictor, mask_similarity_threshold)
        mask_adder = MaskAdder(segmenter)
        mask_processor = MasksToPolygons()

        # Initialize the two submodules
        self.extractor = MatcherExtractor(encoder, feature_selector, mask_adder)
        self.predictor = MatcherPredictor(
            encoder, prompt_generator, point_filter, segmenter, mask_processor
        )
```

---

## Benefits of This Design

### ✅ Explicit Architecture
- Clear separation: Extract vs Predict
- No hidden adapters needed
- Easy to understand data flow

### ✅ Export-Friendly
```python
# Export extractor only
torch.onnx.export(model.extractor, (ref_images, ref_priors), "extractor.onnx")

# Export predictor only
torch.onnx.export(model.predictor, (target_images, embeddings, masks), "predictor.onnx")

# Or export whole model (static mode)
model.learn(ref_images, ref_priors)
torch.onnx.export(model, (target_images,), "model.onnx")
```

### ✅ Testing
```python
# Test extractor independently
embeddings, masks = model.extractor(ref_images, ref_priors)
assert embeddings.shape == (N, D)

# Test predictor independently
results = model.predictor(target_images, embeddings, masks)
assert len(results.masks) == len(target_images)

# Test full pipeline
model.learn(ref_images, ref_priors)
results = model.infer(target_images)
```

### ✅ Component Sharing
```python
# Share encoder between extractor and predictor
encoder = ImageEncoder(...)
extractor = MatcherExtractor(encoder, ...)
predictor = MatcherPredictor(encoder, ...)  # Same encoder instance!
```

### ✅ Flexible Deployment
```python
# Scenario 1: Deploy both (dynamic learning)
extractor_session = ort.InferenceSession("extractor.onnx")
predictor_session = ort.InferenceSession("predictor.onnx")

# Scenario 2: Deploy predictor only (static, with pre-computed embeddings)
predictor_session = ort.InferenceSession("predictor.onnx")
embeddings = load_precomputed_embeddings()

# Scenario 3: Deploy full model (frozen embeddings)
model_session = ort.InferenceSession("model.onnx")
```

---

## Export Backends (Simplified)

Export backends now work directly with the submodules:

```python
# library/src/getiprompt/export/onnx.py
class ONNXExportBackend(ExportBackend):

    def export_static(self, model, reference_images, reference_priors, export_path, **kwargs):
        """Export with frozen embeddings (most common)."""
        model.learn(reference_images, reference_priors)
        model.eval()

        dummy_target = kwargs.get('dummy_target_images', reference_images)
        torch.onnx.export(model, (dummy_target,), str(export_path), ...)

    def export_dynamic(self, model, export_path, **kwargs):
        """Export extractor and predictor separately."""
        base_path = Path(export_path)
        extractor_path = base_path.parent / f"{base_path.stem}_extractor.onnx"
        predictor_path = base_path.parent / f"{base_path.stem}_predictor.onnx"

        # Export extractor
        model.eval()
        dummy_ref_images = kwargs['dummy_ref_images']
        dummy_ref_priors = kwargs['dummy_ref_priors']
        torch.onnx.export(
            model.extractor,  # ← Direct access to submodule!
            (dummy_ref_images, dummy_ref_priors),
            str(extractor_path),
            input_names=['images', 'priors'],
            output_names=['embeddings', 'masks'],
            ...
        )

        # Export predictor
        dummy_target = kwargs['dummy_target_images']
        dummy_embeddings = kwargs['dummy_embeddings']
        dummy_masks = kwargs['dummy_masks']
        torch.onnx.export(
            model.predictor,  # ← Direct access to submodule!
            (dummy_target, dummy_embeddings, dummy_masks),
            str(predictor_path),
            input_names=['target_images', 'ref_embeddings', 'ref_masks'],
            output_names=['masks', 'annotations'],
            ...
        )

        print(f"Exported extractor to {extractor_path}")
        print(f"Exported predictor to {predictor_path}")
```

**No adapters needed!** The submodules are already the right granularity.

---

## Usage Examples

### Basic Usage (Same as Before)
```python
from getiprompt.models import Matcher

matcher = Matcher()

# Learn
matcher.learn(ref_images, ref_priors)

# Infer
results = matcher.infer(target_images)
```

### Direct Submodule Access
```python
# Use extractor directly
embeddings, masks = matcher.extractor(ref_images, ref_priors)

# Use predictor directly
results = matcher.predictor(target_images, embeddings, masks)

# Useful for debugging, testing, or custom pipelines!
```

### Export to Multiple Backends
```python
from getiprompt.export import export_model

# Static ONNX (frozen embeddings)
export_model(matcher, 'onnx', 'matcher.onnx', mode='static',
             reference_images=ref, reference_priors=priors)

# Dynamic ONNX (separate extractor + predictor)
export_model(matcher, 'onnx', 'matcher.onnx', mode='dynamic',
             dummy_ref_images=ref, dummy_ref_priors=priors,
             dummy_target_images=target, dummy_embeddings=emb, dummy_masks=masks)

# TorchScript
export_model(matcher, 'torchscript', 'matcher.pt', mode='static',
             reference_images=ref, reference_priors=priors)
```

### Production Inference with InferenceModel
```python
from getiprompt.inference import InferenceModel

# Backend auto-detected from file extension!
model = InferenceModel(
    extractor="matcher_extractor.trt",  # .trt → TensorRT
    predictor="matcher_predictor.trt"
)

# Same API as PyTorch Model!
model.learn(ref_images, ref_priors)
results = model.infer(target_images)

# Switch backend by changing file extension
model = InferenceModel(
    extractor="matcher_extractor.onnx",  # .onnx → ONNX
    predictor="matcher_predictor.onnx"
)
```

See:
- [EXPORT_BEST_PRACTICES.md](EXPORT_BEST_PRACTICES.md) for export guide
- [INFERENCE_MODEL.md](INFERENCE_MODEL.md) for production deployment

### ONNX Runtime (Dynamic Mode)
```python
import onnxruntime as ort

# Load both submodules
extractor_session = ort.InferenceSession('matcher_extractor.onnx')
predictor_session = ort.InferenceSession('matcher_predictor.onnx')

# Extract embeddings (dynamic!)
embeddings, masks = extractor_session.run(None, {
    'images': new_ref_images,
    'priors': new_priors  # ← Can change anytime!
})

# Predict on targets
results = predictor_session.run(None, {
    'target_images': target_images,
    'ref_embeddings': embeddings,
    'ref_masks': masks
})

# Cache embeddings for multiple predictions
for batch in target_batches:
    results = predictor_session.run(None, {
        'target_images': batch,
        'ref_embeddings': embeddings,  # ← Reuse!
        'ref_masks': masks
    })
```

---

## Alternative Names Considered

| Option | Extractor Name | Predictor Name | Notes |
|--------|----------------|----------------|-------|
| **Current** | `ReferenceExtractor` | `TargetPredictor` | Clear and explicit |
| Option 2 | `Encoder` | `Decoder` | Overloaded terms |
| Option 3 | `Learner` | `Inferer` | Verb-based (less common) |
| Option 4 | `FeatureExtractor` | `MaskPredictor` | More specific |
| Option 5 | `ReferenceEncoder` | `TargetDecoder` | Encoder/Decoder paradigm |

**Recommendation**: Stick with `ReferenceExtractor` and `TargetPredictor` because:
- ✅ Descriptive: Clear what they operate on
- ✅ Not overloaded: Unlike "Encoder"/"Decoder"
- ✅ Noun-based: Follows PyTorch convention (Module, Sequential, etc.)
- ✅ Asymmetric inputs: Makes it clear extractor takes images+priors, predictor takes images+embeddings

---

## Migration from Current Code

### Before (Monolithic)
```python
class Matcher(Model):
    def learn(self, reference_images, reference_priors):
        # ... lots of code ...
        self.reference_features = features

    def infer(self, target_images):
        # ... uses self.reference_features ...
```

### After (Modular)
```python
class Matcher(Model):
    def __init__(self, ...):
        self.extractor = MatcherExtractor(...)
        self.predictor = MatcherPredictor(...)

    # learn() and infer() are in base class!
    # Just delegate to extractor/predictor
```

**Changes needed per model:**
1. Split `learn()` logic into `MatcherExtractor.forward()`
2. Split `infer()` logic into `MatcherPredictor.forward()`
3. Update `__init__()` to instantiate both submodules

---

## Generalization Analysis: Does This Work for All Models?

Let me check all models in the codebase:

### ✅ Matcher
**Extractor**: Images + Priors → Embeddings
**Predictor**: Target Images + Embeddings → Masks
**Pattern**: ✅ Perfect fit

### ✅ PerDino
**Extractor**: Images + Priors → Average Features
**Predictor**: Target Images + Features → Masks (via grid matching)
**Pattern**: ✅ Perfect fit

### ✅ SoftMatcher (inherits from Matcher)
**Extractor**: Same as Matcher
**Predictor**: Same as Matcher (different prompt generator)
**Pattern**: ✅ Perfect fit

### ⚠️ GroundedSAM (Text-based)
```python
def learn(self, reference_images, reference_priors):
    # Only stores text priors, no image processing
    self.text_priors = reference_priors[0].text

def infer(self, target_images):
    # Uses text + detector to generate boxes
    priors = self.prompt_generator(target_images, [self.text_priors] * len(target_images))
    masks = self.segmenter(target_images, priors)
```

**Analysis**:
- ❌ No "embedding extraction" in traditional sense
- ✅ But still has two-stage: store text → use text for detection
- ✅ Can adapt: `extractor` just stores text, `predictor` does detection + segmentation

**Solution**: Generalize to "Reference Processor" that stores ANY reference data:

```python
class GroundedSAMExtractor(ReferenceExtractor):
    def forward(self, images, priors):
        # Extract text (no computation needed)
        text_priors = priors[0].text
        return text_priors, None  # embeddings, masks

class GroundedSAMPredictor(TargetPredictor):
    def forward(self, target_images, text_priors, _):
        # Use text to detect and segment
        priors = self.prompt_generator(target_images, [text_priors] * len(target_images))
        masks = self.segmenter(target_images, priors)
        return results
```

**Pattern**: ⚠️ Adapts (stores text instead of embeddings)

### ⚠️ DinoTxt (Zero-shot classification)
```python
def learn(self, reference_images, reference_priors):
    # Doesn't use images! Only text priors
    self.reference_features = self.dino_encoder.encode_text(
        reference_priors[0],
        self.prompt_templates
    )

def infer(self, target_images):
    # Encode images, match with text features
    target_features = self.dino_encoder.encode_image(target_images)
    logits = target_features @ self.reference_features
```

**Analysis**:
- ❌ `learn()` doesn't use reference_images at all (only text)
- ✅ Still has two-stage: text → embeddings → classification
- ✅ Can adapt: `extractor` encodes text, `predictor` encodes images + matches

**Solution**:
```python
class DinoTxtExtractor(ReferenceExtractor):
    def forward(self, images, priors):  # images unused
        # Encode text prompts to features
        text_features = self.dino_encoder.encode_text(priors[0], self.prompt_templates)
        return text_features, None

class DinoTxtPredictor(TargetPredictor):
    def forward(self, target_images, text_features, _):
        # Encode images and match
        image_features = self.dino_encoder.encode_image(target_images)
        logits = image_features @ text_features
        # ... classification logic ...
        return results
```

**Pattern**: ⚠️ Adapts (text embedding instead of image embedding)

---

## Refined Design: More Generic Names

The pattern DOES generalize, but we need more generic terminology:

### Updated Base Classes

```python
class ReferenceProcessor(nn.Module, ABC):
    """Processes reference data (images, text, etc.) into a reusable representation.

    This could be:
    - Image embeddings (Matcher, PerDino)
    - Text embeddings (DinoTxt)
    - Stored text priors (GroundedSAM)
    - Any preprocessed reference data
    """

    @abstractmethod
    def forward(
        self,
        images: list[Image] | None,  # Optional - DinoTxt doesn't use
        priors: list[Priors]
    ) -> tuple[torch.Tensor | Any, torch.Tensor | None]:
        """Process reference data into reusable representation.

        Returns:
            reference_data: Processed reference (embeddings, text, etc.)
            auxiliary_data: Optional auxiliary data (masks, etc.)
        """
        pass


class TargetProcessor(nn.Module, ABC):
    """Processes target images using reference data to generate predictions.

    Uses the reference data from ReferenceProcessor to:
    - Match features (Matcher, PerDino)
    - Detect objects (GroundedSAM)
    - Classify (DinoTxt)
    - Segment
    """

    @abstractmethod
    def forward(
        self,
        target_images: list[Image],
        reference_data: torch.Tensor | Any,
        auxiliary_data: torch.Tensor | Any | None = None
    ) -> Results:
        """Process targets using reference data."""
        pass
```

### Why More Generic Names?

| Model | Reference Processing | Target Processing |
|-------|---------------------|-------------------|
| **Matcher** | Extract image embeddings | Match & segment |
| **PerDino** | Extract image embeddings | Grid match & segment |
| **GroundedSAM** | Store text priors | Text-based detection & segment |
| **DinoTxt** | Encode text to features | Encode images & classify |

The common pattern:
1. **Process reference data** (could be images, text, or both)
2. **Use processed data** to predict on targets

---

## Recommended Naming

### Option 1: Keep Extractor/Predictor (with flexibility)
```python
class ReferenceExtractor(nn.Module, ABC):
    """Extracts reusable representation from reference data."""
    # Works for: embeddings, text encoding, storing priors

class TargetPredictor(nn.Module, ABC):
    """Predicts on targets using reference representation."""
    # Works for: matching, detection, classification
```

**Pros**: Familiar, concise
**Cons**: "Extractor" implies computation (but GroundedSAM just stores text)

### Option 2: More Generic (Processor)
```python
class ReferenceProcessor(nn.Module, ABC):
    """Processes reference data into reusable representation."""

class TargetProcessor(nn.Module, ABC):
    """Processes targets using reference representation."""
```

**Pros**: More accurate, handles all cases
**Cons**: More verbose, less specific

### Option 3: Encoder/Decoder Paradigm
```python
class ReferenceEncoder(nn.Module, ABC):
    """Encodes reference data."""

class TargetDecoder(nn.Module, ABC):
    """Decodes targets using reference encoding."""
```

**Pros**: Familiar encoder/decoder terminology
**Cons**: "Decoder" is overloaded (usually means generating outputs from latents)

---

## Recommendation: Stick with Extractor/Predictor

**Verdict**: The **Extractor/Predictor** pattern **DOES generalize** to all models:

✅ **Matcher/PerDino/SoftMatcher**: Perfect fit (extract embeddings → predict)
✅ **GroundedSAM**: Adapts (store text → predict)
✅ **DinoTxt**: Adapts (encode text → predict)

**Keep the names** because:
1. ✅ More intuitive than "Processor"
2. ✅ "Extract" is flexible (extract embeddings, extract text, etc.)
3. ✅ "Predictor" is clear (predict masks, classes, etc.)
4. ✅ Shorter and more memorable

**Flexibility**: Make `images` parameter optional in `ReferenceExtractor`:
```python
class ReferenceExtractor(nn.Module, ABC):
    def forward(
        self,
        images: list[Image] | None = None,  # Optional for text-only models
        priors: list[Priors] | None = None
    ) -> tuple[torch.Tensor | Any, torch.Tensor | None]:
        pass
```

---

## Summary

✅ **Pattern generalizes to ALL models**:
- Image-based: Matcher, PerDino, SoftMatcher
- Text-based: GroundedSAM, DinoTxt
- Hybrid: (future models)

✅ **Naming**: Stick with `ReferenceExtractor` / `TargetPredictor`
- Flexible enough for all use cases
- More intuitive than generic "Processor"
- Shorter than "ReferenceEncoder/TargetDecoder"

✅ **Implementation**: Make `images` optional in `ReferenceExtractor` signature

✅ **Export**: See [EXPORT_BEST_PRACTICES.md](EXPORT_BEST_PRACTICES.md) for multi-backend export guide
- Dynamic mode: Export extractor + predictor separately (change references anytime)
- Static mode: Export full model with frozen embeddings (fastest)
- Supports: ONNX, TensorRT, TorchScript, OpenVINO

The key insight: **The two-stage pattern is universal** - process reference data once, use it many times! 🎯
