# SAM3 Refactoring Plan: Remove Transformers Dependencies

## Overview

Refactor the Geti SAM3 implementation to remove transformer library dependencies while maintaining identical accuracy compared to the HuggingFace implementation.

**Current Baseline (from comparison script):**
- HuggingFace SAM3 mIoU: 0.8612
- Geti SAM3 mIoU: 0.8626
- HF vs Geti Agreement: 0.9252 average IoU

**Goals:**
1. Maintain or exceed current accuracy after refactoring
2. Remove all `@dataclass` decorators - use plain dictionaries or `TypedDict` for outputs
3. Remove transformers config dependencies - use explicit function arguments
4. Remove `initialization.py` - use standard torch initialization

---

## Phase 1: Dependency Analysis

### Current Transformers Imports in SAM3 Module

| File | Imports from `transformers` |
|------|----------------------------|
| `modeling_sam3.py` | `CLIPTextModelWithProjection`, `ACT2FN`, `GradientCheckpointingLayer`, `BaseModelOutput`, `BaseModelOutputWithPooling`, `ModelOutput`, `ALL_ATTENTION_FUNCTIONS`, `PreTrainedModel`, `Unpack`, `compile_compatible_method_lru_cache`, `can_return_tuple`, `logging`, `TransformersKwargs`, `check_model_inputs`, `is_flash_attention_requested` |
| `configuration_sam3.py` | `CLIPTextConfig`, `PretrainedConfig`, `AutoConfig` |
| `processing_sam3.py` | `ImageInput`, `ProcessorMixin`, `BatchEncoding`, `PreTokenizedInput`, `TextInput`, `TensorType`, `logging` |
| `image_processing_sam3_fast.py` | `BatchFeature`, `get_size_dict`, `BaseImageProcessorFast`, image utilities, `ImagesKwargs`, `TensorType` |
| `initialization.py` | None (only torch) |

### Categories of Dependencies

1. **Essential Model Components** (MUST keep or replace carefully):
   - `CLIPTextModelWithProjection` - Text encoder backbone
   - `PreTrainedModel` - Base class for weight loading

2. **Utility Functions** (can replace with local implementations):
   - `ACT2FN` - Activation function registry
   - `logging` - Can use standard Python logging

3. **Output Dataclasses** (REMOVE - use plain dicts or TypedDict):
   - `BaseModelOutput`, `BaseModelOutputWithPooling`, `ModelOutput`
   - `Sam3VisionEncoderOutput`, `Sam3GeometryEncoderOutput`, etc.
   - Replace with plain `dict[str, torch.Tensor]` or `TypedDict`

4. **Processing Utilities** (can simplify or replace):
   - `ProcessorMixin`, `BatchEncoding`, `BatchFeature`
   - Image processing utilities

5. **Config Classes** (main refactoring target):
   - All `Sam3*Config` classes

---

## Phase 2: Initialization Module Analysis

### Question: Can we remove `initialization.py`?

**What it does:**
- Guards torch initialization functions with `_is_hf_initialized` flag
- Prevents re-initialization of weights already loaded from checkpoint

**Current usage in modeling_sam3.py:**
```python
from . import initialization as init
# Used in Sam3PreTrainedModel._init_weights():
init.normal_(module.position_embeddings, mean=0.0, std=self.config.initializer_range)
init.copy_(module.rope_embeddings_cos, inv_freq.cos())
init.copy_(module.rope_embeddings_sin, inv_freq.sin())
```

**Verdict: CAN REMOVE with careful handling**

The guarded init is primarily needed for HuggingFace's `from_pretrained()` workflow to prevent double initialization. Options:
1. Use standard `torch.nn.init.*` functions directly
2. Skip custom `_init_weights()` entirely since we always load from pretrained
3. Create a minimal guard wrapper only for the 3 usages

**Recommendation:** Remove `initialization.py` and use torch primitives directly since:
- We always load from pretrained checkpoint
- The guarding is a HF-specific optimization
- Simplifies the codebase

---

## Phase 3: Configuration Refactoring

### Target: Convert Config Classes to Dataclass/Direct Arguments

**Before (transformers style):**
```python
class Sam3ViTRoPEAttention(nn.Module):
    def __init__(self, config: Sam3ViTConfig):
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
```

**After (explicit arguments):**
```python
class Sam3ViTRoPEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        attn_implementation: str = "eager",
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
```

### Config Classes to Refactor

| Config Class | Parameters | Approach |
|-------------|------------|----------|
| `Sam3ViTConfig` | hidden_size, intermediate_size, num_hidden_layers, etc. | Convert to dataclass or inline args |
| `Sam3VisionConfig` | backbone_config, fpn_hidden_size, etc. | Flatten into parent module |
| `Sam3GeometryEncoderConfig` | hidden_size, num_layers, etc. | Convert to dataclass |
| `Sam3DETREncoderConfig` | hidden_size, num_layers, etc. | Convert to dataclass |
| `Sam3DETRDecoderConfig` | hidden_size, num_queries, etc. | Convert to dataclass |
| `Sam3MaskDecoderConfig` | hidden_size, num_upsampling_stages, etc. | Convert to dataclass |
| `Sam3Config` | Composition of all above | Convert to main dataclass |

### New Structure Options

**Option A: Single Dataclass with Nested Structure**
```python
@dataclass
class Sam3ModelConfig:
    # Vision encoder
    vit_hidden_size: int = 1024
    vit_intermediate_size: int = 4736
    vit_num_layers: int = 32
    # ... other params
```

**Option B: Factory Functions with Defaults**
```python
def create_sam3_model(
    hidden_size: int = 256,
    vit_hidden_size: int = 1024,
    # ... other params
) -> Sam3Model:
    ...
```

**Recommendation:** Option A with a simple dataclass hierarchy - cleaner API and easier weight loading.

**UPDATE: Per user requirement, we will NOT use dataclasses for config either.**

**Option C: Explicit Arguments with Defaults (CHOSEN)**
```python
class Sam3ViTRoPEAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int = 1024,
        num_attention_heads: int = 16,
        attention_dropout: float = 0.0,
        attn_implementation: str = "eager",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        # ...
```

This approach:
- No config objects at all
- Each module declares its own defaults
- Parent modules pass explicit values to children
- Factory function can provide standard SAM3 defaults

---

## Phase 4: Weight Loading Strategy

### Challenge: Must maintain compatibility with HF checkpoints

**Current approach:** Uses `PreTrainedModel.from_pretrained()`

**Proposed approach:** Create a custom weight loading function:
```python
def load_sam3_weights(model: Sam3Model, checkpoint_path: str) -> Sam3Model:
    """Load weights from HuggingFace checkpoint format."""
    state_dict = torch.load(checkpoint_path)
    # Apply any necessary key remapping
    model.load_state_dict(state_dict, strict=False)
    return model
```

### Key Mapping Considerations

The current model uses:
```python
_checkpoint_conversion_mapping = {
    r"detector_model.(.+)": r"\1"
}
```

This must be preserved in the custom loader.

---

## Phase 5: Refactoring Strategy

### Step-by-Step Plan

1. **Create local utility modules** (low risk)
   - `sam3_activations.py` - Local ACT2FN equivalent
   - `sam3_outputs.py` - Local dataclass outputs
   - `sam3_utils.py` - Common utilities

2. **Create new config dataclass** (medium risk)
   - `sam3_config.py` - Pure Python dataclasses
   - Preserve all default values exactly

3. **Refactor module by module** (high risk - test after each)
   - Start with leaf modules (MLP, Attention)
   - Progress to composite modules (Encoder, Decoder)
   - Finally refactor main Sam3Model

4. **Implement custom weight loader** (medium risk)
   - Support HuggingFace checkpoint format
   - Maintain key mapping

5. **Update processing modules** (medium risk)
   - Simplify processor classes
   - Remove HF-specific mixins

6. **Remove initialization.py** (low risk)
   - Replace with direct torch.nn.init calls

7. **Verification at each step**
   - Run `sam3_comparison.py`
   - Ensure mIoU ≥ 0.8626
   - Ensure HF-Geti agreement ≥ 0.92

---

## Phase 6: Detailed Module Refactoring

### 6.1 Create Local Activations (`sam3_activations.py`)

```python
"""Local activation function registry."""
import torch.nn.functional as F

ACT2FN = {
    "gelu": F.gelu,
    "relu": F.relu,
    "silu": F.silu,
    "quick_gelu": lambda x: x * torch.sigmoid(1.702 * x),
}

def get_activation(name: str):
    if name not in ACT2FN:
        raise ValueError(f"Unknown activation: {name}")
    return ACT2FN[name]
```

### 6.2 Remove Output Dataclasses - Use TypedDict or Plain Dicts

**Before (dataclass style):**
```python
@dataclass
class Sam3VisionEncoderOutput(BaseModelOutputWithPooling):
    fpn_hidden_states: tuple[torch.FloatTensor, ...] = None
    fpn_position_encoding: tuple[torch.FloatTensor, ...] = None
```

**After (TypedDict style):**
```python
from typing import TypedDict

class Sam3VisionEncoderOutput(TypedDict, total=False):
    last_hidden_state: torch.FloatTensor
    pooler_output: torch.FloatTensor | None
    hidden_states: tuple[torch.FloatTensor, ...] | None
    attentions: tuple[torch.FloatTensor, ...] | None
    fpn_hidden_states: tuple[torch.FloatTensor, ...] | None
    fpn_position_encoding: tuple[torch.FloatTensor, ...] | None
```

**Or even simpler (plain dict with type hints in docstrings):**
```python
def forward(...) -> dict[str, torch.Tensor]:
    """
    Returns:
        dict with keys:
        - last_hidden_state: (batch, seq, hidden)
        - fpn_hidden_states: tuple of FPN features
        - fpn_position_encoding: tuple of position encodings
    """
    return {
        "last_hidden_state": hidden_states,
        "fpn_hidden_states": fpn_features,
        "fpn_position_encoding": pos_encodings,
    }
```

### Dataclasses to Remove

| Dataclass | Location | Replacement |
|-----------|----------|-------------|
| `Sam3VisionEncoderOutput` | modeling_sam3.py | `dict` or `TypedDict` |
| `Sam3GeometryEncoderOutput` | modeling_sam3.py | `dict` or `TypedDict` |
| `Sam3DETREncoderOutput` | modeling_sam3.py | `dict` or `TypedDict` |
| `Sam3DETRDecoderOutput` | modeling_sam3.py | `dict` or `TypedDict` |
| `Sam3MaskDecoderOutput` | modeling_sam3.py | `dict` or `TypedDict` |
| `Sam3ImageSegmentationOutput` | modeling_sam3.py | `dict` or `TypedDict` |

### 6.3 New Config Structure - Explicit Arguments (NO DATACLASS)

Instead of config classes, use explicit arguments with defaults:

```python
# Constants for SAM3 defaults
SAM3_DEFAULTS = {
    "vit_hidden_size": 1024,
    "vit_intermediate_size": 4736,
    "vit_num_layers": 32,
    "vit_num_heads": 16,
    "vit_patch_size": 14,
    "vit_image_size": 1008,
    "vit_window_size": 24,
    "vit_global_attn_indexes": [7, 15, 23, 31],
    "fpn_hidden_size": 256,
    "detr_encoder_hidden_size": 256,
    "detr_encoder_num_layers": 6,
    "detr_encoder_num_heads": 8,
    "detr_decoder_hidden_size": 256,
    "detr_decoder_num_layers": 6,
    "detr_decoder_num_queries": 200,
    "mask_decoder_hidden_size": 256,
    "mask_decoder_num_upsampling_stages": 3,
    "text_encoder_hidden_size": 1024,
    "hidden_act": "gelu",
    "layer_norm_eps": 1e-6,
    "dropout": 0.0,
    "initializer_range": 0.02,
    "attn_implementation": "sdpa",
}

# Example module with explicit args
class Sam3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "gelu",
        hidden_dropout: float = 0.0,
    ):
        super().__init__()
        self.activation_fn = get_activation(hidden_act)
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(hidden_dropout)
```

---

## Phase 7: Dependencies to Keep

Even after refactoring, some dependencies should remain:

1. **CLIP Text Encoder** - `CLIPTextModelWithProjection`
   - Essential for text prompting
   - Complex to reimplement correctly
   - HuggingFace maintains it well

2. **torchvision** - Already used for ops like `batched_nms`

3. **Standard libraries** - torch, numpy, PIL

---

## Phase 8: Testing Strategy

### Validation Script Updates

```python
# sam3_comparison.py modifications
def compare_on_lvis():
    """Compare with explicit accuracy thresholds."""
    # ... existing code ...
    
    # Add strict validation
    assert geti_score >= 0.86, f"Geti mIoU dropped: {geti_score}"
    assert avg_hf_geti >= 0.92, f"HF-Geti agreement dropped: {avg_hf_geti}"
```

### Incremental Testing

After each major change:
```bash
python examples/sam3_comparison.py
```

Expected output must show:
- Geti SAM3 mIoU ≥ 0.8626 (current baseline)
- HF vs Geti Agreement ≥ 0.9252

---

## Phase 9: Implementation Timeline

| Phase | Task | Risk | Effort |
|-------|------|------|--------|
| 1 | Create local utilities (activations only) | Low | 0.5 day |
| 2 | Remove output dataclasses → use dicts | Low | 1 day |
| 3 | Refactor Sam3MLP, Sam3Attention (leaf modules) - explicit args | Medium | 1 day |
| 4 | Refactor Sam3ViT modules - explicit args | Medium | 2 days |
| 5 | Refactor DETR Encoder/Decoder - explicit args | High | 2 days |
| 6 | Refactor Sam3Model main class - explicit args | High | 2 days |
| 7 | Custom weight loader | Medium | 1 day |
| 8 | Simplify processing modules | Low | 1 day |
| 9 | Remove initialization.py and configuration_sam3.py | Low | 0.5 day |
| 10 | Final testing and cleanup | Low | 1 day |

**Total Estimated Effort: 12 days**

---

## Phase 10: Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Accuracy regression | High | Test after every module change |
| Weight loading issues | High | Preserve exact key mapping |
| Numerical precision | Medium | Use fp32 during validation |
| Breaking API changes | Medium | Maintain backward compatible wrapper |
| CLIP dependency issues | Low | Keep CLIP from HF for now |

---

## Summary: Files to Modify/Create

### Files to Create
- `sam3_activations.py` - Local activation functions
- `sam3_types.py` - TypedDict definitions for outputs (optional, can use plain dicts)
- `sam3_weights.py` - Custom weight loading

### Files to Modify
- `modeling_sam3.py` - Major refactoring (remove dataclasses, explicit args)
- `processing_sam3.py` - Simplify
- `image_processing_sam3_fast.py` - Reduce HF dependencies
- `__init__.py` - Update exports

### Files to Delete
- `initialization.py` - No longer needed
- `configuration_sam3.py` - Replaced by explicit arguments

---

## Decision Points

1. **Keep CLIP from transformers?** → YES (complex to reimplement)
2. **Keep torchvision?** → YES (already a dependency)
3. **Remove initialization.py?** → YES (not needed for pretrained loading)
4. **Config style?** → NO CONFIG CLASSES - use explicit function arguments with defaults
5. **Output style?** → NO DATACLASSES - use plain `dict[str, torch.Tensor]` or `TypedDict`
6. **Processing style?** → Simplify, remove mixins

---

## Next Steps

1. Review and approve this plan
2. Start with Phase 1: Create local utilities
3. Run comparison script after each phase
4. Document any API changes
