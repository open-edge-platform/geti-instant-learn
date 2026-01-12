#!/usr/bin/env python3
"""Test EfficientSAM3 text prompt - replicating efficientsam3 repo's text_encoder_simple.py

This script tests the geti-prompt implementation of EfficientSAM3 with text prompts.
Mirrors the original efficientsam3/text_encoder_simple.py exactly.

Expected output (matching original): 6 masks with scores ~[0.56, 0.52, 0.56, 0.62, 0.61, 0.63]
"""

from __future__ import annotations

import sys
from pathlib import Path

# Setup paths
GETI_PROMPT_SRC = Path.home() / "workspace/code/Prompt/geti-prompt/library/src"
EFFICIENTSAM3_ROOT = Path.home() / "workspace/code/Prompt/efficientsam3"

# Add geti-prompt to path
sys.path.insert(0, str(GETI_PROMPT_SRC))

# Paths (same as text_encoder_simple.py - uses 11m model)
CHECKPOINT_PATH = EFFICIENTSAM3_ROOT / "efficient_sam3_tinyvit_11m_mobileclip_s1.pth"
BPE_PATH = EFFICIENTSAM3_ROOT / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
IMAGE_PATH = EFFICIENTSAM3_ROOT / "sam3" / "assets" / "images" / "test_image.jpg"

# Text prompt (same as text_encoder_simple.py)
TEXT_PROMPT = "shoe"


def main():
    from PIL import Image

    # Import geti-prompt implementation
    from getiprompt.models.foundation.efficientsam3.model_builder import (
        build_efficientsam3_image_model,
        EfficientSAM3BackboneType,
        EfficientSAM3TextEncoderType,
    )
    from getiprompt.models.foundation.sam3.sam3_image_processor import Sam3Processor

    # Build model with text encoder (same params as text_encoder_simple.py)
    model = build_efficientsam3_image_model(
        checkpoint_path=str(CHECKPOINT_PATH),
        bpe_path=str(BPE_PATH),
        backbone_type=EfficientSAM3BackboneType.TINYVIT_11M,
        text_encoder_type=EfficientSAM3TextEncoderType.MOBILECLIP_S1,
    )

    # Load image
    image = Image.open(IMAGE_PATH)

    # Process image and predict with text prompt
    processor = Sam3Processor(model)
    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(prompt=TEXT_PROMPT, state=inference_state)
    masks = inference_state["masks"]
    scores = inference_state["scores"]
    
    print(len(scores), scores)


if __name__ == "__main__":
    main()
