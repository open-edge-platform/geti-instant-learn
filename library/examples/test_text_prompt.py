#!/usr/bin/env python3
"""Test EfficientSAM3 text prompt - replicating run_sam3_text_prompt.py"""

from __future__ import annotations

import os
import sys

# Setup paths before any imports
EFFICIENTSAM3_ROOT = os.path.expanduser("~/workspace/code/Prompt/efficientsam3")
GETI_PROMPT_SRC = os.path.expanduser("~/workspace/code/Prompt/geti-prompt/library/src")

# Add geti-prompt to path
sys.path.insert(0, GETI_PROMPT_SRC)

# Paths
checkpoint_path = os.path.join(EFFICIENTSAM3_ROOT, "efficient_sam3_tinyvit_11m_mobileclip_s1.pth")
bpe_path = os.path.join(EFFICIENTSAM3_ROOT, "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
image_path = "/home/devuser/workspace/code/Prompt/efficientsam3/sam3/assets/images/test_image.jpg"

# Text prompt
text_prompt = "car"


def main():
    import torch
    from PIL import Image
    
    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Import geti-prompt implementation
    from getiprompt.models.foundation.efficientsam3.model_builder import build_efficientsam3_image_model
    from getiprompt.models.foundation.sam3.sam3_image_processor import Sam3Processor

    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Image: {image_path}")
    print(f"Text prompt: '{text_prompt}'")

    # Build model
    print("\n=== Building geti-prompt model ===")
    model = build_efficientsam3_image_model(
        checkpoint_path=checkpoint_path,
        bpe_path=bpe_path,
        backbone_type="tinyvit-21m",
        text_encoder_type="MobileCLIP-S1",
        device=device,
    )
    print(f"Model loaded: {type(model).__name__}")

    # Load image
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    # Create processor and run inference
    processor = Sam3Processor(model)
    
    print("\n=== Running text prompt inference ===")
    inference_state = processor.set_image(image)
    inference_state = processor.set_text_prompt(text_prompt, inference_state)

    masks = inference_state["masks"]
    scores = inference_state["scores"]
    boxes = inference_state["boxes"]

    num_masks = int(masks.shape[0])
    print(f"geti-prompt results: {num_masks} masks")
    print(f"Scores: {scores.detach().cpu()}")

    # Now compare with original efficientsam3
    print("\n=== Comparing with original efficientsam3 ===")
    sam3_path = os.path.join(EFFICIENTSAM3_ROOT, "sam3")
    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    from sam3.model_builder import build_efficientsam3_image_model as build_original
    from sam3.model.sam3_image_processor import Sam3Processor as ProcessorOriginal

    model_orig = build_original(
        checkpoint_path=checkpoint_path,
        backbone_type="tinyvit",
        model_name="21m",
        text_encoder_type="MobileCLIP-S1",
    )
    print(f"Original model loaded: {type(model_orig).__name__}")

    processor_orig = ProcessorOriginal(model_orig)
    state_orig = processor_orig.set_image(image)
    state_orig = processor_orig.set_text_prompt(text_prompt, state_orig)

    masks_orig = state_orig["masks"]
    scores_orig = state_orig["scores"]

    num_masks_orig = int(masks_orig.shape[0])
    print(f"Original results: {num_masks_orig} masks")
    print(f"Scores: {scores_orig.detach().cpu()}")

    # Summary
    print("\n=== Summary ===")
    print(f"geti-prompt:  {num_masks} masks")
    print(f"Original:     {num_masks_orig} masks")
    if num_masks == num_masks_orig:
        print("✓ Results match!")
    else:
        print("✗ Results differ")


if __name__ == "__main__":
    main()
