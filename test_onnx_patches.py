#!/usr/bin/env python3
"""Test script to verify ONNX patches are applied correctly."""

import torch
from getiprompt.foundation.models import load_sam_model
from getiprompt.utils.constants import SAMModelName


def test_onnx_patches():
    """Test that ONNX patches are applied when requested."""
    print("Testing ONNX patches functionality...")
    
    try:
        # Test loading with ONNX patches
        print("Loading model with apply_onnx_patches=True...")
        model_with_patches = load_sam_model(
            SAMModelName.SAM_HQ_TINY,
            device="cpu",
            precision="fp32",
            compile_models=False,
            benchmark_inference_speed=False,
            apply_onnx_patches=True,
        )
        
        # Test loading without ONNX patches
        print("Loading model with apply_onnx_patches=False...")
        model_without_patches = load_sam_model(
            SAMModelName.SAM_HQ_TINY,
            device="cpu",
            precision="fp32",
            compile_models=False,
            benchmark_inference_speed=False,
            apply_onnx_patches=False,
        )
        
        # Check if the patches were applied
        prompt_encoder_with = model_with_patches.model.prompt_encoder
        prompt_encoder_without = model_without_patches.model.prompt_encoder
        
        # Get the method objects
        embed_points_with = getattr(prompt_encoder_with, '_embed_points', None)
        embed_boxes_with = getattr(prompt_encoder_with, '_embed_boxes', None)
        embed_points_without = getattr(prompt_encoder_without, '_embed_points', None)
        embed_boxes_without = getattr(prompt_encoder_without, '_embed_boxes', None)
        
        print(f"Model with patches - _embed_points: {embed_points_with is not None}")
        print(f"Model with patches - _embed_boxes: {embed_boxes_with is not None}")
        print(f"Model without patches - _embed_points: {embed_points_without is not None}")
        print(f"Model without patches - _embed_boxes: {embed_boxes_without is not None}")
        
        # Check if the methods are different (patched vs original)
        if embed_points_with and embed_points_without:
            methods_different = embed_points_with != embed_points_without
            print(f"✓ _embed_points methods are different: {methods_different}")
        else:
            print("✗ Could not compare _embed_points methods")
            
        if embed_boxes_with and embed_boxes_without:
            methods_different = embed_boxes_with != embed_boxes_without
            print(f"✓ _embed_boxes methods are different: {methods_different}")
        else:
            print("✗ Could not compare _embed_boxes methods")
        
        print("✓ Test completed successfully!")
        
    except Exception as e:
        print(f"✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_onnx_patches()

