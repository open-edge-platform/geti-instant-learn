#!/usr/bin/env python3
"""
Example script demonstrating how to export PTSamDecoder to ONNX format.
"""

import torch
import numpy as np
from getiprompt.models.models import load_sam_model
from getiprompt.utils.constants import SAMModelName
from lib.src.getiprompt.processes.segmenters.sam_decoder_pt_onnx import PTSamDecoderONNX


def export_sam_decoder_onnx():
    """Export PTSamDecoder to ONNX format."""
    print("Loading SAM model...")
    
    # Load the SAM model with ONNX patches
    sam_predictor = load_sam_model(
        SAMModelName.SAM_HQ_TINY,
        device="cpu",
        precision="fp32",
        compile_models=False,
        benchmark_inference_speed=False,
        apply_onnx_patches=True,  # Important for ONNX compatibility
    )
    
    print("Creating ONNX-compatible decoder...")
    
    # Create ONNX-compatible decoder
    onnx_decoder = PTSamDecoderONNX(
        sam_predictor=sam_predictor,
        mask_similarity_threshold=0.45,
        nms_iou_threshold=0.1,
    )
    
    print("Exporting to ONNX...")
    
    # Export to ONNX
    output_path = "sam_decoder.onnx"
    onnx_decoder.export_onnx(
        output_path=output_path,
        image_size=(1024, 1024),
        # max_points=5,  # Reduce for smaller model
        original_size=(480, 640),
    )
    
    print(f"✅ Successfully exported SAM decoder to {output_path}")
    
    # Test the ONNX model
    print("Testing ONNX export...")
    test_onnx_export(onnx_decoder)


def test_onnx_export(onnx_decoder: PTSamDecoderONNX):
    """Test the ONNX-exportable decoder with dummy data."""
    onnx_decoder.eval()
    
    # Create dummy inputs
    batch_size = 1
    image_size = (1024, 1024)
    original_size = (480, 640)
    
    preprocessed_image = torch.zeros((batch_size, 3, *image_size), dtype=torch.float32)
    point_coords = torch.tensor([[[512, 512], [256, 256]]], dtype=torch.float32)  # 2 points
    point_labels = torch.tensor([[1, 1]], dtype=torch.float32)  # Both positive
    similarity_map = torch.rand(original_size, dtype=torch.float32)
    original_size_tensor = torch.tensor(original_size, dtype=torch.float32)
    
    print("Running inference with dummy data...")
    
    with torch.no_grad():
        masks, scores, used_points = onnx_decoder(
            preprocessed_image,
            point_coords,
            point_labels,
            similarity_map,
            original_size_tensor,
        )
    
    print(f"✅ Inference successful!")
    print(f"   - Output masks shape: {masks.shape}")
    print(f"   - Output scores shape: {scores.shape}")
    print(f"   - Output points shape: {used_points.shape}")
    
    if masks.shape[0] > 0:
        print(f"   - First mask area: {masks[0].sum().item()} pixels")
        print(f"   - First mask score: {scores[0].item():.3f}")


def compare_with_original():
    """Compare ONNX version with original implementation (if needed)."""
    # This could be implemented to validate that the ONNX version
    # produces similar results to the original PTSamDecoder
    pass


if __name__ == "__main__":
    export_sam_decoder_onnx()
