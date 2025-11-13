#!/usr/bin/env python3
"""Test script to verify ONNX tracing capability."""

import torch
import sys
import os

# Add the lib directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib', 'src'))

def test_tracing():
    """Test if the model can be traced with torch.jit.trace."""
    try:
        from getiprompt.components.segmenters.sam_decoder_pt_onnx import PTSamDecoderONNX
        from getiprompt.foundation.models import load_sam_model
        from getiprompt.utils.constants import SAMModelName
        
        print("Loading SAM model...")
        sam_predictor = load_sam_model(
            SAMModelName.SAM_HQ_TINY,
            device="cpu",
            precision="fp32",
            apply_onnx_patches=True,
        )
        
        print("Creating ONNX decoder...")
        onnx_decoder = PTSamDecoderONNX(sam_predictor=sam_predictor)
        onnx_decoder.eval()
        
        # Create dummy inputs
        print("Creating dummy inputs...")
        num_classes = 1
        max_points = 10
        image_size = (1024, 1024)
        original_size = (480, 640)
        
        image = torch.zeros((1, 3, image_size[0], image_size[1]), dtype=torch.float32)
        point_coords = torch.rand((num_classes, max_points, 2), dtype=torch.float32) * 1024
        point_scores = torch.rand((num_classes, max_points, 1), dtype=torch.float32)
        point_labels = torch.ones((num_classes, max_points, 1), dtype=torch.float32)
        class_point_coords = torch.cat([point_coords, point_scores, point_labels], dim=-1)
        similarity_maps = torch.rand(num_classes, original_size[0], original_size[1], dtype=torch.float32)
        original_size_tensor = torch.tensor(original_size, dtype=torch.float32)
        
        # Test forward pass
        print("Testing forward pass...")
        with torch.no_grad():
            output = onnx_decoder(image, class_point_coords, similarity_maps, original_size_tensor)
            print(f"Forward pass successful! Output shapes: {[o.shape for o in output]}")
        
        # Test tracing
        print("Testing torch.jit.trace...")
        try:
            traced_model = torch.jit.trace(
                onnx_decoder,
                (image, class_point_coords, similarity_maps, original_size_tensor)
            )
            print("✅ Tracing successful!")
            
            # Test traced model
            print("Testing traced model...")
            with torch.no_grad():
                traced_output = traced_model(image, class_point_coords, similarity_maps, original_size_tensor)
                print(f"Traced model output shapes: {[o.shape for o in traced_output]}")
                print("✅ Traced model execution successful!")
                
        except Exception as e:
            print(f"❌ Tracing failed: {e}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tracing()
    sys.exit(0 if success else 1)



matcher.export(path = "model/")

-> dino.xml
-> sam.xml

dino = load_model(path = "dino.xml")
sam = load_model(path = "sam.xml")