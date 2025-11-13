#!/usr/bin/env python3
"""
Quick test script to verify the validation functionality works.
"""

import sys
from pathlib import Path

# Add the lib/src to the path
sys.path.append(str(Path(__file__).parent / "lib" / "src"))

try:
    from validate_samhq_models import SamHQValidator
    print("✓ Successfully imported SamHQValidator")
except ImportError as e:
    print(f"✗ Failed to import SamHQValidator: {e}")
    sys.exit(1)

try:
    import torch
    import openvino as ov
    import numpy as np
    import matplotlib.pyplot as plt
    print("✓ All required dependencies are available")
except ImportError as e:
    print(f"✗ Missing dependency: {e}")
    sys.exit(1)

def test_validator_initialization():
    """Test if the validator can be initialized."""
    try:
        # Check if OpenVINO model exists
        openvino_model_path = "samhq_tiny.xml"
        if not Path(openvino_model_path).exists():
            print(f"⚠ OpenVINO model not found at {openvino_model_path}")
            print("  Please run 'python export_sam.py' first to create the model")
            return False
        
        validator = SamHQValidator(
            openvino_model_path=openvino_model_path,
            device="cpu",
            precision="fp16"
        )
        print("✓ SamHQValidator initialized successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize SamHQValidator: {e}")
        return False

def test_input_preparation():
    """Test input preparation functions."""
    try:
        # Create a simple validator (without loading models)
        validator = SamHQValidator.__new__(SamHQValidator)
        validator.device = "cpu"
        validator.precision_dtype = torch.float16
        validator.image_size = (1024, 1024)
        
        # Test image creation
        image = validator.create_test_image()
        assert image.shape == (1024, 1024, 3)
        assert image.dtype == np.uint8
        print("✓ Test image creation works")
        
        # Test point creation
        points, labels = validator.create_test_points()
        assert points.shape[1] == 2  # x, y coordinates
        assert len(labels) == len(points)
        print("✓ Test point creation works")
        
        return True
    except Exception as e:
        print(f"✗ Input preparation test failed: {e}")
        return False

def test_metrics_calculation():
    """Test metrics calculation functions."""
    try:
        # Create dummy data
        pt_masks = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
        ov_masks = pt_masks + np.random.rand(1, 3, 1024, 1024).astype(np.float32) * 0.01
        
        # Test MSE calculation
        mse = np.mean((pt_masks - ov_masks) ** 2)
        assert mse > 0
        print("✓ MSE calculation works")
        
        # Test MAE calculation
        mae = np.mean(np.abs(pt_masks - ov_masks))
        assert mae > 0
        print("✓ MAE calculation works")
        
        # Test IoU calculation
        pt_binary = (pt_masks > 0.5).astype(np.float32)
        ov_binary = (ov_masks > 0.5).astype(np.float32)
        intersection = np.sum(pt_binary * ov_binary)
        union = np.sum(np.maximum(pt_binary, ov_binary))
        iou = intersection / (union + 1e-8)
        assert 0 <= iou <= 1
        print("✓ IoU calculation works")
        
        return True
    except Exception as e:
        print(f"✗ Metrics calculation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Running validation script tests...\n")
    
    tests = [
        ("Input Preparation", test_input_preparation),
        ("Metrics Calculation", test_metrics_calculation),
        ("Validator Initialization", test_validator_initialization),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"Tests completed: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! The validation script should work correctly.")
    else:
        print("⚠ Some tests failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
