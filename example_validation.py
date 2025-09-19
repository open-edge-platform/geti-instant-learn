#!/usr/bin/env python3
"""
Simple example script showing how to use the SamHQ validation script.

This script demonstrates basic usage of the validation functionality.
"""
import numpy as np
from validate_samhq_models import SamHQValidator
import cv2


def simple_validation_example():
    """Run a simple validation example."""
    print("Running simple SamHQ validation example...")
    
    # Initialize validator
    validator = SamHQValidator(
        openvino_model_path="samhq_tiny.xml",
        device="cpu",
        precision="fp32"
    )
    
    # test image
    image = cv2.imread("data/target/images/can_05.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Test1: Create test points
    # points = np.array(
    #     [
    #         [780, 1020],
    #         [1160, 260],
    #         [1192, 260],
    #     ],
    #     dtype=np.float32
    # )  
    # labels = np.array([1, 0, 0])  # Foreground point

    # Test2: Create test points
    points = np.array(
        [
            [812, 988],
            [1160, 260],
            [1192, 260],
        ],
        dtype=np.float32
    )  
    labels = np.array([1, 0, 0])  # Foreground point
    
    # Prepare inputs
    pt_inputs = validator.prepare_pytorch_inputs(image, points, labels, original_size)
    ov_inputs = validator.prepare_openvino_inputs(image, points, labels, original_size)
    
    # Run inference
    print("Running PyTorch inference...")
    pt_outputs = validator.run_pytorch_inference(pt_inputs)
    
    print("Running OpenVINO inference...")
    ov_outputs = validator.run_openvino_inference(ov_inputs)
    
    # Compare outputs
    metrics = validator.compare_outputs(pt_outputs, ov_outputs)
    
    # Print results
    print("\nComparison Results:")
    print("-" * 40)
    for key, value in metrics.items():
        if 'mse' in key or 'mae' in key or 'max_diff' in key:
            print(f"{key:25s}: {value:.6f}")
        elif 'iou' in key:
            print(f"{key:25s}: {value:.4f}")
    
    # Visualize results
    print("\nGenerating visualization...")
    validator.visualize_comparison(
        image, points, labels, pt_outputs, ov_outputs,
        "simple_validation_example.png"
    )
    
    print("Validation complete! Check 'simple_validation_example.png' for visualization.")


if __name__ == "__main__":
    simple_validation_example()
