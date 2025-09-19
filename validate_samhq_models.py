#!/usr/bin/env python3
"""
Validation script to compare SamHQ PyTorch and OpenVINO models.

This script validates the input and output of SamHQ models by:
1. Loading both PyTorch and OpenVINO models
2. Running inference on the same inputs
3. Comparing outputs using various metrics
4. Providing performance benchmarks
5. Visualizing differences
"""

import argparse
import time
from typing import Dict, Tuple, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openvino as ov
import torch

from getiprompt.models.models import load_sam_model
from getiprompt.utils.constants import SAMModelName
from getiprompt.utils.utils import precision_to_torch_dtype
from segment_anything_hq.utils.transforms import ResizeLongestSide


class SamHQValidator:
    """Validator for comparing PyTorch and OpenVINO SamHQ models."""
    
    def __init__(
        self,
        input_image_path: str = None,
        points: np.ndarray = None,
        labels: np.ndarray = None,
        pytorch_model_path: str = None,
        openvino_model_path: str = "samhq_tiny.xml",
        device: str = "cpu",
        pytorch_device: str = "cuda",
        precision: str = "fp16",
        transform = ResizeLongestSide(1024),
    ):
        """
        Initialize the validator.
        
        Args:
            pytorch_model_path: Path to PyTorch model (if None, loads from registry)
            openvino_model_path: Path to OpenVINO IR model
            device: Device to run inference on
            precision: Model precision
        """
        self.device = device
        self.precision = precision
        self.precision_dtype = precision_to_torch_dtype(precision)
        self.input_image_path = input_image_path
        self.points = points
        self.labels = labels
        self.transform = transform
        self.pytorch_device = pytorch_device
        # Load PyTorch model
        print("Loading PyTorch SamHQ model...")
        self.pytorch_predictor = load_sam_model(
            SAMModelName.SAM_HQ_TINY,
            device=pytorch_device,
            precision=precision,
            compile_models=False,
            benchmark_inference_speed=False,
        )
        
        # Load OpenVINO model
        print("Loading OpenVINO SamHQ model...")
        self.openvino_core = ov.Core()
        self.openvino_model = self.openvino_core.read_model(openvino_model_path)
        self.openvino_compiled_model = self.openvino_core.compile_model(
            self.openvino_model,
            device.upper(),
        )
        
        # Get input/output info
        self.openvino_inputs = self.openvino_compiled_model.inputs
        self.openvino_outputs = self.openvino_compiled_model.outputs
        
        print(f"OpenVINO inputs: {[inp.get_names() for inp in self.openvino_inputs]}")
        print(f"OpenVINO outputs: {[out.get_names() for out in self.openvino_outputs]}")
        
        # Test image size
        self.image_size = (1024, 1024)
        
    def create_test_image(
        self, 
        size: Tuple[int, int] = (1024, 1024)
    ) -> np.ndarray:
        """Create a test image with geometric shapes."""
        if self.input_image_path:
            image = cv2.imread(self.input_image_path)
            original_size = image.shape[:2]
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image, original_size
        
        image = np.zeros((*size, 3), dtype=np.uint8)

        # Add some geometric shapes
        cv2.rectangle(image, (200, 200), (400, 400), (255, 0, 0), -1)  # Red rectangle
        cv2.circle(image, (700, 300), 100, (0, 255, 0), -1)  # Green circle
        cv2.rectangle(image, (300, 600), (600, 800), (0, 0, 255), -1)  # Blue rectangle
        
        # Add some noise
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        original_size = image.shape[:2]
        return image, original_size
    
    def create_test_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create test point coordinates and labels."""

        if self.points is not None and self.labels is not None:
            return self.points, self.labels

        # Create points in different regions
        points = np.array([
            [300, 300],  # Inside red rectangle
            [700, 300],  # Inside green circle
            [450, 700],  # Inside blue rectangle
        ])
        labels = np.array([1, 1, 1])
        return points, labels
    
    def prepare_pytorch_inputs(
        self, 
        image: np.ndarray, 
        points: np.ndarray, 
        labels: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Prepare inputs for PyTorch model."""
        # Set image in predictor
        self.pytorch_predictor.set_image(image, image_format="RGB")
        return {
            'point_coords': points,
            'point_labels': labels,
            'multimask_output': True,
            'return_logits': True,
            'original_image_size': original_size
        }
    
    def prepare_openvino_inputs(
        self, 
        image: np.ndarray, 
        points: np.ndarray, 
        labels: np.ndarray,
        original_size: Tuple[int, int]
    ) -> Dict[str, np.ndarray]:
        """Prepare inputs for OpenVINO model."""
        # Convert image to tensor format [1, 3, H, W]
        image = self.transform.apply_image(image)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)
        image_tensor = image_tensor.to(self.precision_dtype)
        
        # Convert points to tensor format [1, N, 2]
        points = self.transform.apply_coords(points, original_size)
        points_tensor = torch.from_numpy(points).float().unsqueeze(0)
        points_tensor = points_tensor.to(self.precision_dtype)
        
        # Convert labels to tensor format [1, N]
        labels_tensor = torch.from_numpy(labels).int()
        labels_tensor = labels_tensor.unsqueeze(0)
        
        # Original image size
        original_size = np.array(original_size, dtype=np.int32)
        
        return {
            'image': image_tensor.numpy(),
            'point_coords': points_tensor.numpy(),
            'point_labels': labels_tensor.numpy(),
            'original_image_size': original_size
        }
    
    def run_pytorch_inference(self, inputs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run PyTorch inference."""
        with torch.no_grad():
            outputs = self.pytorch_predictor.predict(
                point_coords=inputs['point_coords'],
                point_labels=inputs['point_labels'],
            )
            
        # Unpack outputs
        masks, iou_predictions, low_res_masks = outputs
        
        return {
            'masks': masks,
            'iou_predictions': iou_predictions,
            'low_res_masks': low_res_masks,
        }
    
    def run_openvino_inference(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run OpenVINO inference."""
        outputs = self.openvino_compiled_model(inputs)
        
        # Convert outputs to numpy arrays
        result = {}
        for i, output in enumerate(self.openvino_outputs):
            output_name = list(output.get_names())[0]
            result[output_name] = outputs[output]

        result["masks"] = result["masks"][0]
        result["iou_predictions"] = result["iou_predictions"][0]
        result["low_res_masks"] = result["low_res_masks"][0]
        return result
    
    def compare_outputs(
            self, 
            pytorch_outputs: Dict[str, torch.Tensor], 
            openvino_outputs: Dict[str, np.ndarray]
        ) -> Dict[str, float]:
        """Compare outputs from both models."""
        metrics = {}
        
        # Compare masks
        if 'masks' in pytorch_outputs and 'masks' in openvino_outputs:
            pt_masks = pytorch_outputs['masks']
            ov_masks = openvino_outputs['masks']
            
            # Ensure same shape
            if pt_masks.shape != ov_masks.shape:
                print(f"Warning: Shape mismatch - PyTorch: {pt_masks.shape}, OpenVINO: {ov_masks.shape}")
                min_shape = tuple(min(pt_masks.shape[i], ov_masks.shape[i]) for i in range(len(pt_masks.shape)))
                pt_masks = pt_masks[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
                ov_masks = ov_masks[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            
            # Convert to same dtype for comparison
            pt_masks = pt_masks.astype(np.float32)
            ov_masks = ov_masks.astype(np.float32)
            
            # Calculate metrics
            mse = np.mean((pt_masks - ov_masks) ** 2)
            mae = np.mean(np.abs(pt_masks - ov_masks))
            max_diff = np.max(np.abs(pt_masks - ov_masks))
            
            metrics['masks_mse'] = mse
            metrics['masks_mae'] = mae
            metrics['masks_max_diff'] = max_diff
            
            # IoU comparison (for binary masks)
            pt_binary = (pt_masks > 0.5).astype(np.float32)
            ov_binary = (ov_masks > 0.5).astype(np.float32)
            
            intersection = np.sum(pt_binary * ov_binary)
            union = np.sum(np.maximum(pt_binary, ov_binary))
            iou = intersection / (union + 1e-8)
            metrics['masks_iou'] = iou
        
        # Compare IoU predictions
        if 'iou_predictions' in pytorch_outputs and 'iou_predictions' in openvino_outputs:
            pt_iou = pytorch_outputs['iou_predictions']
            ov_iou = openvino_outputs['iou_predictions']
            
            # Ensure same shape
            if pt_iou.shape != ov_iou.shape:
                min_shape = tuple(min(pt_iou.shape[i], ov_iou.shape[i]) for i in range(len(pt_iou.shape)))
                pt_iou = pt_iou[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
                ov_iou = ov_iou[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            
            pt_iou = pt_iou.astype(np.float32)
            ov_iou = ov_iou.astype(np.float32)
            
            mse = np.mean((pt_iou - ov_iou) ** 2)
            mae = np.mean(np.abs(pt_iou - ov_iou))
            max_diff = np.max(np.abs(pt_iou - ov_iou))
            
            metrics['iou_predictions_mse'] = mse
            metrics['iou_predictions_mae'] = mae
            metrics['iou_predictions_max_diff'] = max_diff
        
        # Compare low resolution masks
        if 'low_res_masks' in pytorch_outputs and 'low_res_masks' in openvino_outputs:
            pt_low_res = pytorch_outputs['low_res_masks']
            ov_low_res = openvino_outputs['low_res_masks']
            
            # Ensure same shape
            if pt_low_res.shape != ov_low_res.shape:
                min_shape = tuple(min(pt_low_res.shape[i], ov_low_res.shape[i]) for i in range(len(pt_low_res.shape)))
                pt_low_res = pt_low_res[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
                ov_low_res = ov_low_res[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            
            pt_low_res = pt_low_res.astype(np.float32)
            ov_low_res = ov_low_res.astype(np.float32)
            
            mse = np.mean((pt_low_res - ov_low_res) ** 2)
            mae = np.mean(np.abs(pt_low_res - ov_low_res))
            max_diff = np.max(np.abs(pt_low_res - ov_low_res))
            
            metrics['low_res_masks_mse'] = mse
            metrics['low_res_masks_mae'] = mae
            metrics['low_res_masks_max_diff'] = max_diff
        
        return metrics
    
    def visualize_comparison(self, 
                           image: np.ndarray,
                           points: np.ndarray,
                           labels: np.ndarray,
                           pytorch_outputs: Dict[str, torch.Tensor],
                           openvino_outputs: Dict[str, np.ndarray],
                           save_path: str = None):
        """Visualize comparison between PyTorch and OpenVINO outputs."""
        fig, axes = plt.subplots(2, 2, figsize=(20, 10))
        fig.suptitle('SamHQ PyTorch vs OpenVINO Comparison', fontsize=16)
        
        # Original image with points
        axes[0, 0].imshow(image)
        axes[0, 0].scatter(points[:, 0], points[:, 1], c='red', s=100, marker='x')
        axes[0, 0].set_title('Input Image with Points')
        axes[0, 0].axis('off')
        
        # PyTorch masks
        if 'masks' in pytorch_outputs:
            pt_masks = pytorch_outputs['masks']
            for i in range(min(3, pt_masks.shape[0])):
                axes[0, 1].imshow(pt_masks[i], cmap='viridis', alpha=0.7)
            axes[0, 1].set_title('PyTorch Masks')
            axes[0, 1].axis('off')
        
        # OpenVINO masks
        if 'masks' in openvino_outputs:
            ov_masks = openvino_outputs['masks'][0]
            axes[1, 0].imshow(ov_masks, cmap='viridis', alpha=0.7)
            axes[1, 0].set_title('OpenVINO Masks')
            axes[1, 0].axis('off')
        
        # Difference
        if 'masks' in pytorch_outputs and 'masks' in openvino_outputs:
            pt_masks = pytorch_outputs['masks']
            ov_masks = openvino_outputs['masks']
            
            # Ensure same shape
            if pt_masks.shape != ov_masks.shape:
                min_shape = tuple(min(pt_masks.shape[i], ov_masks.shape[i]) for i in range(len(pt_masks.shape)))
                pt_masks = pt_masks[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
                ov_masks = ov_masks[tuple(slice(0, min_shape[i]) for i in range(len(min_shape)))]
            
            diff = (pt_masks ^ ov_masks)
            axes[1, 1].imshow(diff[0], cmap='hot')
            axes[1, 1].set_title('Absolute Difference')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def run_validation(
        self, 
        save_visualizations: bool = True
    ) -> Dict[str, Any]:
        """Run comprehensive validation."""
        print("Starting SamHQ model validation...")
        
        results = {
            'test_cases': [],
            'performance_benchmark': {},
            'summary_metrics': {}
        }
            
        # Create test data
        image, original_size = self.create_test_image()
        points, labels = self.create_test_points()
        
        # Prepare inputs
        pt_inputs = self.prepare_pytorch_inputs(image, points, labels, original_size)
        ov_inputs = self.prepare_openvino_inputs(image, points, labels, original_size)
        
        # Run inference
        pt_outputs = self.run_pytorch_inference(pt_inputs)
        ov_outputs = self.run_openvino_inference(ov_inputs)
        
        # Compare outputs
        metrics = self.compare_outputs(pt_outputs, ov_outputs)
        
        # Store results
        test_case = {
            'image_shape': image.shape,
            'points': points.tolist(),
            'labels': labels.tolist(),
            'pytorch_inference_time': pt_outputs['inference_time'],
            'openvino_inference_time': ov_outputs['inference_time'],
            'metrics': metrics
        }
        results['test_cases'].append(test_case)
        
        # Save visualization for first test case
        if save_visualizations:
            self.visualize_comparison(
                image, points, labels, pt_outputs, ov_outputs,
                f"samhq_comparison_test_{i}.png"
            )
        
        # Calculate summary metrics
        all_metrics = [tc['metrics'] for tc in results['test_cases']]
        summary = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            summary[f'{key}_mean'] = np.mean(values)
            summary[f'{key}_std'] = np.std(values)
            summary[f'{key}_min'] = np.min(values)
            summary[f'{key}_max'] = np.max(values)
        
        results['summary_metrics'] = summary
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print validation results in a formatted way."""
        print("\n" + "="*80)
        print("SamHQ PyTorch vs OpenVINO Validation Results")
        print("="*80)
        
        # Summary metrics
        print("\nSUMMARY METRICS:")
        print("-" * 40)
        summary = results['summary_metrics']
        
        for key, value in summary.items():
            if 'mse' in key:
                print(f"{key:30s}: {value:.6f}")
            elif 'mae' in key:
                print(f"{key:30s}: {value:.6f}")
            elif 'max_diff' in key:
                print(f"{key:30s}: {value:.6f}")
            elif 'iou' in key:
                print(f"{key:30s}: {value:.4f}")
        
        # Performance benchmark
        print("\nPERFORMANCE BENCHMARK:")
        print("-" * 40)
        perf = results['performance_benchmark']
        print(f"{'PyTorch Mean Time':30s}: {perf['pytorch_mean_time']:.4f}s")
        print(f"{'PyTorch Std Time':30s}: {perf['pytorch_std_time']:.4f}s")
        print(f"{'OpenVINO Mean Time':30s}: {perf['openvino_mean_time']:.4f}s")
        print(f"{'OpenVINO Std Time':30s}: {perf['openvino_std_time']:.4f}s")
        print(f"{'Speedup (PT/OV)':30s}: {perf['speedup']:.2f}x")
        
        # Individual test cases
        print("\nINDIVIDUAL TEST CASES:")
        print("-" * 40)
        for tc in results['test_cases']:
            print(f"\nTest Case {tc['test_id'] + 1}:")
            print(f"  Points: {tc['points']}")
            print(f"  Labels: {tc['labels']}")
            print(f"  PyTorch Time: {tc['pytorch_inference_time']:.4f}s")
            print(f"  OpenVINO Time: {tc['openvino_inference_time']:.4f}s")
            
            metrics = tc['metrics']
            for key, value in metrics.items():
                if 'mse' in key or 'mae' in key or 'max_diff' in key:
                    print(f"  {key:25s}: {value:.6f}")
                elif 'iou' in key:
                    print(f"  {key:25s}: {value:.4f}")
