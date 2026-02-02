"""Test script to verify tracing functionality."""

from pathlib import Path

import torch
from getiprompt.data import Sample, Batch
from getiprompt.models import Matcher
from getiprompt.components.sam import SAMPredictor, SamDecoder
from getiprompt.utils.constants import SAMModelName
from getiprompt.data.utils.image import read_image
from getiprompt.utils.constants import Backend
import numpy as np
import openvino
import onnxruntime as ort
from getiprompt.utils.utils import device_to_openvino_device, precision_to_openvino_type
import onnx


def export_sam_decoder(export_dir: Path, backend: Backend = Backend.ONNX) -> Path:
    """Export just the SamDecoder module for debugging."""
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize SAM predictor
    predictor = SAMPredictor(SAMModelName.SAM_HQ_TINY, device="cpu")
    
    # Create SamDecoder
    decoder = SamDecoder(
        sam_predictor=predictor,
        confidence_threshold=0.38,
        nms_iou_threshold=0.1,
        max_masks_per_category=40,
        use_mask_refinement=False,
        merge_masks_per_class=True,
    )
    decoder.eval()
    
    # Create dummy inputs matching forward_export signature
    h, w = 224, 224
    num_categories = 1
    max_points = 10
    feat_size = 14  # typical feature map size
    
    dummy_image = torch.randn(1, 3, h, w)
    dummy_category_ids = torch.tensor([1], dtype=torch.int64)
    dummy_point_prompts = torch.zeros(num_categories, max_points, 4)
    # Add one foreground point and one background point
    dummy_point_prompts[0, 0] = torch.tensor([100, 100, 0.9, 1])  # fg point
    dummy_point_prompts[0, 1] = torch.tensor([10, 10, 0.5, -1])   # bg point
    dummy_similarities = torch.rand(num_categories, feat_size, feat_size)
    
    onnx_path = export_dir / "sam_decoder.onnx"
    
    # Export to ONNX
    with torch.no_grad():
        torch.onnx.export(
            decoder,
            (dummy_image, dummy_category_ids, dummy_point_prompts, dummy_similarities),
            str(onnx_path),
            input_names=["image", "category_ids", "point_prompts", "similarities"],
            output_names=["masks", "scores", "labels"],
            opset_version=20,
            verbose=True,
        )
    
    print(f"Exported ONNX to {onnx_path}")
    
    if backend == Backend.OPENVINO:
        # Convert ONNX to OpenVINO
        import openvino as ov
        core = ov.Core()
        ov_model = core.read_model(str(onnx_path))
        ov.save_model(ov_model, str(export_dir / "sam_decoder.xml"))
        print(f"Exported OpenVINO to {export_dir / 'sam_decoder.xml'}")
        return export_dir / "sam_decoder.xml"
    
    return onnx_path


def infer_sam_decoder_onnx(
    exported_path: Path,
    image: torch.Tensor,
    category_ids: torch.Tensor,
    point_prompts: torch.Tensor,
    similarities: torch.Tensor,
):
    """Run SamDecoder inference with ONNX Runtime."""
    session = ort.InferenceSession(str(exported_path), providers=["CPUExecutionProvider"])
    
    outputs = session.run(
        None,
        {
            "image": image.numpy().astype(np.float32),
            "category_ids": category_ids.numpy().astype(np.int64),
            "point_prompts": point_prompts.numpy().astype(np.float32),
            "similarities": similarities.numpy().astype(np.float32),
        },
    )
    
    masks, scores, labels = outputs
    print(f"ONNX - Masks: {masks.shape}, Scores: {scores.shape}, Labels: {labels.shape}")
    return masks, scores, labels


def infer_sam_decoder_openvino(
    exported_path: Path,
    image: torch.Tensor,
    category_ids: torch.Tensor,
    point_prompts: torch.Tensor,
    similarities: torch.Tensor,
):
    """Run SamDecoder inference with OpenVINO."""
    core = openvino.Core()
    model = core.read_model(str(exported_path))
    compiled = core.compile_model(model, "CPU")
    
    outputs = compiled({
        "image": image.numpy().astype(np.float32),
        "category_ids": category_ids.numpy().astype(np.int64),
        "point_prompts": point_prompts.numpy().astype(np.float32),
        "similarities": similarities.numpy().astype(np.float32),
    })
    
    masks, scores, labels = outputs.values()
    print(f"OpenVINO - Masks: {masks.shape}, Scores: {scores.shape}, Labels: {labels.shape}")
    return masks, scores, labels


def export_matcher(exported_path: Path, backend: Backend = Backend.ONNX) -> Path:
    ref_image = read_image("library/examples/assets/fss-1000/images/apple/1.jpg")

    # Initialize SAM predictor (auto-downloads weights)
    predictor = SAMPredictor(SAMModelName.SAM_HQ_TINY, device="cuda")

    # Set image and generate mask from a point click
    predictor.set_image(ref_image)
    ref_mask, _, _ = predictor.forward(
        point_coords=torch.tensor([[[51, 150]]], device="cuda"),  # Click on apple
        point_labels=torch.tensor([[1]], device="cuda"),           # 1 = foreground
        multimask_output=False,
    )

    matcher = Matcher(
        device="cpu", 
        precision="fp32", 
        use_mask_refinement=False,
        encoder_model="dinov3_small",
    )

    # Create reference sample with the generated mask
    ref_sample = Sample(
        image=ref_image,
        masks=ref_mask[0].cpu(),
        categories=["apple"],
        category_ids=[1]
    )

    # Fit on reference
    ref_features = matcher.fit(Batch.collate([ref_sample]))

    return matcher.export(
        reference_features=ref_features, 
        export_dir=exported_path,
        backend=backend,
    )



def infer_with_openvino(
    exported_path: Path,
    image_path, 
    device: str = "CPU",
    precision: str = "fp32",
):
    target_image = read_image(image_path)
    core = openvino.Core()
    ov_device = device_to_openvino_device(device)
    # core.set_property(ov_device, {hint.inference_precision: precision_to_openvino_type(precision)})
    ov_model = core.read_model(exported_path)
    compiled_model = core.compile_model(ov_model, ov_device)
    outputs = compiled_model(target_image.numpy()[None, ...].astype(np.float32))
    masks, scores, labels = outputs.values()
    print(f"Masks shape: {masks.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Labels shape: {labels.shape}")

    return masks, scores, labels


def infer_with_onnx(
    exported_path: Path,
    image_path,
    device: str = "cpu",
):
    """Run inference using ONNX Runtime.

    Args:
        exported_path: Path to the .onnx model file.
        image_path: Path to the input image.
        device: Device to run inference on ("cpu" or "cuda").

    Returns:
        Tuple of (masks, scores, labels) as numpy arrays.
    """
    model = onnx.load(exported_path)

    target_image = read_image(image_path)

    # Create ONNX Runtime session
    providers = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(exported_path), providers=providers)

    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Prepare input: add batch dimension and convert to numpy
    input_data = target_image.numpy()[None, ...].astype(np.float32)

    # Run inference
    outputs = session.run(output_names, {input_name: input_data})

    masks, scores, labels = outputs
    print(f"Masks shape: {masks.shape}")
    print(f"Scores shape: {scores.shape}")
    print(f"Labels shape: {labels.shape}")

    return masks, scores, labels


if __name__ == "__main__":
    export_dir = Path("./exported")
    
    # --- Export and test SamDecoder only (smaller graph for Netron) ---
    # print("\n=== Exporting SamDecoder ===")
    # export_sam_decoder(export_dir, backend=Backend.ONNX)
    
    # # Test inputs using real image
    # test_image = read_image("library/examples/assets/fss-1000/images/apple/1.jpg")
    # h, w = test_image.shape[1], test_image.shape[2]
    # test_image = test_image.unsqueeze(0)  # Add batch dim: [1, 3, H, W]
    
    # test_category_ids = torch.tensor([1], dtype=torch.int64)
    # test_point_prompts = torch.zeros(1, 10, 4)
    # test_point_prompts[0, 0] = torch.tensor([51, 150, 0.9, 1])  # fg point on apple
    # test_point_prompts[0, 1] = torch.tensor([10, 10, 0.5, -1])  # bg point
    # test_similarities = torch.ones(1, 14, 14)  # Still random for now
    
    # print("\n=== Testing SamDecoder ONNX ===")
    # infer_sam_decoder_onnx(
    #     export_dir / "sam_decoder.onnx",
    #     test_image, 
    #     test_category_ids, 
    #     test_point_prompts, 
    #     test_similarities,
    # )
    
    # print("\n=== Exporting SamDecoder to OpenVINO ===")
    # export_sam_decoder(export_dir, backend=Backend.OPENVINO)
    
    # print("\n=== Testing SamDecoder OpenVINO ===")
    # infer_sam_decoder_openvino(
    #     export_dir / "sam_decoder.xml",
    #     test_image, 
    #     test_category_ids, 
    #     test_point_prompts, 
    #     test_similarities,
    # )
    
    # --- Full Matcher export (commented out) ---
    # print("\n=== Exporting full Matcher ===")
    # exported_path = export_matcher(exported_path=export_dir, backend=Backend.ONNX)
    # infer_with_onnx(
    #     exported_path=export_dir / "matcher.onnx",
    #     image_path="library/examples/assets/fss-1000/images/apple/2.jpg",
    #     device="cpu",
    # )
    
    exported_path = export_matcher(exported_path=export_dir, backend=Backend.OPENVINO)
    infer_with_openvino(
        exported_path=export_dir / "matcher.xml",
        image_path="library/examples/assets/fss-1000/images/apple/2.jpg",
        device="CPU",
    )
