"""Test script to verify tracing functionality."""

from pathlib import Path

import torch

from getiprompt.models.foundation import load_sam_model
from getiprompt.utils.constants import SAMModelName
from getiprompt.components.mask_decoder import SamDecoder
from getiprompt.components.encoders import PyTorchImageEncoder, AVAILABLE_IMAGE_ENCODERS, OpenVINOImageEncoder

from getiprompt.data.utils.image import read_image

import copy

def export_sam(exported_path: Path):
    """Test if the model can be traced with torch.jit.trace."""
    # 1. Load PyTorch model
    predictor = load_sam_model(
        SAMModelName.SAM_HQ_TINY, 
        backend="pytorch",
        precision="fp32",
        device="cpu"
    )

    # 2. Export (choose one)
    ov_path = predictor.export(exported_path)  # Simple ✅
    return ov_path


def infer_with_openvino(image_path, point_prompts):
    point_prompts = copy.deepcopy(point_prompts)
    exported_path = Path("exported") / "exported_sam.xml"
    image = read_image(image_path)
    if not exported_path.exists():
        exported_path = export_sam(Path("exported"))
    predictor = load_sam_model(
        SAMModelName.SAM_HQ_TINY,
        backend="openvino",
        model_path=exported_path,
        device="cpu"
    )
    sam_decoder = SamDecoder(predictor)
    predictions = sam_decoder.forward(
        [image],
        point_prompts,
    )
    return predictions


def infer_with_pytorch(image_path, point_prompts):
    point_prompts = copy.deepcopy(point_prompts)
    image = read_image(image_path)
    predictor = load_sam_model(
        SAMModelName.SAM_HQ_TINY,
        backend="pytorch",
        precision="fp32",
        device="cpu"
    )
    sam_decoder = SamDecoder(predictor)
    predictions = sam_decoder.forward(
        [image],
        point_prompts,
    )
    return predictions


def export_image_encoder(
    model_id: str = "dinov3_large",
    exported_path: Path = Path("exported"),
):
    encoder = PyTorchImageEncoder(
        model_id=model_id, 
        device="cpu", 
        precision="fp32",
    )
    encoder.export(exported_path)
    return exported_path


def infer_image_encoder_with_ov(
    image_path: Path,
    model_id: str = "dinov3_large",    
):
    exported_path = Path("exported") / "image_encoder.xml"
    if not exported_path.exists():
        exported_path = export_image_encoder(model_id)
    encoder = OpenVINOImageEncoder(
        model_path=exported_path,
        device="cpu", 
        input_size=518,
    )
    image = read_image(image_path)
    features = encoder([image])
    print(features.shape)
    return features


if __name__ == "__main__":
    image_path = "library/tests/assets/fss-1000/images/apple/1.jpg"
    # point_prompts = [
    #     {0: torch.tensor([[78, 152, 1, 1]], dtype=torch.float32)},
    # ]

    # ov_predictions = infer_with_openvino(image_path, point_prompts)
    # pt_predictions = infer_with_pytorch(image_path, point_prompts)
    
    # # compare mask
    # ov_masks = ov_predictions[0]['pred_masks'].cpu().numpy()
    # pt_masks = pt_predictions[0]['pred_masks'].cpu().numpy()
    # print((ov_masks.astype(int) - pt_masks.astype(int)).sum())

    infer_image_encoder_with_ov(image_path)