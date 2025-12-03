"""Test script to verify tracing functionality."""

from pathlib import Path

import torch
from getiprompt.components.sam.base import load_sam_model
from getiprompt.utils.constants import Backend, SAMModelName
from getiprompt.components.mask_decoder import SamDecoder
from getiprompt.components.encoders import ImageEncoder

from getiprompt.data.utils.image import read_image

import copy


def export_sam(exported_path: Path, backend: Backend = Backend.OPENVINO) -> Path:
    """Test if the model can be traced with torch.jit.trace."""
    # 1. Load PyTorch model
    predictor = load_sam_model(
        SAMModelName.SAM_HQ_TINY,
        backend=Backend.PYTORCH,
        precision="fp32",
        device="cpu",
    )

    # 2. Export (choose one)
    exported_path = predictor.export(exported_path, backend=backend)  # Simple ✅
    return exported_path


def infer_with_openvino(image_path, point_prompts, backend: Backend = Backend.OPENVINO):
    point_prompts = copy.deepcopy(point_prompts)
    exported_path = Path("exported") / "exported_sam.xml"
    image = read_image(image_path)
    if not exported_path.exists():
        exported_path = export_sam(Path("exported"), backend=backend)
    predictor = load_sam_model(
        SAMModelName.SAM_HQ_TINY,
        backend=Backend.OPENVINO,
        model_path=exported_path,
        device="cpu",
    )
    sam_decoder = SamDecoder(predictor, target_length=1024)
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
        backend=Backend.PYTORCH,
        precision="fp32",
        device="cpu",
    )
    sam_decoder = SamDecoder(predictor, target_length=1024)
    predictions = sam_decoder.forward(
        [image],
        point_prompts,
    )
    return predictions


def export_image_encoder(
    model_id: str = "dinov3_large",
    exported_path: Path = Path("exported"),
    backend: Backend = Backend.OPENVINO,
):
    encoder = ImageEncoder(
        model_id=model_id,
        backend=Backend.TIMM,
        device="cpu",
        precision="fp32",
    )
    return encoder.export(exported_path, backend=backend)


def infer_image_encoder_with_ov(
    image_path: Path,
    model_id: str = "dinov3_large",
    backend: Backend = Backend.OPENVINO,
):
    exported_path = Path("exported") / "image_encoder.xml"
    if not exported_path.exists():
        exported_path = export_image_encoder(model_id, backend=backend)
    encoder = ImageEncoder(
        model_id=model_id,
        backend=Backend.OPENVINO,
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

    point_prompts = [
        {0: torch.tensor([[78, 152, 1, 1]], dtype=torch.float32)},
    ]

    ov_predictions = infer_with_openvino(image_path, point_prompts)
    pt_predictions = infer_with_pytorch(image_path, point_prompts)

    # # compare mask
    ov_masks = ov_predictions[0]["pred_masks"].cpu().numpy()
    pt_masks = pt_predictions[0]["pred_masks"].cpu().numpy()
    print((ov_masks.astype(int) - pt_masks.astype(int)).sum())

    infer_image_encoder_with_ov(image_path)
