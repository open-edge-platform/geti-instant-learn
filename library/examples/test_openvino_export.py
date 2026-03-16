"""Test Matcher OpenVINO export: ONNX export on CUDA, OpenVINO conversion, and inference validation."""

import logging
import time
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPORT_DIR = Path(__file__).parent.parent / "exports" / "matcher_openvino"
ASSETS = Path(__file__).parent / "assets" / "coco"
REF_IMAGE = ASSETS / "000000286874.jpg"
REF_MASK = ASSETS / "000000286874_mask.png"
TARGET_IMAGES = [
    ASSETS / "000000390341.jpg",
    ASSETS / "000000173279.jpg",
    ASSETS / "000000267704.jpg",
]


def step_1_pytorch_predict():
    """Run PyTorch prediction on CUDA as baseline."""
    logger.info("=== Step 1: PyTorch CUDA prediction (baseline) ===")
    from instantlearn.data import Sample
    from instantlearn.models import Matcher

    model = Matcher(device="cuda", precision="bf16")
    ref_sample = Sample(image_path=str(REF_IMAGE), mask_paths=str(REF_MASK))
    model.fit(ref_sample)
    logger.info("Fit complete. Running predict on %d target images...", len(TARGET_IMAGES))

    predictions = model.predict([str(p) for p in TARGET_IMAGES])
    for i, pred in enumerate(predictions):
        masks = pred["pred_masks"]
        scores = pred["pred_scores"]
        labels = pred["pred_labels"]
        logger.info(
            "  Target %d: %d masks, scores=%s, labels=%s",
            i,
            masks.shape[0],
            scores.cpu().numpy().round(3),
            labels.cpu().numpy(),
        )

    # Save baseline for comparison
    baseline = {
        "masks": [p["pred_masks"].cpu().numpy() for p in predictions],
        "scores": [p["pred_scores"].cpu().numpy() for p in predictions],
        "labels": [p["pred_labels"].cpu().numpy() for p in predictions],
    }
    return model, baseline


def step_2_export_onnx(model):
    """Export to ONNX."""
    logger.info("=== Step 2: ONNX export ===")
    from instantlearn.utils.constants import Backend

    onnx_path = model.export(export_dir=EXPORT_DIR, backend=Backend.ONNX)
    logger.info("ONNX exported to: %s (%.1f MB)", onnx_path, onnx_path.stat().st_size / 1e6)
    return onnx_path


def step_3_validate_onnx(onnx_path, input_size):
    """Validate ONNX with ONNX Runtime."""
    logger.info("=== Step 3: ONNX Runtime validation ===")
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    outputs = session.run(None, {"target_image": dummy_input})
    logger.info("ONNX Runtime outputs: %d tensors", len(outputs))
    for i, out in enumerate(outputs):
        logger.info("  Output %d: shape=%s dtype=%s", i, out.shape, out.dtype)
    return outputs


def step_4_export_openvino(model):
    """Export to OpenVINO (ONNX → OpenVINO IR)."""
    logger.info("=== Step 4: OpenVINO export (FP32) ===")
    from instantlearn.utils.constants import Backend

    ov_path = model.export(export_dir=EXPORT_DIR, backend=Backend.OPENVINO, compress_to_fp16=False)
    logger.info("OpenVINO FP32 exported to: %s", ov_path)

    # Also export FP16
    logger.info("=== Step 4b: OpenVINO export (FP16) ===")
    fp16_dir = EXPORT_DIR / "fp16"
    fp16_dir.mkdir(parents=True, exist_ok=True)
    ov_path_fp16 = model.export(export_dir=fp16_dir, backend=Backend.OPENVINO, compress_to_fp16=True)
    logger.info("OpenVINO FP16 exported to: %s", ov_path_fp16)

    return ov_path, ov_path_fp16


def step_5_validate_openvino_cpu(ov_path, input_size):
    """Validate OpenVINO model on CPU."""
    logger.info("=== Step 5: OpenVINO CPU validation ===")
    import openvino

    core = openvino.Core()
    ov_model = core.read_model(str(ov_path))
    logger.info("Model inputs: %s", [(inp.get_any_name(), inp.shape) for inp in ov_model.inputs])
    logger.info("Model outputs: %s", [(out.get_any_name(), out.shape) for out in ov_model.outputs])

    compiled = core.compile_model(ov_model, "CPU")
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    start = time.time()
    result = compiled(dummy_input)
    elapsed = time.time() - start
    logger.info("CPU inference: %.3fs", elapsed)

    outputs = list(compiled.outputs)
    for i, port in enumerate(outputs):
        output = result[port]
        logger.info("  Output %d (%s): shape=%s dtype=%s", i, port.get_any_name(), output.shape, output.dtype)

    return result


def step_6_validate_openvino_gpu(ov_path, input_size):
    """Validate OpenVINO model on GPU (Intel)."""
    logger.info("=== Step 6: OpenVINO GPU validation ===")
    import openvino
    from openvino import properties

    core = openvino.Core()
    available = core.available_devices
    logger.info("Available OpenVINO devices: %s", available)

    if "GPU" not in available:
        logger.warning("No GPU device available. Skipping GPU validation.")
        return None

    ov_model = core.read_model(str(ov_path))

    # Try FP16 precision hint for GPU
    core.set_property("GPU", {properties.hint.inference_precision: openvino.Type.f16})

    logger.info("Compiling for GPU...")
    start = time.time()
    compiled = core.compile_model(ov_model, "GPU")
    logger.info("GPU compilation: %.2fs", time.time() - start)

    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)

    # Warmup
    for _ in range(3):
        compiled(dummy_input)

    # Benchmark
    times = []
    for _ in range(20):
        start = time.time()
        result = compiled(dummy_input)
        times.append(time.time() - start)

    logger.info("GPU inference: avg=%.3fs, min=%.3fs, max=%.3fs", np.mean(times), np.min(times), np.max(times))

    outputs = list(compiled.outputs)
    for i, port in enumerate(outputs):
        output = result[port]
        logger.info("  Output %d (%s): shape=%s dtype=%s", i, port.get_any_name(), output.shape, output.dtype)

    return result


def step_7_compare_with_real_image(ov_path, input_size, baseline):
    """Run OpenVINO inference on a real target image and compare with PyTorch baseline."""
    logger.info("=== Step 7: Real image comparison (CPU) ===")
    import openvino
    from torch.nn import functional

    from instantlearn.data.utils import read_image

    core = openvino.Core()
    ov_model = core.read_model(str(ov_path))
    compiled = core.compile_model(ov_model, "CPU")

    for i, target_path in enumerate(TARGET_IMAGES):
        img = read_image(str(target_path))  # [3, H, W]
        # Prepare input: the model expects raw uint8-range [1, 3, H, W] float tensor
        # EncoderForwardFeaturesWrapper does /255 + normalize internally
        img_np = img.numpy().astype(np.float32)
        img_resized = functional.interpolate(
            torch.from_numpy(img_np).unsqueeze(0),
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        ).numpy()

        result = compiled(img_resized)
        outputs = list(compiled.outputs)
        masks = result[outputs[0]]
        scores = result[outputs[1]]
        labels = result[outputs[2]] if len(outputs) > 2 else np.zeros_like(scores, dtype=np.int64)

        # Filter valid predictions
        valid = scores > 0.1
        n_valid = valid.sum()

        logger.info(
            "  Target %d (%s): %d valid masks (of %d total), max_score=%.3f",
            i,
            target_path.name,
            n_valid,
            masks.shape[0],
            scores.max() if len(scores) > 0 else 0.0,
        )

        # Compare with baseline
        if baseline and i < len(baseline["masks"]):
            baseline_n = baseline["masks"][i].shape[0]
            logger.info(
                "    Baseline had %d masks, max_score=%.3f",
                baseline_n,
                baseline["scores"][i].max() if len(baseline["scores"][i]) > 0 else 0.0,
            )


if __name__ == "__main__":
    logger.info("Starting Matcher OpenVINO export test")
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA device: %s", torch.cuda.get_device_name(0))

    # Step 1: PyTorch baseline prediction
    model, baseline = step_1_pytorch_predict()
    input_size = model.encoder.input_size
    logger.info("Encoder input size: %d", input_size)

    # Step 2: ONNX export
    onnx_path = step_2_export_onnx(model)

    # Step 3: ONNX Runtime validation
    step_3_validate_onnx(onnx_path, input_size)

    # Step 4: OpenVINO export (FP32 + FP16)
    ov_path, ov_path_fp16 = step_4_export_openvino(model)

    # Step 5: OpenVINO CPU validation
    step_5_validate_openvino_cpu(ov_path, input_size)

    # Step 6: OpenVINO GPU validation (if available)
    step_6_validate_openvino_gpu(ov_path, input_size)

    # Step 6b: OpenVINO GPU validation with FP16 model
    logger.info("=== Step 6b: OpenVINO GPU with FP16 model ===")
    step_6_validate_openvino_gpu(ov_path_fp16, input_size)

    # Step 7: Real image comparison
    step_7_compare_with_real_image(ov_path, input_size, baseline)

    logger.info("=== All steps completed ===")
