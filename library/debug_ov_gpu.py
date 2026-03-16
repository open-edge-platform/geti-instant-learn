"""Diagnostic: compare PyTorch vs ONNX vs OpenVINO (CPU/GPU) to isolate GPU noise."""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from instantlearn.data import Sample
from instantlearn.models import Matcher

root_dir = Path("examples/assets/coco/")
ref_sample = Sample(
    image_path=str(root_dir / "000000286874.jpg"),
    mask_paths=str(root_dir / "000000286874_mask.png"),
)
target_sample = Sample(image_path=str(root_dir / "000000173279.jpg"))

# --- PyTorch baseline ---
model = Matcher(device="cuda")
model.fit(ref_sample)
predictions = model.predict(target_sample)
pt_masks = predictions[0]["pred_masks"].cpu().numpy()
pt_scores = predictions[0]["pred_scores"].cpu().numpy()
print(f"[PyTorch]  masks={pt_masks.shape}, scores={pt_scores.round(3)}")

# --- Export ONNX ---
onnx_path = model.export(backend="onnx")
print(f"ONNX exported to {onnx_path}")

# --- Export OpenVINO FP32 (no compression) ---
ov_fp32_path = model.export(backend="openvino", compress_to_fp16=False)
print(f"OV FP32 exported to {ov_fp32_path}")

# --- Export OpenVINO FP16 ---
ov_fp16_dir = Path("exports/matcher_fp16")
ov_fp16_dir.mkdir(parents=True, exist_ok=True)
ov_fp16_path = model.export(
    export_dir=ov_fp16_dir,
    backend="openvino",
    compress_to_fp16=True,
)
print(f"OV FP16 exported to {ov_fp16_path}")

# --- Prepare input ---
import openvino

core = openvino.Core()

# Read FP32 model to get expected shape
ov_model_fp32 = core.read_model(str(ov_fp32_path))
expected_shape = ov_model_fp32.input(0).shape
img = target_sample.image.numpy().astype(np.float32)
tensor = torch.from_numpy(img)
tensor = F.interpolate(
    tensor[None],
    size=(expected_shape[2], expected_shape[3]),
    mode="bilinear",
)
input_data = tensor.numpy()
print(f"\nInput shape: {input_data.shape}, dtype: {input_data.dtype}")

# --- ONNX Runtime inference ---
import onnxruntime as ort

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
inp_name = sess.get_inputs()[0].name
onnx_out = sess.run(None, {inp_name: input_data})
onnx_masks, onnx_scores, onnx_labels = onnx_out
print(f"\n[ONNX CPU] masks={onnx_masks.shape}, scores={onnx_scores.round(3)}, labels={onnx_labels}")
print(f"  mask stats: min={onnx_masks.min():.3f} max={onnx_masks.max():.3f} sum={onnx_masks.sum():.0f}")

# --- OpenVINO FP32 CPU ---
compiled_cpu_fp32 = core.compile_model(ov_model_fp32, "CPU")
out = compiled_cpu_fp32(input_data)
cpu32_masks, cpu32_scores, cpu32_labels = out.values()
print(f"\n[OV CPU FP32] masks={cpu32_masks.shape}, scores={cpu32_scores.round(3)}, labels={cpu32_labels}")
print(f"  mask stats: min={cpu32_masks.min():.3f} max={cpu32_masks.max():.3f} sum={cpu32_masks.sum():.0f}")

# --- OpenVINO FP32 GPU ---
compiled_gpu_fp32 = core.compile_model(ov_model_fp32, "GPU")
out = compiled_gpu_fp32(input_data)
gpu32_masks, gpu32_scores, gpu32_labels = out.values()
print(f"\n[OV GPU FP32] masks={gpu32_masks.shape}, scores={gpu32_scores.round(3)}, labels={gpu32_labels}")
print(f"  mask stats: min={gpu32_masks.min():.3f} max={gpu32_masks.max():.3f} sum={gpu32_masks.sum():.0f}")

# --- OpenVINO FP16 CPU ---
ov_model_fp16 = core.read_model(str(ov_fp16_path))
compiled_cpu_fp16 = core.compile_model(ov_model_fp16, "CPU")
out = compiled_cpu_fp16(input_data)
cpu16_masks, cpu16_scores, cpu16_labels = out.values()
print(f"\n[OV CPU FP16] masks={cpu16_masks.shape}, scores={cpu16_scores.round(3)}, labels={cpu16_labels}")
print(f"  mask stats: min={cpu16_masks.min():.3f} max={cpu16_masks.max():.3f} sum={cpu16_masks.sum():.0f}")

# --- OpenVINO FP16 GPU ---
compiled_gpu_fp16 = core.compile_model(ov_model_fp16, "GPU")
out = compiled_gpu_fp16(input_data)
gpu16_masks, gpu16_scores, gpu16_labels = out.values()
print(f"\n[OV GPU FP16] masks={gpu16_masks.shape}, scores={gpu16_scores.round(3)}, labels={gpu16_labels}")
print(f"  mask stats: min={gpu16_masks.min():.3f} max={gpu16_masks.max():.3f} sum={gpu16_masks.sum():.0f}")

# --- OpenVINO FP32 GPU with FP32 execution hint ---
compiled_gpu_fp32_hint = core.compile_model(
    ov_model_fp32,
    "GPU",
    config={"INFERENCE_PRECISION_HINT": "f32"},
)
out = compiled_gpu_fp32_hint(input_data)
gpu32h_masks, gpu32h_scores, gpu32h_labels = out.values()
print(f"\n[OV GPU FP32+f32_hint] masks={gpu32h_masks.shape}, scores={gpu32h_scores.round(3)}, labels={gpu32h_labels}")
print(f"  mask stats: min={gpu32h_masks.min():.3f} max={gpu32h_masks.max():.3f} sum={gpu32h_masks.sum():.0f}")

# --- Summary comparison ---
print("\n" + "=" * 60)
print("SUMMARY: mask pixel-sum (higher = more mask coverage)")
print("=" * 60)
all_results = [
    ("PyTorch", pt_masks),
    ("ONNX CPU", onnx_masks),
    ("OV CPU FP32", cpu32_masks),
    ("OV GPU FP32", gpu32_masks),
    ("OV CPU FP16", cpu16_masks),
    ("OV GPU FP16", gpu16_masks),
    ("OV GPU FP32+f32_hint", gpu32h_masks),
]
for name, m in all_results:
    binary = (m > 0.5).astype(np.float32) if m.dtype == np.float32 else m.astype(np.float32)
    print(f"  {name:25s}: pixel_sum={binary.sum():>10.0f}  unique_vals={len(np.unique(m)):>5d}")
