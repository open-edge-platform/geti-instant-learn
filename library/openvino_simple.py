from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import openvino
import torch
import torch.nn.functional as F

from instantlearn.data import Sample
from instantlearn.models import Matcher
from instantlearn.utils.constants import SAMModelName

# reference image
root_dir = Path("examples/assets/coco")
ref_sample = Sample(
    image_path=str(root_dir / "000000286874.jpg"),
    mask_paths=str(root_dir / "000000286874_mask.png"),
)

# fit the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Matcher(device=device, sam=SAMModelName.SAM_HQ)
model.fit(ref_sample)

# Target sample
target_sample = Sample(image_path=str(root_dir / "000000173279.jpg"))
# predict


tic = time()
predictions = model.predict(target_sample)
pt_masks = predictions[0]["pred_masks"].cpu().numpy()
pt_time = time() - tic
pt_scores = predictions[0]["pred_scores"].cpu().numpy().round(3)
pt_labels = predictions[0]["pred_labels"].cpu().numpy()

# Export to OpenVINO

ov_path = model.export(backend="openvino", compress_to_fp16=True)
print(f"Model exported to {ov_path}")

# Free PyTorch model from memory before loading OpenVINO
del model
if device == "cuda":
    torch.cuda.empty_cache()

core = openvino.Core()
ov_model = core.read_model(str(ov_path))
compiled_model = core.compile_model(ov_model, "GPU")

# PASS THE IMAGE
input_data = target_sample.image.numpy()

# input_data.shape is (3, 426, 640)
# expected is [1,3,512,512]

expected_shape = compiled_model.input(0).shape
if input_data.shape != tuple(expected_shape[1:]):
    tensor = torch.from_numpy(input_data)
    tensor = F.interpolate(tensor[None], size=(expected_shape[2], expected_shape[3]), mode="bilinear")
    input_data = tensor.numpy()

tic = time()
outputs = compiled_model(input_data)
ov_gpu_time = time() - tic
ov_gpu_masks, ov_gpu_scores, ov_gpu_labels = outputs.values()

# Free GPU compiled model before loading CPU model
del compiled_model

# do the same but on CPU
compiled_model = core.compile_model(ov_model, "CPU")
tic = time()
outputs = compiled_model(input_data)
ov_cpu_time = time() - tic
ov_cpu_masks, ov_cpu_scores, ov_cpu_labels = outputs.values()

# Print all inference results together
print(f"PyTorch ({device}) Inference: {pt_masks.shape[0]} masks, scores={pt_scores}, labels={pt_labels}")
print(f"PyTorch ({device}) Inference time: {pt_time:.2f} seconds")
print(f"OpenVINO GPU Inference: {ov_gpu_masks.shape[0]} masks, scores={ov_gpu_scores.round(3)}, labels={ov_gpu_labels}")
print(f"OpenVINO GPU Inference time: {ov_gpu_time:.2f} seconds")
print(f"OpenVINO CPU Inference: {ov_cpu_masks.shape[0]} masks, scores={ov_cpu_scores.round(3)}, labels={ov_cpu_labels}")
print(f"OpenVINO CPU Inference time: {ov_cpu_time:.2f} seconds")


def _combine_masks(masks: np.ndarray) -> np.ndarray:
    """Return a single 2D mask by taking the union of all predicted masks."""
    return masks.any(axis=0).astype(np.float32)


fig, axes = plt.subplots(1, 3, figsize=(12, 4))
titles = ["PyTorch", "OpenVINO GPU", "OpenVINO CPU"]
all_masks = [pt_masks, ov_gpu_masks, ov_cpu_masks]

for ax, title, masks in zip(axes, titles, all_masks, strict=False):
    ax.imshow(_combine_masks(masks), cmap="gray")
    ax.set_title(title)
    ax.axis("off")

fig.tight_layout()
out_path = root_dir / "masks_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Mask comparison saved to {out_path}")
