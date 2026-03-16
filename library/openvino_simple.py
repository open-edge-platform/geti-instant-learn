from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
import openvino

from instantlearn.data import Sample
from instantlearn.models import Matcher

# reference image
root_dir = "/home/rgangire/workspace/code/GetiPrompt/geti-prompt/library/examples/assets/coco/"
ref_sample = Sample(
    image_path=str(Path(root_dir) / "000000286874.jpg"),
    mask_paths=str(Path(root_dir) / "000000286874_mask.png"),
)

# fit the model
device = "cuda"
model = Matcher(device=device)
model.fit(ref_sample)

# Target sample
target_sample = Sample(image_path=str(Path(root_dir) / "000000173279.jpg"))
# predict


tic = time()
predictions = model.predict(target_sample)
pt_masks = predictions[0]["pred_masks"].cpu().numpy()
print(
    f"PyTorch Inference: {pt_masks.shape[0]} masks, scores={predictions[0]['pred_scores'].cpu().numpy().round(3)}, labels={predictions[0]['pred_labels'].cpu().numpy()}",
)
print(f"PyTorch Inference time: {time() - tic:.2f} seconds")

# Export to OpenVINO

ov_path = model.export(backend="openvino", compress_to_fp16=True)
print(f"Model exported to {ov_path}")


core = openvino.Core()
ov_model = core.read_model(str(ov_path))
compiled_model = core.compile_model(ov_model, "GPU")

# PASS THE IMAGE
input_data = target_sample.image.numpy()

# input_data.shape is (3, 426, 640)
# expected is [1,3,512,512]

import torch
import torch.nn.functional as F

expected_shape = compiled_model.input(0).shape
if input_data.shape != tuple(expected_shape[1:]):
    tensor = torch.from_numpy(input_data)
    tensor = F.interpolate(tensor[None], size=(expected_shape[2], expected_shape[3]), mode="bilinear")
    input_data = tensor.numpy()

tic = time()
outputs = compiled_model(input_data)
print(f"OpenVINO GPU Inference time: {time() - tic:.2f} seconds")
ov_gpu_masks, scores, labels = outputs.values()
print(f"OpenVINO GPU Inference: {ov_gpu_masks.shape[0]} masks, scores={scores.round(3)}, labels={labels}")


# do the same but on CPU
compiled_model = core.compile_model(ov_model, "CPU")
tic = time()
outputs = compiled_model(input_data)
print(f"OpenVINO CPU Inference time: {time() - tic:.2f} seconds")
ov_cpu_masks, scores, labels = outputs.values()
print(f"OpenVINO CPU Inference: {ov_cpu_masks.shape[0]} masks, scores={scores.round(3)}, labels={labels}")


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
out_path = Path(root_dir) / "masks_comparison.png"
fig.savefig(out_path, dpi=150)
print(f"Mask comparison saved to {out_path}")
