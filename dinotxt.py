import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from getiprompt.data.utils.image import read_image
from getiprompt.models.dinotxt import DinoTxtZeroShotClassification
from getiprompt.utils.constants import DINOv3BackboneSize
from getiprompt.data import Sample, Batch


def main():
    # Configuration parameters
    data_root = Path("/home/yuchunli/git/geti-prompt/library/tests/assets/fss-1000/images")
    precision = "bf16"  # Options: "bf16", "fp16", "fp32"

    # Import dataset
    label_names = []
    target_images = []
    gt_label_names = []
    for path in data_root.rglob("*/*.jpg"):
        label_name = path.parent.name
        if label_name not in label_names:
            label_names.append(label_name)
        gt_label_names.append(label_name)
        img = read_image(path, as_tensor=True)
        target_images.append(img)

    gt_labels = [label_names.index(gt_label_name) for gt_label_name in gt_label_names]


    # Initialize DinoTxt pipeline
    dinotxt = DinoTxtZeroShotClassification(
        precision=precision,
        backbone_size=DINOv3BackboneSize.LARGE,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Learn from text prompts (class names)
    start_time = timeit.default_timer()

    samples = [
        Sample(
            categories=label_names[label_id],
            category_ids=[label_id], 
            is_reference=[True]
        ) 
        for label_id in range(len(label_names))
    ]
    batch = Batch.collate(samples)

    dinotxt.learn(batch)

    learn_time = timeit.default_timer() - start_time
    print(f"Learning completed in {learn_time:.2f} seconds")


    # Perform inference on target images
    print("Starting inference...")
    inference_start_time = timeit.default_timer()

    target_samples = [
        Sample(
            image=image,
            categories=[gt_label_name],
            category_ids=[gt_label],
            is_reference=[False]
        )
        for image, gt_label, gt_label_name in zip(target_images, gt_labels, gt_label_names, strict=True)
    ]
    target_batch = Batch.collate(target_samples)
    predictions = dinotxt.infer(target_batch=target_batch)

    # Convert to tensors
    pred_labels = torch.stack([prediction["pred_labels"] for prediction in predictions]).cuda()
    gt_labels = torch.tensor(gt_labels).cuda()

    accuracy = sum(pred_labels == gt_labels) / len(gt_labels)
    print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    inference_time = timeit.default_timer() - inference_start_time
    print(f"Inference completed in {inference_time:.2f} seconds")


if __name__ == "__main__":
    main()