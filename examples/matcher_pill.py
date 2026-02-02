from getiprompt.models import Matcher
from getiprompt.components.sam import SAMPredictor
from getiprompt.utils.constants import SAMModelName
from getiprompt.data.utils import read_image
import torch
import numpy as np
from segment_anything_hq import sam_model_registry
from segment_anything_hq.predictor import SamPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


def generate_masks_(ref_image_path, ref_points):
    ref_image = read_image(ref_image_path)

    predictor = SAMPredictor(SAMModelName.SAM_HQ_TINY, device="cuda")

    # Set image and generate mask from a point click
    predictor.set_image(ref_image)
    ref_masks, _, _ = predictor.forward(
        point_coords=torch.tensor([ref_points], device="cuda"),  # Click on apple
        point_labels=torch.tensor([len(ref_points) * [1]], device="cuda"),
    )
    return ref_masks


def main():
    ref_image = "/home/yuchunli/datasets/pill/train/1_jpg.rf.25aa0bf88f95f5d17fd9cde41f2c768f.jpg"
    ref_points = [
        [238, 81],
        # [410, 81],
        # [238, 162],
        # [417, 162],
    ]
    ref_masks = generate_masks_(ref_image, ref_points)

    matcher = Matcher()


if __name__ == "__main__":
    main()