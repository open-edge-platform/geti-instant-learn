import torch
from getiprompt.components.sam import SAMPredictor
from getiprompt.utils.constants import SAMModelName
from getiprompt.data.utils import read_image
from getiprompt.models import Matcher, GroundedSAM
from getiprompt.data import Sample


def matcher_example():
    """Example of using the Matcher model with an apple image."""
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

    model = Matcher(device="cuda")

    # Create reference sample with the generated mask
    ref_sample = Sample(
        image=ref_image,
        masks=ref_mask[0],
        categories=["apple"],
        category_ids=[1]
    )

    # Fit on reference - no need for Batch.collate()
    model.fit(ref_sample)

    # Predict on target image - no need for Batch.collate()
    target_image = read_image("library/examples/assets/fss-1000/images/apple/2.jpg")
    target_sample = Sample(image=target_image)
    predictions = model.predict(target_sample)

def groundedSAM_example():
    """Example of using the GroundedSAM model with an apple image."""
    model = GroundedSAM(device="cuda")

    # Create reference sample with category labels (no masks needed)
    ref_sample = Sample(
        categories=["apple"],
        category_ids=[0],
    )

    # Fit on reference - no need for Batch.collate()
    model.fit(ref_sample)

    # Predict on target image - no need for Batch.collate()
    target_image = read_image("library/examples/assets/fss-1000/images/apple/2.jpg")
    target_sample = Sample(image=target_image)
    predictions = model.predict(target_sample)

    # Access results
    masks = predictions[0]["pred_masks"]   # Predicted segmentation masks
    boxes = predictions[0]["pred_boxes"]   # Detected bounding boxes
    labels = predictions[0]["pred_labels"] # Category labels


if __name__ == "__main__":
    matcher_example()
    # groundedSAM_example()