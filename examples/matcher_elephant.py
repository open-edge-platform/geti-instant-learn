import torch
from getiprompt.components.sam import SAMPredictor
from getiprompt.utils.constants import SAMModelName
from getiprompt.data.utils import read_image
from getiprompt.models import Matcher, GroundedSAM
from getiprompt.data import Sample
from getiprompt.visualizer import visualize_single_image


def matcher_example():
    """Example of using the Matcher model."""
    target_image = read_image("/home/yuchunli/datasets/coco/images/train2017/000000390341.jpg")
    ref_image = read_image("/home/yuchunli/datasets/coco/images/train2017/000000286874.jpg")

    # Initialize SAM predictor (auto-downloads weights)
    predictor = SAMPredictor(SAMModelName.SAM_HQ_TINY, device="cuda")

    # Set image and generate mask from a point click
    predictor.set_image(ref_image)
    ref_mask, _, _ = predictor.forward(
        point_coords=torch.tensor([[[280, 237]]], device="cuda"),
        point_labels=torch.tensor([[1]], device="cuda"),           # 1 = foreground
        multimask_output=False,
    )

    model = Matcher(device="cuda")

    # Create reference sample with the generated mask
    ref_sample = Sample(
        image=ref_image,
        masks=ref_mask[0],
    )

    # Fit on reference - no need for Batch.collate()
    model.fit(ref_sample)

    # Predict on target image - no need for Batch.collate()
    target_sample = Sample(image=target_image)
    predictions = model.predict(target_sample)

    # save visualization
    visualize_single_image(
        image=target_sample.image,
        prediction=predictions[0],
        file_name="matcher_elephant_output.png",
        output_folder=".",
        color_map={0: (0, 255, 0)},
    )

if __name__ == "__main__":
    matcher_example()