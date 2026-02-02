import torch
from getiprompt.components.sam import SAMPredictor
from getiprompt.utils.constants import SAMModelName
from getiprompt.data.utils import read_image
from getiprompt.data import Sample, Batch
from getiprompt.models import PerDino
from getiprompt.visualizer import visualize_single_image


def perdino_example():
    ref_image = read_image("/home/yuchunli/datasets/coffee_berry_dataset/images/berry/scene00001.jpg")

    # Initialize SAM predictor (auto-downloads weights)
    predictor = SAMPredictor(SAMModelName.SAM_HQ_TINY, device="cuda")

    # Set image and generate mask from a point click
    predictor.set_image(ref_image)
    ref_mask, _, _ = predictor.forward(
        point_coords=torch.tensor([[[845, 481]]], device="cuda"),
        point_labels=torch.tensor([[1]], device="cuda"),
        multimask_output=False,
    )

    model = PerDino(device="cuda", confidence_threshold=0.1, num_foreground_points=80, use_nms=False)

    # Create reference sample with the generated mask
    ref_sample = Sample(
        image=ref_image,
        masks=ref_mask[0],
        categories=["berry"],
        category_ids=[0]
    )

    # Fit on reference - no need for Batch.collate()
    model.fit(Batch.collate([ref_sample]))

    # Predict on target image - no need for Batch.collate()
    target_image = read_image("/home/yuchunli/datasets/coffee_berry_dataset/images/berry/scene00001.jpg")
    target_sample = Sample(image=target_image)
    predictions = model.predict(Batch.collate([target_sample]))

    # save visualization
    visualize_single_image(
        image=target_sample.image,
        prediction=predictions[0],
        file_name="perdino_coffee_berry_output.png",
        output_folder=".",
        color_map={0: (0, 255, 0)},
    )

if __name__ == "__main__":
    perdino_example()