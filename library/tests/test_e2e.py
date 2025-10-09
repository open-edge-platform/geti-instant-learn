import argparse

import pytest

from getiprompt.datasets import Dataset
from getiprompt.models.factory import load_model
from getiprompt.types.image import Image
from getiprompt.types.masks import Masks
from getiprompt.types.priors import Priors
from getiprompt.types.text import Text
from getiprompt.utils.constants import ModelName, SAMModelName
from getiprompt.utils.data import load_dataset


def get_priors_for_category(
    dataset: Dataset,
    category_name: str,
    n_shot: int,
) -> tuple[list[Image], list[Priors]]:
    """Get reference images and priors for a specific category."""
    # Get reference images and masks for the category
    ref_images_np = dataset.get_images_by_category(category_name, start=0, end=n_shot)
    ref_masks_np = dataset.get_masks_by_category(category_name, start=0, end=n_shot)

    # Convert to required types
    ref_images = [Image(img) for img in ref_images_np]
    ref_priors = []

    for mask_np in ref_masks_np:
        # Create mask object
        mask_obj = Masks()
        mask_np_3d = mask_np[:, :, None] if len(mask_np.shape) == 2 else mask_np
        mask_obj.add(mask_np_3d)

        # Create text prior
        text_prior = Text()
        text_prior.add(category_name, class_id=0)

        # Create priors object
        prior = Priors(masks=[mask_obj], text=text_prior)
        ref_priors.append(prior)

    return ref_images, ref_priors


@pytest.fixture
def arg_fixture(
    category_name: str = "cupcake", n_shot: int = 1, num_batches: int = 4, device: str = "cuda"
) -> argparse.Namespace:
    args = argparse.Namespace()

    # Required pipeline parameters
    args.grounding_model = "fushh7/llmdet_swin_tiny_hf"
    args.box_threshold = 0.4
    args.text_threshold = 0.3
    args.num_foreground_points = 40
    args.num_background_points = 2
    args.num_grid_cells = 16
    args.similarity_threshold = 0.65
    args.mask_similarity_threshold = 0.38
    args.precision = "bf16"
    args.compile_models = False
    args.benchmark_inference_speed = False
    args.image_size = None
    args.device = device
    args.n_shot = n_shot
    args.num_batches = num_batches
    args.class_name = category_name

    # Additional args for SoftMatcher (in case it's used)
    args.use_sampling = False
    args.use_spatial_sampling = False
    args.approximate_matching = False
    args.softmatching_score_threshold = 0.5
    args.softmatching_bidirectional = True
    return args


@pytest.fixture
def lvis_dataset_fixture(arg_fixture: argparse.Namespace) -> Dataset:
    return load_dataset("lvis", target_classes=arg_fixture.class_name, batch_size=5)


@pytest.mark.parametrize("sam_model_fixture", SAMModelName)
@pytest.mark.parametrize("pipeline_name_fixture", ModelName)
def test_e2e(
    arg_fixture: argparse.Namespace,
    lvis_dataset_fixture: Dataset,
    sam_model_fixture: SAMModelName,
    pipeline_name_fixture: ModelName,
):
    pipeline = load_model(sam=sam_model_fixture, model_name=pipeline_name_fixture, args=arg_fixture)

    ref_images, ref_priors = get_priors_for_category(
        lvis_dataset_fixture,
        arg_fixture.class_name,
        arg_fixture.n_shot,
    )

    pipeline.learn(reference_images=ref_images, reference_priors=ref_priors)
    pipeline.infer(target_images=ref_images)
