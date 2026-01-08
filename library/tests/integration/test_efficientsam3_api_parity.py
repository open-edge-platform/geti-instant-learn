"""Test to verify EfficientSAM3 API parity with the efficientsam3 repo.

This test replicates the exact API calls from:
efficientsam3/sam3/efficientsam3_examples/efficientsam3_for_sam1_task_example.ipynb

to verify that geti-prompt's implementation works identically.
"""

import numpy as np
import pytest
import torch
from PIL import Image


@pytest.fixture
def device():
    """Get compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_image():
    """Load sample image (truck.jpg from efficientsam3 repo)."""
    img_path = "/home/devuser/workspace/code/Prompt/efficientsam3/sam3/assets/images/truck.jpg"
    return Image.open(img_path)


@pytest.fixture
def checkpoint_path():
    """Path to EfficientSAM3 checkpoint."""
    return "/home/devuser/workspace/code/Prompt/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth"


@pytest.fixture
def bpe_path():
    """Path to BPE tokenizer vocabulary."""
    return "/home/devuser/workspace/code/Prompt/efficientsam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


class TestEfficientSAM3APIParity:
    """Test API parity with efficientsam3 repo."""

    def test_build_model(self, device, checkpoint_path, bpe_path):
        """Test that model builds correctly with same parameters as efficientsam3."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )

        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        # Verify model has required attributes
        assert hasattr(model, "inst_interactive_predictor")
        assert hasattr(model, "_prepare_backbone_features")
        assert hasattr(model, "no_mem_embed")
        assert model.inst_interactive_predictor is not None

    def test_single_point_prompt(self, device, checkpoint_path, bpe_path, sample_image):
        """Test single point prompt - matches efficientsam3 notebook cell #VSC-fcc60ad9."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )
        from getiprompt.models.foundation.sam3.sam3_image_processor import (
            Sam3Processor,
        )

        # Build model (same as notebook)
        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        # Create processor and set image
        processor = Sam3Processor(model)
        inference_state = processor.set_image(sample_image)

        # Verify inference_state structure
        assert "original_height" in inference_state
        assert "original_width" in inference_state
        assert "backbone_out" in inference_state

        # Single point prompt (same as notebook)
        input_point = np.array([[800, 650]])
        input_label = np.array([1])

        # Predict with multimask_output=True (same as notebook)
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Verify output shapes match expected format
        assert isinstance(masks, np.ndarray)
        assert isinstance(scores, np.ndarray)
        assert isinstance(logits, np.ndarray)

        # With multimask_output=True, should return 3 masks
        assert masks.ndim == 3, f"Expected 3D masks array, got {masks.ndim}D"
        assert masks.shape[0] == 3, f"Expected 3 masks, got {masks.shape[0]}"
        assert scores.shape[0] == 3, f"Expected 3 scores, got {scores.shape[0]}"

        # Sort by score (same as notebook)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        assert scores[0] >= scores[1] >= scores[2], "Scores should be sorted descending"

    def test_multi_point_with_mask_input(
        self, device, checkpoint_path, bpe_path, sample_image
    ):
        """Test multi-point with mask input - matches notebook cell #VSC-1479aa68."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )
        from getiprompt.models.foundation.sam3.sam3_image_processor import (
            Sam3Processor,
        )

        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        processor = Sam3Processor(model)
        inference_state = processor.set_image(sample_image)

        # First prediction to get logits for mask_input
        input_point = np.array([[800, 650]])
        input_label = np.array([1])
        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        # Multi-point with mask input (same as notebook)
        input_point = np.array([[800, 650], [1500, 460]])
        input_label = np.array([1, 1])
        mask_input = logits[np.argmax(scores), :, :]  # Choose best mask's logits

        masks2, scores2, _ = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            mask_input=mask_input[None, :, :],  # Add batch dimension
            multimask_output=False,
        )

        # With multimask_output=False, should return 1 mask
        assert masks2.shape[0] == 1, f"Expected 1 mask, got {masks2.shape[0]}"
        assert scores2.shape[0] == 1, f"Expected 1 score, got {scores2.shape[0]}"

    def test_box_prompt(self, device, checkpoint_path, bpe_path, sample_image):
        """Test box prompt - matches notebook cell #VSC-134aab98."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )
        from getiprompt.models.foundation.sam3.sam3_image_processor import (
            Sam3Processor,
        )

        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        processor = Sam3Processor(model)
        inference_state = processor.set_image(sample_image)

        # Box prompt (same as notebook)
        input_box = np.array([425, 600, 700, 875])

        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # Add batch dimension
            multimask_output=False,
        )

        assert masks.shape[0] == 1, f"Expected 1 mask, got {masks.shape[0]}"

    def test_combined_point_and_box(
        self, device, checkpoint_path, bpe_path, sample_image
    ):
        """Test combined point and box - matches notebook cell #VSC-d14be139."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )
        from getiprompt.models.foundation.sam3.sam3_image_processor import (
            Sam3Processor,
        )

        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        processor = Sam3Processor(model)
        inference_state = processor.set_image(sample_image)

        # Combined prompts (same as notebook)
        input_box = np.array([425, 600, 700, 875])
        input_point = np.array([[575, 750]])
        input_label = np.array([0])  # Background point

        masks, scores, logits = model.predict_inst(
            inference_state,
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,  # Can pass without batch dimension
            multimask_output=False,
        )

        assert masks.shape[0] == 1, f"Expected 1 mask, got {masks.shape[0]}"

    def test_batched_box_prompts(
        self, device, checkpoint_path, bpe_path, sample_image
    ):
        """Test batched box prompts - matches notebook cell #VSC-8a99beec."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )
        from getiprompt.models.foundation.sam3.sam3_image_processor import (
            Sam3Processor,
        )

        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        processor = Sam3Processor(model)
        inference_state = processor.set_image(sample_image)

        # Multiple boxes (same as notebook)
        input_boxes = np.array([
            [75, 275, 1725, 850],
            [425, 600, 700, 875],
            [1375, 550, 1650, 800],
            [1240, 675, 1400, 750],
        ])

        masks, scores, _ = model.predict_inst(
            inference_state,
            point_coords=None,
            point_labels=None,
            box=input_boxes,  # Pass multiple boxes directly
            multimask_output=False,
        )

        # Should return 4 masks, one per box
        assert masks.shape[0] == 4, f"Expected 4 masks, got {masks.shape[0]}"
        assert scores.shape[0] == 4, f"Expected 4 scores, got {scores.shape[0]}"

    def test_batch_inference(self, device, checkpoint_path, bpe_path, sample_image):
        """Test batch inference with predict_inst_batch - matches notebook cells #VSC-e3417781, #VSC-08af7d50."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )
        from getiprompt.models.foundation.sam3.sam3_image_processor import (
            Sam3Processor,
        )

        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        processor = Sam3Processor(model)

        # Two images (same as notebook)
        image1 = sample_image  # truck.jpg
        image2 = Image.open(
            "/home/devuser/workspace/code/Prompt/efficientsam3/sam3/assets/images/groceries.jpg"
        )

        image1_boxes = np.array([
            [75, 275, 1725, 850],
            [425, 600, 700, 875],
            [1375, 550, 1650, 800],
            [1240, 675, 1400, 750],
        ])

        image2_boxes = np.array([
            [450, 170, 520, 350],
            [350, 190, 450, 350],
            [500, 170, 580, 350],
            [580, 170, 640, 350],
        ])

        img_batch = [image1, image2]
        boxes_batch = [image1_boxes, image2_boxes]

        # Set image batch
        inference_state = processor.set_image_batch(img_batch)

        # Verify batch inference_state
        assert "original_heights" in inference_state
        assert "original_widths" in inference_state
        assert len(inference_state["original_heights"]) == 2

        # Batch prediction
        masks_batch, scores_batch, _ = model.predict_inst_batch(
            inference_state,
            None,
            None,
            box_batch=boxes_batch,
            multimask_output=False,
        )

        # Should return lists
        assert isinstance(masks_batch, list)
        assert isinstance(scores_batch, list)
        assert len(masks_batch) == 2
        assert len(scores_batch) == 2

        # First image has 4 boxes, so 4 masks
        assert masks_batch[0].shape[0] == 4
        # Second image has 4 boxes, so 4 masks
        assert masks_batch[1].shape[0] == 4

    def test_batch_point_inference(
        self, device, checkpoint_path, bpe_path, sample_image
    ):
        """Test batch point inference - matches notebook cells #VSC-e83bb90e, #VSC-3d9c65c4."""
        from getiprompt.models.foundation.efficientsam3.model_builder import (
            build_efficientsam3_image_model,
        )
        from getiprompt.models.foundation.sam3.sam3_image_processor import (
            Sam3Processor,
        )

        model = build_efficientsam3_image_model(
            bpe_path=bpe_path,
            enable_inst_interactivity=True,
            checkpoint_path=checkpoint_path,
            backbone_type="tinyvit-21m",
            text_encoder_type="MobileCLIP-S1",
            device=str(device),
        )

        processor = Sam3Processor(model)

        # Two images
        image1 = sample_image
        image2 = Image.open(
            "/home/devuser/workspace/code/Prompt/efficientsam3/sam3/assets/images/groceries.jpg"
        )

        img_batch = [image1, image2]

        # Point prompts (same format as notebook)
        # Bx1x2 where B corresponds to number of objects
        image1_pts = np.array([[[500, 375]], [[650, 750]]])
        image1_labels = np.array([[1], [1]])

        image2_pts = np.array([[[400, 300]], [[600, 300]]])
        image2_labels = np.array([[1], [1]])

        pts_batch = [image1_pts, image2_pts]
        labels_batch = [image1_labels, image2_labels]

        # Set image batch
        inference_state = processor.set_image_batch(img_batch)

        # Batch prediction with points
        masks_batch, scores_batch, _ = model.predict_inst_batch(
            inference_state,
            pts_batch,
            labels_batch,
            box_batch=None,
            multimask_output=True,
        )

        assert len(masks_batch) == 2
        assert len(scores_batch) == 2

        # Select best masks (same as notebook)
        best_masks = []
        for masks, scores in zip(masks_batch, scores_batch):
            best_masks.append(masks[range(len(masks)), np.argmax(scores, axis=-1)])

        assert len(best_masks) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
