import re
from pathlib import Path

import torch
from PIL import Image

from getiprompt.data.base import Batch, Sample
from getiprompt.data.folder import FolderDataset
from getiprompt.data.utils.image import read_image
from getiprompt.models import SAM3

from getiprompt.visualizer import visualize_single_image


def text_prompt():
    """Example: Using text prompts to segment objects."""
    target_image = read_image("/home/yuchunli/Desktop/test_images/albania_linnell_dsc_0501-2.jpg")

    model = SAM3(device="cuda", confidence_threshold=0.5)

    sample = Sample(
        image=target_image,
        categories=["sheep", "headscarf", "woman", "bush", "tree", "cane"],
        category_ids=[0, 1, 2, 3, 4, 5],
    )

    predictions = model.predict(Batch.collate([sample]))

    category_colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 0, 255),
        3: (255, 165, 0),
        4: (128, 128, 0),
        5: (128, 0, 128),
    }

    visualize_single_image(
        target_image,
        predictions[0],
        "target_image.png",
        "output",
        category_colors,
    )


def text_box_prompt():
    """Example: Combining text and box prompts."""
    target_image = read_image("/home/yuchunli/Desktop/test_images/000000136466.jpg")
    model = SAM3(
        device="cuda",
        confidence_threshold=0.2,
        compile_models=False,
        precision="fp32",
    )

    pot_box = [75, 16, 160, 100]  # x1, y1, x2, y2
    dial_box = [134, 140, 158, 166]  # x1, y1, x2, y2
    bboxes = torch.stack([torch.tensor(pot_box), torch.tensor(dial_box)], dim=0)

    target_sample = Sample(
        image=target_image,
        categories=["pot", "dial"],
        category_ids=torch.tensor([0, 1]),
        bboxes=bboxes,
    )

    predictions = model.predict(Batch.collate([target_sample]))
    visualize_single_image(
        target_image,
        predictions[0],
        "target_image.png",
        "output",
        {0: [255, 0, 0], 1: [0, 0, 255]},
    )


def compare_prompting_modes():
    """Compare different SAM3 prompting modes on the same image."""
    print("=" * 60)
    print("Comparing SAM3 Prompting Modes")
    print("=" * 60)

    fss1000_root = Path(__file__).parent.parent / "library" / "tests" / "assets" / "fss-1000"
    output_dir = Path(__file__).parent.parent / "output"
    output_dir.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load dataset
    dataset = FolderDataset(
        root=fss1000_root,
        categories=["apple"],
        n_shots=1,
    )

    ref_sample = dataset.get_reference_dataset()[0]
    target_sample = dataset.get_target_dataset()[0]

    # Mode 1: Text prompting
    print("\n1. Text Prompting Mode:")
    model_text = SAM3(device=device, precision="fp32", confidence_threshold=0.3)
    target_with_text = Sample(
        image=target_sample.image,
        categories=["apple"],
        category_ids=[0],
    )
    pred_text = model_text.predict(Batch.collate([target_with_text]))[0]
    n_text = pred_text["pred_masks"].shape[0] if pred_text["pred_masks"].numel() > 0 else 0
    print(f"   Found {n_text} objects using text prompt 'apple'")

    visualize_single_image(
        target_sample.image,
        pred_text,
        "sam3_text_prompt.png",
        str(output_dir),
        {0: (255, 0, 0)},
    )

    # Mode 2: Exemplar Feature Injection
    print("\n2. Exemplar Feature Injection Mode:")
    model_exemplar = SAM3(device=device, precision="fp32", confidence_threshold=0.3)
    model_exemplar.fit(Batch.collate([ref_sample]))  # Extract features from reference
    target_no_prompt = Sample(image=target_sample.image)  # No text or boxes!
    pred_exemplar = model_exemplar.predict(Batch.collate([target_no_prompt]))[0]
    n_exemplar = pred_exemplar["pred_masks"].shape[0] if pred_exemplar["pred_masks"].numel() > 0 else 0
    print(f"   Found {n_exemplar} objects using visual exemplar (no text/boxes on target)")

    visualize_single_image(
        target_sample.image,
        pred_exemplar,
        "sam3_exemplar_injection.png",
        str(output_dir),
        {0: (0, 255, 0)},
    )

    print(f"\nResults saved to {output_dir}")
    print("=" * 60)


def text_prompt_coffee_bean():
    from getiprompt.models.foundation.vlm_fo1.model.builder import load_pretrained_model
    from getiprompt.models.foundation.vlm_fo1.mm_utils import (
        extract_predictions_to_indexes,
        prepare_inputs,
    )
    from getiprompt.models.foundation.vlm_fo1.task_templates import OD_Counting_template
    
    """Example: Using text prompts to segment objects with VLM-FO1 refinement.

    This demonstrates SAM3 + VLM-FO1 pipeline:
    1. SAM3 generates initial detections (boxes, masks, scores)
    2. VLM-FO1 acts as a reasoning layer to filter which boxes match the prompt
    3. Results are saved in FolderDataset-compatible format
    """
    image_paths = list(Path("/home/yuchunli/git/geti-prompt/coffee-berries").glob("*.png"))
    prompt_text = "berry"

    # Create output directory structure for FolderDataset compatibility
    output_root = Path("/home/yuchunli/git/geti-prompt/output/coffee_berry_dataset")
    images_dir = output_root / "images" / prompt_text
    masks_dir = output_root / "masks" / prompt_text
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_root}")
    print(f"  Images: {images_dir}")
    print(f"  Masks: {masks_dir}")

    # Initialize SAM3 model
    sam3_model = SAM3(device="cuda", precision="bf16", confidence_threshold=0.0)

    # Initialize VLM-FO1 model for box refinement
    vlm_fo1_model_path = "omlab/VLM-FO1_Qwen2.5-VL-3B-v01"
    tokenizer, vlm_fo1_model, image_processors = load_pretrained_model(
        model_path=vlm_fo1_model_path,
        device="cuda:0",
    )

    for image_path in image_paths:
        print(f"\nProcessing: {image_path.name}")
        target_image = read_image(image_path)

        sample = Sample(
            image=target_image,
            categories=[prompt_text],
            category_ids=[0],
        )

        # Step 1: Get SAM3 predictions
        predictions = sam3_model.predict(Batch.collate([sample]))
        pred = predictions[0]

        # ========================================
        # VLM-FO1 box refinement
        # ========================================
        pred_boxes = pred["pred_boxes"]  # [N, 5] with (x1, y1, x2, y2, score)
        pred_masks = pred["pred_masks"]
        pred_labels = pred["pred_labels"]

        if pred_boxes.numel() > 0:
            # Sort by confidence and limit to top 100
            scores_tensor = pred_boxes[:, 4]
            sorted_indices = torch.argsort(scores_tensor, descending=True)
            sorted_boxes = pred_boxes[sorted_indices][:100]
            sorted_masks = pred_masks[sorted_indices][:100]
            sorted_labels = pred_labels[sorted_indices][:100]
            sorted_scores = sorted_boxes[:, 4]

            # Adaptive confidence threshold based on detection quality
            if len(sorted_scores) > 0 and sorted_scores[0] > 0.75:
                conf_threshold = 0.3  # High-quality detections: stricter filter
            else:
                conf_threshold = 0.05  # Uncertain detections: keep more for VLM-FO1

            # Filter by confidence threshold
            conf_mask = sorted_scores > conf_threshold
            filtered_boxes = sorted_boxes[conf_mask]
            filtered_masks = sorted_masks[conf_mask]
            filtered_labels = sorted_labels[conf_mask]
            filtered_scores = sorted_scores[conf_mask]

            # Track high confidence boxes (score > 0.8) - these are very reliable
            high_conf_mask = filtered_scores > 0.8
            high_conf_boxes = filtered_boxes[high_conf_mask]
            print(
                f"  SAM3: {pred_boxes.shape[0]} raw -> {filtered_boxes.shape[0]} after threshold "
                f"(conf_thresh={conf_threshold:.2f}, high_conf={high_conf_boxes.shape[0]})"
            )

            if filtered_boxes.numel() > 0:
                # Extract boxes (x1, y1, x2, y2) for VLM-FO1
                bboxes = filtered_boxes[:, :4].tolist()
                scores = filtered_scores.tolist()

                # Load PIL image for VLM-FO1
                pil_image = Image.open(image_path).convert("RGB")

                # Prepare VLM-FO1 prompt
                fo1_prompt = OD_Counting_template.format(prompt_text)
                content = [
                    {"type": "image_url", "image_url": {"url": pil_image}},
                    {"type": "text", "text": fo1_prompt},
                ]
                messages = [{"role": "user", "content": content, "bbox_list": bboxes}]

                # Run VLM-FO1 inference
                generation_kwargs = prepare_inputs(
                    vlm_fo1_model_path,
                    vlm_fo1_model,
                    image_processors,
                    tokenizer,
                    messages,
                    max_tokens=4096,
                    top_p=0.05,
                    temperature=0.0,
                    do_sample=False,
                    image_size=1024,
                )

                with torch.inference_mode():
                    output_ids = vlm_fo1_model.generate(**generation_kwargs)
                    outputs = tokenizer.decode(
                        output_ids[0, generation_kwargs["inputs"].shape[1] :]
                    ).strip()
                    print(f"  VLM-FO1 output: {outputs}")

                # Parse VLM-FO1 output to get filtered box indices
                if "<ground>" in outputs:
                    prediction_dict = extract_predictions_to_indexes(outputs)
                else:
                    match_pattern = r"<region(\d+)>"
                    matches = re.findall(match_pattern, outputs)
                    prediction_dict = {f"<region{m}>": {int(m)} for m in matches}

                # Collect refined box indices (indices into filtered_boxes)
                refined_indices = []
                for indices in prediction_dict.values():
                    for box_index in indices:
                        if box_index < len(bboxes):
                            refined_indices.append(box_index)

                # Filter predictions using refined indices
                if refined_indices:
                    refined_masks = filtered_masks[refined_indices]
                    refined_boxes = filtered_boxes[refined_indices]
                    refined_labels = filtered_labels[refined_indices]

                    refined_pred = {
                        "pred_masks": refined_masks,
                        "pred_boxes": refined_boxes,
                        "pred_labels": refined_labels,
                    }
                    print(
                        f"  VLM-FO1 refined: {len(refined_indices)} boxes from {filtered_boxes.shape[0]} filtered detections"
                    )
                else:
                    # No VLM-FO1 matches, use filtered predictions
                    refined_pred = {
                        "pred_masks": filtered_masks,
                        "pred_boxes": filtered_boxes,
                        "pred_labels": filtered_labels,
                    }
                    print("  No VLM-FO1 matches, using confidence-filtered detections")
            else:
                refined_pred = pred
                print(f"  No detections above threshold for {image_path.name}")
        else:
            refined_pred = pred
            print(f"  No detections from SAM3 for {image_path.name}")

        # ========================================
        # Save to FolderDataset-compatible format
        # ========================================
        # Use image stem as filename (e.g., "image1" from "image1.png")
        output_name = image_path.stem

        # Save original image
        pil_image_save = Image.open(image_path).convert("RGB")
        output_image_path = images_dir / f"{output_name}.jpg"
        pil_image_save.save(output_image_path, "JPEG", quality=95)

        # Save combined mask (union of all predicted masks)
        pred_masks = refined_pred["pred_masks"]
        if pred_masks.numel() > 0 and pred_masks.shape[0] > 0:
            # Combine all masks into single binary mask (union)
            combined_mask = pred_masks.any(dim=0)  # [H, W] boolean tensor
            # Convert to uint8 (0 or 255) for saving as PNG
            mask_np = (combined_mask.cpu().numpy() * 255).astype("uint8")
            mask_image = Image.fromarray(mask_np, mode="L")
            output_mask_path = masks_dir / f"{output_name}.png"
            mask_image.save(output_mask_path)
            print(f"  Saved: {output_image_path.name} + {output_mask_path.name} ({pred_masks.shape[0]} instances combined)")
        else:
            # Save empty mask if no detections
            h, w = target_image.shape[-2:]
            empty_mask = Image.new("L", (w, h), 0)
            output_mask_path = masks_dir / f"{output_name}.png"
            empty_mask.save(output_mask_path)
            print(f"  Saved: {output_image_path.name} + {output_mask_path.name} (empty mask)")

        # Visualize refined results
        category_colors = {0: (255, 255, 0)}
        visualize_single_image(
            target_image,
            refined_pred,
            f"target_image_{image_path.stem}.jpg",
            "output",
            category_colors,
        )

    # Print summary
    print("\n" + "=" * 60)
    print("Done! Dataset saved in FolderDataset-compatible format:")
    print(f"  Root: {output_root}")
    print(f"  Structure: images/{prompt_text}/*.jpg, masks/{prompt_text}/*.png")
    print("\nTo load this dataset:")
    print(f'  dataset = FolderDataset(root="{output_root}", categories=["{prompt_text}"])')
    print("=" * 60)


if __name__ == "__main__":
    # Run the exemplar feature injection example (FSS-1000 style)
    # exemplar_feature_injection_fss1000()

    # Optional: Compare different prompting modes
    # compare_prompting_modes()

    # Other examples:
    text_prompt()
    # text_box_prompt()
    # visual_prompt_with_sam3_processor()

    # text_prompt_coffee_bean()
