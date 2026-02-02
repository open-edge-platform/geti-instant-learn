from transformers import Sam3Processor, Sam3Model, Sam3TrackerModel, Sam3TrackerProcessor, Sam3VideoModel, Sam3VideoProcessor
import torch
from PIL import Image
import matplotlib
import numpy as np
import requests
from transformers.video_utils import load_video
from accelerate import Accelerator
import cv2
import os
import glob

def overlay_masks(image, masks, categories=None):
    """
    Overlay masks on an image with colored transparency.
    
    Args:
        image: PIL Image or numpy array
        masks: Tensor of shape (n_masks, H, W)
        categories: Optional list of category names for each mask.
                   If provided, masks with the same category get the same color.
                   Can be a flat list (one category per mask) or used with 
                   multi-category results from text_prompt().
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.convert("RGBA")
    masks = 255 * masks.cpu().numpy().astype(np.uint8)
    
    n_masks = masks.shape[0]
    
    if categories is not None:
        # Assign colors by category (same category = same color)
        unique_categories = list(dict.fromkeys(categories))  # preserve order
        n_categories = len(unique_categories)
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_categories)
        category_colors = {
            cat: tuple(int(c * 255) for c in cmap(i)[:3])
            for i, cat in enumerate(unique_categories)
        }
        colors = [category_colors[cat] for cat in categories]
    else:
        # Default: each mask gets a unique color
        cmap = matplotlib.colormaps.get_cmap("rainbow").resampled(n_masks)
        colors = [
            tuple(int(c * 255) for c in cmap(i)[:3])
            for i in range(n_masks)
        ]

    for mask, color in zip(masks, colors):
        mask = Image.fromarray(mask)
        overlay = Image.new("RGBA", image.size, color + (0,))
        alpha = mask.point(lambda v: int(v * 0.5))
        overlay.putalpha(alpha)
        image = Image.alpha_composite(image, overlay)
    return image


def text_prompt():
    """Example: Using text prompts to segment objects with cached vision features."""
    image_path = "/home/yuchunli/Desktop/test_images/albania_linnell_dsc_0501-2.jpg"
    image = Image.open(image_path).convert("RGB")

    learn_categories = ["sheep", "headscarf", "woman", "bush", "tree", "cane"]
    device = "cuda"
    model = Sam3Model.from_pretrained("facebook/sam3", attn_implementation="sdpa").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    # Process image once
    img_inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Cache vision features (expensive - only done once)
    with torch.no_grad():
        vision_embeds = model.get_vision_features(img_inputs.pixel_values)

    # Loop over categories (cheap - only text encoding + decoder)
    all_masks = []
    all_categories = []
    for category in learn_categories:
        text_inputs = processor(text=category, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model(
                vision_embeds=vision_embeds,  # Reuse cached vision features
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
            )
        
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=0.5,
            mask_threshold=0.5,
            target_sizes=img_inputs.get("original_sizes").tolist()
        )[0]
        
        # Collect masks and their category labels
        n_masks = results["masks"].shape[0]
        if n_masks > 0:
            all_masks.append(results["masks"])
            all_categories.extend([category] * n_masks)
    
    print(f"Segmented categories: {learn_categories}")
    print(f"Total masks: {len(all_categories)}")
    
    # Combine all masks and visualize with category-based colors
    if all_masks:
        combined_masks = torch.cat(all_masks, dim=0)
        overlayed_image = overlay_masks(image, combined_masks, categories=all_categories)
        overlayed_image.show()
        overlayed_image.convert("RGB").save("output/hf_output.jpg")

    return list(zip(all_categories, all_masks))


if __name__ == "__main__":
    text_prompt()
