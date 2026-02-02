"""Generate a composite banner image for the README using elephant example.

Creates a 4-panel image showing:
1. Reference image (elephant)
2. Reference mask overlay
3. Target image (elephants)
4. Target image with prediction mask overlay

Uses PIL for modern text rendering with better fonts.
"""

import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Get a modern font for text rendering.

    Args:
        size: Font size in pixels.

    Returns:
        PIL font object.
    """
    # Try common modern fonts available on Linux
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)

    # Fallback to default font
    logger.warning("No TrueType font found, using default font")
    return ImageFont.load_default()


def add_text_label(
    image: np.ndarray,
    text: str,
    position: str = "top-right",
    font_size: int = 14,
) -> np.ndarray:
    """Add a text label to an image using PIL for better typography.

    Args:
        image: Input image (BGR format from cv2).
        text: Text to display.
        position: Label position - "top-right", "top-left", "bottom-right", "bottom-left".
        font_size: Font size in pixels.

    Returns:
        Image with text label (BGR format).
    """
    # Convert BGR to RGB for PIL
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)

    font = get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    h, w = image.shape[:2]
    padding = 6
    margin = 8

    # Calculate position based on corner
    if "right" in position:
        x = w - text_w - padding - margin
    else:
        x = margin

    if "top" in position:
        y = margin
    else:
        y = h - text_h - padding * 2 - margin

    # Draw semi-transparent background rectangle
    bg_coords = (x - padding, y - padding, x + text_w + padding, y + text_h + padding)

    # Create overlay for semi-transparent background
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rounded_rectangle(bg_coords, radius=4, fill=(0, 0, 0, 180))

    # Composite the overlay
    pil_image = pil_image.convert("RGBA")
    pil_image = Image.alpha_composite(pil_image, overlay)

    # Draw text
    draw = ImageDraw.Draw(pil_image)
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # Convert back to BGR
    result = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    return result


def create_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a mask on an image with color.

    Args:
        image: Input image (BGR format).
        mask: Binary mask (single channel, values 0-255 or 0-1).
        color: BGR color for the mask overlay.
        alpha: Transparency of the overlay (0-1).

    Returns:
        Image with mask overlay.
    """
    result = image.copy()

    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :] = color

    # Normalize mask to boolean
    if mask.max() > 1:
        mask_bool = mask > 127
    else:
        mask_bool = mask > 0.5

    # Apply overlay only where mask is True
    result[mask_bool] = cv2.addWeighted(
        image, 1 - alpha, colored_mask, alpha, 0
    )[mask_bool]

    return result


def resize_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image to target height while maintaining aspect ratio.

    Args:
        image: Input image.
        target_height: Target height in pixels.

    Returns:
        Resized image.
    """
    h, w = image.shape[:2]
    scale = target_height / h
    new_w = int(w * scale)
    return cv2.resize(image, (new_w, target_height), interpolation=cv2.INTER_AREA)


def generate_elephant_masks() -> tuple[np.ndarray, np.ndarray]:
    """Generate masks for elephant images using the Matcher model.

    Returns:
        Tuple of (reference_mask, target_mask) as numpy arrays.
    """
    from getiprompt.components.sam import SAMPredictor
    from getiprompt.data import Sample
    from getiprompt.data.utils import read_image
    from getiprompt.models import Matcher
    from getiprompt.utils.constants import SAMModelName

    assets_path = Path(__file__).parent.parent / "assets"
    ref_image_path = assets_path / "000000286874.jpg"
    target_image_path = assets_path / "000000390341.jpg"

    logger.info("Loading images...")
    ref_image = read_image(str(ref_image_path))
    target_image = read_image(str(target_image_path))

    logger.info("Initializing SAM predictor...")
    predictor = SAMPredictor(SAMModelName.SAM_HQ_TINY, device="cuda")

    # Generate reference mask from point click on elephant
    logger.info("Generating reference mask...")
    predictor.set_image(ref_image)
    ref_mask, _, _ = predictor.forward(
        point_coords=torch.tensor([[[280, 237]]], device="cuda"),
        point_labels=torch.tensor([[1]], device="cuda"),
        multimask_output=False,
    )

    logger.info("Running Matcher model...")
    model = Matcher(device="cuda")

    ref_sample = Sample(image=ref_image, masks=ref_mask[0])
    model.fit(ref_sample)

    target_sample = Sample(image=target_image)
    predictions = model.predict(target_sample)

    # Convert masks to numpy (ensure float32 type for cv2 compatibility)
    ref_mask_np = ref_mask[0].cpu().numpy().squeeze().astype(np.float32)

    # Predictions is a list of dicts with "pred_masks", "pred_scores", "pred_labels"
    target_masks = predictions[0]["pred_masks"].cpu().numpy().astype(np.float32)

    # Combine multiple target masks if present
    if target_masks.ndim == 3:
        target_mask_np = target_masks.max(axis=0)
    else:
        target_mask_np = target_masks

    return ref_mask_np, target_mask_np


def main():
    """Generate the README banner image."""
    assets_path = Path(__file__).parent.parent / "assets"
    output_path = assets_path / "readme-matcher-example.png"

    ref_image_path = assets_path / "000000286874.jpg"
    target_image_path = assets_path / "000000390341.jpg"

    # Check if masks need to be generated
    ref_mask_path = assets_path / "000000286874_mask.png"
    target_mask_path = assets_path / "000000390341_mask.png"

    if not ref_mask_path.exists() or not target_mask_path.exists():
        logger.info("Generating masks using Matcher model...")
        ref_mask_np, target_mask_np = generate_elephant_masks()

        # Save masks for future use
        cv2.imwrite(str(ref_mask_path), (ref_mask_np * 255).astype(np.uint8))
        cv2.imwrite(str(target_mask_path), (target_mask_np * 255).astype(np.uint8))
        logger.info(f"Saved masks to {assets_path}")
    else:
        logger.info("Loading existing masks...")
        ref_mask_np = cv2.imread(str(ref_mask_path), cv2.IMREAD_GRAYSCALE)
        target_mask_np = cv2.imread(str(target_mask_path), cv2.IMREAD_GRAYSCALE)

    # Load images
    ref_image = cv2.imread(str(ref_image_path))
    target_image = cv2.imread(str(target_image_path))

    if ref_image is None or target_image is None:
        raise FileNotFoundError("Failed to load elephant images from assets/")

    # Target height for all images
    target_height = 280

    # Resize images
    ref_image = resize_to_height(ref_image, target_height)
    target_image = resize_to_height(target_image, target_height)

    # Resize masks to match their corresponding images
    ref_mask_np = cv2.resize(
        ref_mask_np, (ref_image.shape[1], ref_image.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    target_mask_np = cv2.resize(
        target_mask_np, (target_image.shape[1], target_image.shape[0]), interpolation=cv2.INTER_NEAREST
    )

    # Create overlays - green color for masks
    green = (0, 200, 0)
    ref_overlay = create_mask_overlay(ref_image, ref_mask_np, color=green, alpha=0.45)
    prediction_overlay = create_mask_overlay(target_image, target_mask_np, color=green, alpha=0.45)

    # Add labels (top-right corner)
    panel1 = add_text_label(ref_image, "Reference", position="top-right")
    panel2 = add_text_label(ref_overlay, "Reference Mask", position="top-right")
    panel3 = add_text_label(target_image, "Target", position="top-right")
    panel4 = add_text_label(prediction_overlay, "Prediction", position="top-right")

    # Add padding between images
    padding = 10
    pad_color = (255, 255, 255)  # White padding

    # Create vertical padding strip
    v_pad = np.full((target_height, padding, 3), pad_color, dtype=np.uint8)

    # Concatenate horizontally with padding
    composite = np.hstack([panel1, v_pad, panel2, v_pad, panel3, v_pad, panel4])

    # Add top and bottom padding
    h_pad = np.full((padding, composite.shape[1], 3), pad_color, dtype=np.uint8)
    composite = np.vstack([h_pad, composite, h_pad])

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), composite)
    logger.info(f"Saved composite image to: {output_path}")


if __name__ == "__main__":
    main()
