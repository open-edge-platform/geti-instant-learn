# Copyright 2025 The Meta AI Authors and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Image processing utilities for SAM3 model inference."""

import math
from copy import deepcopy
from itertools import product
from typing import Any, ClassVar, Optional, Union

import numpy as np
import torch
from torch.nn import functional
from torchvision.ops.boxes import batched_nms
from transformers.image_processing_utils import BatchFeature, get_size_dict
from transformers.image_processing_utils_fast import BaseImageProcessorFast
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    pil_torch_interpolation_mapping,
)
from transformers.processing_utils import ImagesKwargs, Unpack
from transformers.utils import TensorType


# Stub for auto_docstring if not available
def auto_docstring(cls: type) -> type:
    """Decorator stub for automatic docstring generation.

    Args:
        cls (type): The class to decorate.

    Returns:
        type: The decorated class unchanged.
    """
    return cls


class FastImageProcessorKwargs(ImagesKwargs, total=False):
    """Additional keyword arguments for fast image processing.

    Attributes:
        mask_size (dict[str, int], optional): The size {"height": int, "width": int}
            to resize the segmentation maps to.
    """

    mask_size: dict[str, int]


def _compute_stability_score(masks: torch.Tensor, mask_threshold: float, stability_score_offset: int) -> torch.Tensor:
    """Compute stability scores for masks.

    One mask is always contained inside the other. Saves memory by preventing
    unnecessary cast to torch.int64.

    Args:
        masks (torch.Tensor): Binary masks of shape (batch_size, height, width).
        mask_threshold (float): Threshold for binarizing masks.
        stability_score_offset (int): Offset for stability score calculation.

    Returns:
        torch.Tensor: Stability scores of shape (batch_size,).
    """
    # One mask is always contained inside the other.
    # Save memory by preventing unnecessary cast to torch.int64
    intersections = (
        (masks > (mask_threshold + stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    )
    unions = (masks > (mask_threshold - stability_score_offset)).sum(-1, dtype=torch.int16).sum(-1, dtype=torch.int32)
    return intersections / unions


def _mask_to_rle(input_mask: torch.Tensor) -> list[dict[str, Any]]:
    """Encode masks using run-length encoding (RLE).

    Encodes masks in the format expected by pycoco tools.

    Args:
        input_mask (torch.Tensor): Binary masks of shape (batch_size, height, width).

    Returns:
        list[dict[str, Any]]: List of RLE-encoded masks with keys 'size' and 'counts'.
    """
    # Put in fortran order and flatten height and width
    batch_size, height, width = input_mask.shape
    input_mask = input_mask.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = input_mask[:, 1:] ^ input_mask[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(batch_size):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1] + 1
        if len(cur_idxs) == 0:
            # No changes => either all 0 or all 1
            # If the entire mask is 0, RLE is [height*width] or if the entire mask is 1,
            # RLE is [0, height*width].
            if input_mask[i, 0] == 0:
                out.append({"size": [height, width], "counts": [height * width]})
            else:
                out.append({"size": [height, width], "counts": [0, height * width]})
            continue
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if input_mask[i, 0] == 0 else [0]
        counts = [*counts, cur_idxs[0].item(), *btw_idxs.tolist(), height * width - cur_idxs[-1].item()]
        out.append({"size": [height, width], "counts": counts})
    return out


def _batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """Compute bounding boxes around masks.

    Computes the bounding boxes around the given input masks. The bounding boxes
    are in the XYXY format which corresponds to the following indices:
        - LEFT: left hand side of the bounding box
        - TOP: top of the bounding box
        - RIGHT: right of the bounding box
        - BOTTOM: bottom of the bounding box

    Return [0,0,0,0] for an empty mask. For input shape channel_1 x channel_2 x
    ... x height x width, the output shape is channel_1 x channel_2 x ... x 4.

    Args:
        masks (torch.Tensor): Masks of shape (batch, nb_mask, height, width).

    Returns:
        torch.Tensor: Bounding boxes in XYXY format.
    """
    # torch.max below raises an error on empty inputs, just skip in this case

    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)

    # Normalize shape to Cxheightxwidth
    shape = masks.shape
    height, width = shape[-2:]

    # Get top and bottom edges
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(height, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords += height * (~in_height)
    top_edges, _ = torch.min(in_height_coords, dim=-1)

    # Get left and right edges
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(width, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords += width * (~in_width)
    left_edges, _ = torch.min(in_width_coords, dim=-1)

    # If the mask is empty the right edge will be to the left of the left edge.
    # Replace these boxes with [0, 0, 0, 0]
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out *= (~empty_filter).unsqueeze(-1)

    # Return to original shape
    return out.reshape(*shape[:-2], 4)


def _is_box_near_crop_edge(
    boxes: torch.Tensor,
    crop_box: list[int],
    orig_box: list[int],
    atol: float = 20.0,
) -> torch.Tensor:
    """Check if boxes are near crop edges.

    Filter masks at the edge of a crop, but not at the edge of the original image.

    Args:
        boxes (torch.Tensor): Bounding boxes to check.
        crop_box (list[int]): Crop box coordinates [x1, y1, x2, y2].
        orig_box (list[int]): Original image box coordinates [x1, y1, x2, y2].
        atol (float, optional): Tolerance for edge detection. Defaults to 20.0.

    Returns:
        torch.Tensor: Boolean tensor indicating which boxes are near crop edges.
    """
    crop_box_torch = torch.as_tensor(crop_box, dtype=torch.float, device=boxes.device)
    orig_box_torch = torch.as_tensor(orig_box, dtype=torch.float, device=boxes.device)

    left, top, _, _ = crop_box
    offset = torch.tensor([[left, top, left, top]], device=boxes.device)
    # Check if boxes has a channel dimension
    if len(boxes.shape) == 3:
        offset = offset.unsqueeze(1)
    boxes = (boxes + offset).float()

    near_crop_edge = torch.isclose(boxes, crop_box_torch[None, :], atol=atol, rtol=0)
    near_image_edge = torch.isclose(boxes, orig_box_torch[None, :], atol=atol, rtol=0)
    near_crop_edge = torch.logical_and(near_crop_edge, ~near_image_edge)
    return torch.any(near_crop_edge, dim=1)


def _pad_masks(masks: torch.Tensor, crop_box: list[int], orig_height: int, orig_width: int) -> torch.Tensor:
    """Pad masks to original image size.

    Args:
        masks (torch.Tensor): Masks to pad of shape (batch, height, width).
        crop_box (list[int]): Crop box coordinates [left, top, right, bottom].
        orig_height (int): Original image height.
        orig_width (int): Original image width.

    Returns:
        torch.Tensor: Padded masks.
    """
    left, top, right, bottom = crop_box
    if left == 0 and top == 0 and right == orig_width and bottom == orig_height:
        return masks
    # Coordinate transform masks
    pad_x, pad_y = orig_width - (right - left), orig_height - (bottom - top)
    pad = (left, pad_x - left, top, pad_y - top)
    return torch.nn.functional.pad(masks, pad, value=0)


def _generate_crop_boxes(
    image: torch.Tensor,
    target_size: int,
    crop_n_layers: int = 0,
    overlap_ratio: float = 512 / 1500,
    points_per_crop: int | None = 32,
    crop_n_points_downscale_factor: list[int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate crop boxes for hierarchical image processing.

    Generates a list of crop boxes of different sizes. Each layer has (2**i)**2
    boxes for the ith layer.

    Args:
        image (torch.Tensor): Image to generate crops for.
        target_size (int): Size of the smallest crop.
        crop_n_layers (int, optional): If >0, mask prediction will be run again on
            crops of the image. Sets the number of layers to run. Defaults to 0.
        overlap_ratio (float, optional): Sets the degree to which crops overlap. In
            the first crop layer, crops will overlap by this fraction of the image
            length. Defaults to 512 / 1500.
        points_per_crop (int | None, optional): Number of points to sample per crop.
            Defaults to 32.
        crop_n_points_downscale_factor (list[int] | None, optional): The number of
            points-per-side sampled in layer n is scaled down by this factor**n.
            Defaults to None.

    Returns:
        tuple: Containing:
            - crop_boxes (torch.Tensor): Crop box coordinates.
            - points_per_crop (torch.Tensor): Points sampled per crop.
            - cropped_images (torch.Tensor): Cropped image patches.
            - input_labels (torch.Tensor): Input labels for crops.

    Raises:
        TypeError: If image is a list (only single images allowed).
    """
    if isinstance(image, list):
        msg = "Only one image is allowed for crop generation."
        raise TypeError(msg)
    if crop_n_points_downscale_factor is None:
        crop_n_points_downscale_factor = 1
    original_size = image.shape[-2:]

    points_grid = []
    for i in range(crop_n_layers + 1):
        n_points = int(points_per_crop / (crop_n_points_downscale_factor**i))
        points_grid.append(_build_point_grid(n_points))

    crop_boxes, layer_idxs = _generate_per_layer_crops(crop_n_layers, overlap_ratio, original_size)

    cropped_images, point_grid_per_crop = _generate_crop_images(
        crop_boxes,
        image,
        points_grid,
        layer_idxs,
        target_size,
        original_size,
    )
    crop_boxes_t = torch.tensor(crop_boxes)
    crop_boxes_t = crop_boxes_t.float()
    points_per_crop_t = torch.stack(point_grid_per_crop)
    points_per_crop_t = points_per_crop_t.unsqueeze(0).permute(0, 2, 1, 3)
    cropped_images_t = torch.stack(cropped_images)

    input_labels = torch.ones_like(points_per_crop_t[:, :, :, 0], dtype=torch.int64)

    return crop_boxes_t, points_per_crop_t, cropped_images_t, input_labels


def _generate_per_layer_crops(
    crop_n_layers: int,
    overlap_ratio: float,
    original_size: tuple[int, int],
) -> tuple[list[list[int]], list[int]]:
    """Generate crops for each layer.

    Generates 2 ** (layers idx + 1) crops for each crop_n_layers. Crops are in
    the XYWH format: X (x coordinate), Y (y coordinate), W (width), H (height).

    Args:
        crop_n_layers (int): Number of crop layers.
        overlap_ratio (float): Overlap ratio for crops.
        original_size (tuple[int, int]): Original image size (height, width).

    Returns:
        tuple: Containing crop_boxes and layer indices.
    """
    crop_boxes, layer_idxs = [], []
    im_height, im_width = original_size
    short_side = min(im_height, im_width)

    # Original image
    crop_boxes.append([0, 0, im_width, im_height])
    layer_idxs.append(0)
    for i_layer in range(crop_n_layers):
        n_crops_per_side = 2 ** (i_layer + 1)
        overlap = int(overlap_ratio * short_side * (2 / n_crops_per_side))

        crop_width = math.ceil((overlap * (n_crops_per_side - 1) + im_width) / n_crops_per_side)
        crop_height = math.ceil((overlap * (n_crops_per_side - 1) + im_height) / n_crops_per_side)

        crop_box_x0 = [int((crop_width - overlap) * i) for i in range(n_crops_per_side)]
        crop_box_y0 = [int((crop_height - overlap) * i) for i in range(n_crops_per_side)]

        for left, top in product(crop_box_x0, crop_box_y0):
            box = [left, top, min(left + crop_width, im_width), min(top + crop_height, im_height)]
            crop_boxes.append(box)
            layer_idxs.append(i_layer + 1)

    return crop_boxes, layer_idxs


def _build_point_grid(n_per_side: int) -> torch.Tensor:
    """Generate a 2D grid of evenly spaced points.

    Generates a 2D grid of points evenly spaced in [0,1]x[0,1].

    Args:
        n_per_side (int): Number of points per side.

    Returns:
        torch.Tensor: Grid of points of shape (n_per_side**2, 2).
    """
    offset = 1 / (2 * n_per_side)
    points_one_side = torch.linspace(offset, 1 - offset, n_per_side)
    points_x = torch.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = torch.tile(points_one_side[:, None], (1, n_per_side))
    return torch.stack([points_x, points_y], dim=-1).reshape(-1, 2)


def _generate_crop_images(
    crop_boxes: list,
    image: torch.Tensor,
    points_grid: list,
    layer_idxs: list[int],
    target_size: int,
    original_size: tuple[int, int],
    _input_data_format: ChannelDimension | None = None,  # noqa: ARG001
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Generate cropped images and corresponding point grids.

    Takes as an input bounding boxes that are used to crop the image. Based in the
    crops, the corresponding points are also passed.

    Args:
        crop_boxes (list): List of crop box coordinates.
        image (torch.Tensor): Image to crop.
        points_grid (list): Grid of points for each layer.
        layer_idxs (list[int]): Layer indices for each crop.
        target_size (int): Target size for crops.
        original_size (tuple[int, int]): Original image size.
        _input_data_format (ChannelDimension | None, optional): Channel dimension format.
            Defaults to None.

    Returns:
        tuple: Containing cropped images and point grids per crop.
    """
    cropped_images = []
    total_points_per_crop = []
    for i, crop_box in enumerate(crop_boxes):
        left, top, right, bottom = crop_box
        cropped_im = image[:, top:bottom, left:right]

        cropped_images.append(cropped_im)

        cropped_im_size = cropped_im.shape[-2:]
        points_scale = torch.tensor(cropped_im_size).flip(dims=(0,)).unsqueeze(0)

        points = points_grid[layer_idxs[i]] * points_scale
        normalized_points = _normalize_coordinates(target_size, points, original_size)
        total_points_per_crop.append(normalized_points)

    return cropped_images, total_points_per_crop


def _normalize_coordinates(
    target_size: int,
    coords: torch.Tensor,
    original_size: tuple[int, int],
    is_bounding_box: bool = False,
) -> torch.Tensor:
    """Normalize coordinates to target image size.

    Expects a tensor of length 2 in the final dimension. Requires the original
    image size in (height, width) format.

    Args:
        target_size (int): Target image size.
        coords (torch.Tensor): Coordinates to normalize.
        original_size (tuple[int, int]): Original image size (height, width).
        is_bounding_box (bool, optional): Whether coordinates are bounding boxes.
            Defaults to False.

    Returns:
        torch.Tensor: Normalized coordinates.
    """
    old_height, old_width = original_size

    scale = target_size * 1.0 / max(old_height, old_width)
    new_height, new_width = old_height * scale, old_width * scale
    new_width = int(new_width + 0.5)
    new_height = int(new_height + 0.5)

    coords = deepcopy(coords).float()

    if is_bounding_box:
        coords = coords.reshape(-1, 2, 2)

    coords[..., 0] *= new_width / old_width
    coords[..., 1] *= new_height / old_height

    if is_bounding_box:
        coords = coords.reshape(-1, 4)

    return coords


def _rle_to_mask(rle: dict[str, Any]) -> torch.Tensor:
    """Compute binary mask from RLE encoding.

    Compute a binary mask from an uncompressed RLE representation.

    Args:
        rle (dict[str, Any]): RLE dictionary with keys 'size' and 'counts'.

    Returns:
        torch.Tensor: Binary mask tensor.
    """
    height, width = rle["size"]
    mask = torch.empty(height * width, dtype=torch.bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity = not parity
    mask = mask.reshape(width, height)
    return mask.transpose(0, 1)  # Reshape to original shape


def _post_process_for_mask_generation(
    rle_masks: list,
    iou_scores: torch.Tensor,
    mask_boxes: torch.Tensor,
    amg_crops_nms_thresh: float = 0.7,
) -> tuple[list, torch.Tensor, list, torch.Tensor]:
    """Post-process masks using NMS algorithm.

    Perform NMS (Non Maximum Suppression) on the outputs.

    Args:
        rle_masks (list): RLE-encoded masks.
        iou_scores (torch.Tensor): IoU scores of shape (nb_masks, 1).
        mask_boxes (torch.Tensor): Bounding boxes for masks.
        amg_crops_nms_thresh (float, optional): NMS threshold. Defaults to 0.7.

    Returns:
        tuple: Containing filtered masks, scores, RLE masks, and boxes.
    """
    keep_by_nms = batched_nms(
        boxes=mask_boxes.float(),
        scores=iou_scores,
        idxs=torch.zeros(mask_boxes.shape[0]),
        iou_threshold=amg_crops_nms_thresh,
    )

    iou_scores = iou_scores[keep_by_nms]
    rle_masks = [rle_masks[i] for i in keep_by_nms]
    mask_boxes = mask_boxes[keep_by_nms]
    masks = [_rle_to_mask(rle) for rle in rle_masks]

    return masks, iou_scores, rle_masks, mask_boxes


def _scale_boxes(boxes: torch.Tensor, target_sizes: list[tuple[int, int]] | torch.Tensor) -> torch.Tensor:
    """Scale batch of bounding boxes to target sizes.

    Args:
        boxes (torch.Tensor): Bounding boxes of shape (batch_size, num_boxes, 4).
            Each box is expected to be in (x1, y1, x2, y2) format.
        target_sizes (list[tuple[int, int]] | torch.Tensor): Target sizes to scale
            to. Each target size is expected to be in (height, width) format.

    Returns:
        torch.Tensor: Scaled bounding boxes of shape (batch_size, num_boxes, 4).

    Raises:
        TypeError: If target_sizes is not a list, tuple, or torch.Tensor.
    """
    if isinstance(target_sizes, (list, tuple)):
        image_height = torch.tensor([i[0] for i in target_sizes])
        image_width = torch.tensor([i[1] for i in target_sizes])
    elif isinstance(target_sizes, torch.Tensor):
        image_height, image_width = target_sizes.unbind(1)
    else:
        msg = "`target_sizes` must be a list, tuple or torch.Tensor"
        raise TypeError(msg)

    scale_factor = torch.stack([image_width, image_height, image_width, image_height], dim=1)
    scale_factor = scale_factor.unsqueeze(1).to(boxes.device)
    boxes *= scale_factor
    return boxes


@auto_docstring
class ImageProcessorFast(BaseImageProcessorFast):
    """Fast image processor for SAM3 model.

    Image processor for fast SAM3 inference.
    """

    resample = PILImageResampling.BILINEAR
    image_mean = IMAGENET_STANDARD_MEAN
    image_std = IMAGENET_STANDARD_STD
    size: ClassVar[dict[str, int]] = {"height": 1008, "width": 1008}
    mask_size: ClassVar[dict[str, int]] = {"height": 288, "width": 288}
    do_resize = True
    do_rescale = True
    do_normalize = True
    do_convert_rgb = True

    valid_kwargs = FastImageProcessorKwargs

    # modular artefacts
    do_pad = None
    pad_size = None
    mask_pad_size = None

    def __init__(self, **kwargs: Unpack[FastImageProcessorKwargs]) -> None:
        """Initialize the image processor.

        Args:
            **kwargs: Additional keyword arguments for image processing.
        """
        super().__init__(**kwargs)

    def _further_process_kwargs(
        self,
        size: SizeDict | None = None,
        mask_size: SizeDict | None = None,
        default_to_square: bool | None = None,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        data_format: ChannelDimension | None = None,
        **kwargs: Any,
    ) -> dict:
        """Update kwargs that need further processing before being validated.

        Can be overridden by subclasses to customize the processing of kwargs.
        """
        if kwargs is None:
            kwargs = {}
        if size is not None:
            size = SizeDict(**get_size_dict(size=size, default_to_square=default_to_square))
        if mask_size is not None:
            mask_size = SizeDict(**get_size_dict(mask_size, param_name="mask_size"))
        if isinstance(image_mean, list):
            image_mean = tuple(image_mean)
        if isinstance(image_std, list):
            image_std = tuple(image_std)
        if data_format is None:
            data_format = ChannelDimension.FIRST

        kwargs["size"] = size
        kwargs["mask_size"] = mask_size
        kwargs["image_mean"] = image_mean
        kwargs["image_std"] = image_std
        kwargs["data_format"] = data_format

        # torch resize uses interpolation instead of resample
        # Check if resample is an int before checking if it's an instance of PILImageResampling
        # because if pillow < 9.1.0, resample is an int and PILImageResampling is a module.
        # Checking PILImageResampling will fail with error `TypeError: isinstance() arg 2 must be a type or tuple of
        # types`.
        resample = kwargs.pop("resample")
        kwargs["interpolation"] = (
            pil_torch_interpolation_mapping[resample] if isinstance(resample, (PILImageResampling, int)) else resample
        )

        return kwargs

    @auto_docstring
    def preprocess(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None = None,
        **kwargs: Unpack[FastImageProcessorKwargs],
    ) -> BatchFeature:
        r"""Preprocess images and segmentation maps.

        Args:
            images (ImageInput): Images to preprocess.
            segmentation_maps (ImageInput, optional): Segmentation maps to preprocess.
            **kwargs (Unpack[FastImageProcessorKwargs]): Additional keyword arguments for preprocessing.

        Returns:
            BatchFeature: Preprocessed outputs.
        """
        # Ensure do_convert_rgb is in kwargs for compatibility with stable transformers
        if "do_convert_rgb" not in kwargs:
            kwargs["do_convert_rgb"] = self.do_convert_rgb
        return super().preprocess(images, segmentation_maps, **kwargs)

    def _preprocess_image_like_inputs(
        self,
        images: ImageInput,
        segmentation_maps: ImageInput | None,
        do_convert_rgb: bool,
        input_data_format: ChannelDimension,
        device: Union[str, "torch.device"] | None = None,
        **kwargs: Unpack[FastImageProcessorKwargs],
    ) -> BatchFeature:
        """Preprocess image-like inputs."""
        images = self._prepare_image_like_inputs(
            images=images,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
        )
        original_sizes = [image.shape[-2:] for image in images]
        images_kwargs = kwargs.copy()
        pixel_values = self._preprocess(images, **images_kwargs)
        data = {
            "pixel_values": pixel_values,
            "original_sizes": original_sizes,
        }

        if segmentation_maps is not None:
            processed_segmentation_maps = self._prepare_image_like_inputs(
                images=segmentation_maps,
                expected_ndims=2,
                do_convert_rgb=False,
                input_data_format=ChannelDimension.FIRST,
            )

            segmentation_maps_kwargs = kwargs.copy()
            segmentation_maps_kwargs.update(
                {
                    "do_normalize": False,
                    "do_rescale": False,
                    "interpolation": pil_torch_interpolation_mapping[PILImageResampling.NEAREST],
                    "size": segmentation_maps_kwargs.pop("mask_size"),
                },
            )
            processed_segmentation_maps = self._preprocess(
                images=processed_segmentation_maps,
                **segmentation_maps_kwargs,
            )
            data["labels"] = processed_segmentation_maps.squeeze(1).to(torch.int64)

        return BatchFeature(data=data, tensor_type=kwargs["return_tensors"])

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        return_tensors: str | TensorType | None,
        disable_grouping: bool | None = None,
        **kwargs: Any,
    ) -> "torch.Tensor":
        return (
            super()
            ._preprocess(images, return_tensors=return_tensors, disable_grouping=disable_grouping, **kwargs)
            .pixel_values
        )

    def generate_crop_boxes(
        self,
        image: "torch.Tensor",
        target_size: int,
        crop_n_layers: int = 0,
        overlap_ratio: float = 512 / 1500,
        points_per_crop: int | None = 32,
        crop_n_points_downscale_factor: list[int] | None = 1,
        device: Optional["torch.device"] = None,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Generates a list of crop boxes of different sizes. Each layer has (2**i)**2 boxes for the ith layer.

        Args:
            image (`torch.Tensor`):
                Input original image
            target_size (`int`):
                Target size of the resized image
            crop_n_layers (`int`, *optional*, defaults to 0):
                If >0, mask prediction will be run again on crops of the image. Sets the number of layers to run, where
                each layer has 2**i_layer number of image crops.
            overlap_ratio (`float`, *optional*, defaults to 512/1500):
                Sets the degree to which crops overlap. In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            points_per_crop (`int`, *optional*, defaults to 32):
                Number of points to sam3ple from each crop.
            crop_n_points_downscale_factor (`list[int]`, *optional*, defaults to 1):
                The number of points-per-side sam3pled in layer n is scaled down by crop_n_points_downscale_factor**n.
            device (`torch.device`, *optional*, defaults to None):
                Device to use for the computation. If None, cpu will be used.
            input_data_format (`str` or `ChannelDimension`, *optional*):
                The channel dimension format of the input image. If not provided, it will be inferred.
            return_tensors (`str`, *optional*, defaults to `pt`):
                If `pt`, returns `torch.Tensor`.
        """
        image = self._process_image(image)
        crop_boxes, points_per_crop, cropped_images, input_labels = _generate_crop_boxes(
            image,
            target_size,
            crop_n_layers,
            overlap_ratio,
            points_per_crop,
            crop_n_points_downscale_factor,
        )
        if device is None:
            device = torch.device("cpu")
        crop_boxes = crop_boxes.to(device)
        points_per_crop = points_per_crop.to(device)
        # cropped_images stays as torch.Tensor
        input_labels = input_labels.to(device)

        return crop_boxes, points_per_crop, cropped_images, input_labels

    def filter_masks(
        self,
        masks: "torch.Tensor",
        iou_scores: "torch.Tensor",
        original_size: tuple[int, int],
        cropped_box_image: list[int],
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        mask_threshold: float = 0,
        stability_score_offset: int = 1,
    ) -> tuple[list[dict[str, Any]], "torch.Tensor", "torch.Tensor"]:
        """Filters the predicted masks by selecting only the ones that meets several criteria.

        The first criterion being that the iou scores needs to be greater than `pred_iou_thresh`.
        The second criterion is that the stability score needs to be greater than
        `stability_score_thresh`. The method also converts the predicted masks to bounding boxes
        and pad the predicted masks if necessary.

        Args:
            masks (`torch.Tensor`):
                Input masks.
            iou_scores (`torch.Tensor`):
                List of IoU scores.
            original_size (`tuple[int,int]`):
                Size of the original image.
            cropped_box_image (`torch.Tensor`):
                The cropped image.
            pred_iou_thresh (`float`, *optional*, defaults to 0.88):
                The threshold for the iou scores.
            stability_score_thresh (`float`, *optional*, defaults to 0.95):
                The threshold for the stability score.
            mask_threshold (`float`, *optional*, defaults to 0):
                The threshold for the predicted masks.
            stability_score_offset (`float`, *optional*, defaults to 1):
                The offset for the stability score used in the `_compute_stability_score` method.

        Raises:
            ValueError: If masks and iou_scores have different batch sizes.
        """
        original_height, original_width = original_size
        iou_scores = iou_scores.flatten(0, 1)
        masks = masks.flatten(0, 1)

        if masks.shape[0] != iou_scores.shape[0]:
            msg = "masks and iou_scores must have the sam3e batch size."
            raise ValueError(msg)

        if masks.device != iou_scores.device:
            iou_scores = iou_scores.to(masks.device)

        batch_size = masks.shape[0]

        keep_mask = torch.ones(batch_size, dtype=torch.bool, device=masks.device)

        if pred_iou_thresh > 0.0:
            keep_mask &= iou_scores > pred_iou_thresh

        # compute stability score
        if stability_score_thresh > 0.0:
            stability_scores = _compute_stability_score(masks, mask_threshold, stability_score_offset)
            keep_mask &= stability_scores > stability_score_thresh

        scores = iou_scores[keep_mask]
        masks = masks[keep_mask]

        # binarize masks
        masks = masks > mask_threshold
        converted_boxes = _batched_mask_to_box(masks)

        keep_mask = ~_is_box_near_crop_edge(
            converted_boxes,
            cropped_box_image,
            [0, 0, original_width, original_height],
        )

        scores = scores[keep_mask]
        masks = masks[keep_mask]
        converted_boxes = converted_boxes[keep_mask]

        masks = _pad_masks(masks, cropped_box_image, original_height, original_width)
        # conversion to rle is necessary to run non-maximum suppression
        masks = _mask_to_rle(masks)

        return masks, scores, converted_boxes

    def post_process_masks(
        self,
        masks: list[torch.Tensor] | torch.Tensor | np.ndarray,
        original_sizes: list[tuple[int, int]] | torch.Tensor,
        mask_threshold: float = 0.0,
        binarize: bool = True,
        max_hole_area: float = 0.0,  # noqa: ARG002
        max_sprinkle_area: float = 0.0,  # noqa: ARG002
        apply_non_overlapping_constraints: bool = False,
        **kwargs: Any,  # noqa: ARG002
    ) -> list["torch.Tensor"]:
        """Remove padding and upscale masks to the original image size.

        Args:
            masks (`Union[torch.Tensor, List[torch.Tensor], np.ndarray, List[np.ndarray]]`):
                Batched masks from the mask_decoder in (batch_size, num_channels, height, width) format.
            original_sizes (`Union[torch.Tensor, List[Tuple[int,int]]]`):
                The original sizes of each image before it was resized to the model's expected input shape, in (height,
                width) format.
            mask_threshold (`float`, *optional*, defaults to 0.0):
                Threshold for binarization and post-processing operations.
            binarize (`bool`, *optional*, defaults to `True`):
                Whether to binarize the masks.
            max_hole_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a hole to fill.
            max_sprinkle_area (`float`, *optional*, defaults to 0.0):
                The maximum area of a sprinkle to fill.
            apply_non_overlapping_constraints (`bool`, *optional*, defaults to `False`):
                Whether to apply non-overlapping constraints to the masks.
            **kwargs: Additional keyword arguments (unused, for compatibility).

        Returns:
            (`torch.Tensor`): Batched masks in batch_size, num_channels, height, width) format, where (height, width)
            is given by original_size.

        Raises:
            TypeError: If input masks are not torch.Tensor or np.ndarray.
        """
        if isinstance(original_sizes, (torch.Tensor, np.ndarray)):
            original_sizes = original_sizes.tolist()
        # TODO(future): add connected components kernel for postprocessing  # noqa: TD003, FIX002
        output_masks = []
        for i, original_size in enumerate(original_sizes):
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.from_numpy(masks[i])
            elif not isinstance(masks[i], torch.Tensor):
                msg = "Input masks should be a list of `torch.tensors` or a list of `np.ndarray`"
                raise TypeError(msg)
            interpolated_mask = functional.interpolate(masks[i], original_size, mode="bilinear", align_corners=False)
            if apply_non_overlapping_constraints:
                interpolated_mask = self._apply_non_overlapping_constraints(interpolated_mask)
            if binarize:
                interpolated_mask = interpolated_mask > mask_threshold
            output_masks.append(interpolated_mask)

        return output_masks

    def post_process_for_mask_generation(
        self,
        all_masks: "torch.Tensor",
        all_scores: "torch.Tensor",
        all_boxes: "torch.Tensor",
        crops_nms_thresh: float,
    ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        """Post processes mask that are generated by calling the Non Maximum Suppression algorithm.

        Post-processes mask generated for predicted masks.

        Args:
            all_masks (`torch.Tensor`):
                List of all predicted segmentation masks
            all_scores (`torch.Tensor`):
                List of all predicted iou scores
            all_boxes (`torch.Tensor`):
                List of all bounding boxes of the predicted masks
            crops_nms_thresh (`float`):
                Threshold for NMS (Non Maximum Suppression) algorithm.
        """
        return _post_process_for_mask_generation(all_masks, all_scores, all_boxes, crops_nms_thresh)

    def _apply_non_overlapping_constraints(self, pred_masks: torch.Tensor) -> torch.Tensor:
        """Apply non-overlapping constraints to the object scores in pred_masks.

        Here we keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        return torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))

    def post_process_semantic_segmentation(
        self,
        outputs: dict | Any,
        target_sizes: list[tuple] | None = None,
        threshold: float = 0.5,
    ) -> list["torch.Tensor"]:
        """Converts the output of [`Model`] into semantic segmentation maps.

        Args:
            outputs (`dict` or dataclass):
                Raw outputs of the model containing semantic_seg.
            target_sizes (`list[tuple]` of length `batch_size`, *optional*):
                List of tuples corresponding to the requested final size (height, width) of each prediction. If unset,
                predictions will not be resized.
            threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the semantic segmentation masks.

        Returns:
            semantic_segmentation: `list[torch.Tensor]` of length `batch_size`, where each item is a semantic
            segmentation map of shape (height, width) corresponding to the target_sizes entry (if `target_sizes` is
            specified). Each entry is a binary mask (0 or 1).

        Raises:
            ValueError: If semantic segmentation output is not available or target_sizes length doesn't
                match batch size.
        """
        # Get semantic segmentation output
        # semantic_seg has shape (batch_size, 1, height, width)
        semantic_logits = outputs["semantic_seg"] if isinstance(outputs, dict) else outputs.semantic_seg

        if semantic_logits is None:
            msg = (
                "Semantic segmentation output is not available in the model outputs. "
                "Make sure the model was run with semantic segmentation enabled."
            )
            raise ValueError(
                msg,
            )

        # Apply sigmoid to convert logits to probabilities
        semantic_probs = semantic_logits.sigmoid()

        # Resize and binarize semantic segmentation maps
        if target_sizes is not None:
            if len(semantic_logits) != len(target_sizes):
                msg = "Make sure that you pass in as many target sizes as the batch dimension of the logits"
                raise ValueError(
                    msg,
                )

            semantic_segmentation = []

            for idx in range(len(semantic_logits)):
                resized_probs = torch.nn.functional.interpolate(
                    semantic_probs[idx].unsqueeze(dim=0),
                    size=target_sizes[idx],
                    mode="bilinear",
                    align_corners=False,
                )
                # Binarize: values > threshold become 1, otherwise 0
                semantic_map = (resized_probs[0, 0] > threshold).to(torch.long)
                semantic_segmentation.append(semantic_map)
        else:
            # Binarize without resizing
            semantic_segmentation = (semantic_probs[:, 0] > threshold).to(torch.long)
            semantic_segmentation = [semantic_segmentation[i] for i in range(semantic_segmentation.shape[0])]

        return semantic_segmentation

    def post_process_object_detection(
        self,
        outputs: dict | Any,
        threshold: float = 0.3,
        target_sizes: list[tuple] | None = None,
    ) -> list[dict[str, "torch.Tensor"]]:
        """Converts the raw output of [`Model`] into final bounding boxes.

        Converts to (top_left_x, top_left_y, bottom_right_x, bottom_right_y) format.

        Args:
            outputs (`dict` or dataclass):
                Raw outputs of the model containing pred_boxes, pred_logits, and optionally presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep object detection predictions.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted box on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.

        Raises:
            ValueError: If target_sizes length doesn't match batch size.
        """
        # Support both dict and dataclass-style outputs
        pred_logits = (
            outputs["pred_logits"] if isinstance(outputs, dict) else outputs.pred_logits
        )  # (batch_size, num_queries)
        pred_boxes = (
            outputs["pred_boxes"] if isinstance(outputs, dict) else outputs.pred_boxes
        )  # (batch_size, num_queries, 4) in xyxy format
        presence_logits = (
            outputs["presence_logits"] if isinstance(outputs, dict) else outputs.presence_logits
        )  # (batch_size, 1) or None

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            msg = "Make sure that you pass in as many target sizes as images"
            raise ValueError(msg)

        # Compute scores: combine pred_logits with presence_logits if available
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (batch_size, 1)
            batch_scores *= presence_scores  # Broadcast multiplication

        # Boxes are already in xyxy format from the model
        batch_boxes = pred_boxes

        # Convert from relative [0, 1] to absolute [0, height/width] coordinates
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for scores, boxes in zip(batch_scores, batch_boxes, strict=False):
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            results.append({"scores": scores, "boxes": boxes})

        return results

    def post_process_instance_segmentation(
        self,
        outputs: dict | Any,
        threshold: float = 0.3,
        mask_threshold: float = 0.5,
        target_sizes: list[tuple] | None = None,
    ) -> list[dict[str, "torch.Tensor"]]:
        """Converts the raw output of [`Model`] into instance segmentation predictions with bounding boxes and masks.

        Args:
            outputs (`dict` or dataclass):
                Raw outputs of the model containing pred_boxes, pred_logits, pred_masks, and optionally
                presence_logits.
            threshold (`float`, *optional*, defaults to 0.3):
                Score threshold to keep instance predictions.
            mask_threshold (`float`, *optional*, defaults to 0.5):
                Threshold for binarizing the predicted masks.
            target_sizes (`list[tuple[int, int]]`, *optional*):
                List of tuples (`tuple[int, int]`) containing the target size `(height, width)` of each image in the
                batch. If unset, predictions will not be resized.

        Returns:
            `list[dict]`: A list of dictionaries, each dictionary containing the following keys:
                - **scores** (`torch.Tensor`): The confidence scores for each predicted instance on the image.
                - **boxes** (`torch.Tensor`): Image bounding boxes in (top_left_x, top_left_y, bottom_right_x,
                  bottom_right_y) format.
                - **masks** (`torch.Tensor`): Binary segmentation masks for each instance, shape (num_instances,
                  height, width).

        Raises:
            ValueError: If target_sizes length doesn't match batch size.
        """
        # Support both dict and dataclass-style outputs
        pred_logits = (
            outputs["pred_logits"] if isinstance(outputs, dict) else outputs.pred_logits
        )  # (batch_size, num_queries)
        pred_boxes = (
            outputs["pred_boxes"] if isinstance(outputs, dict) else outputs.pred_boxes
        )  # (batch_size, num_queries, 4) in xyxy format
        pred_masks = (
            outputs["pred_masks"] if isinstance(outputs, dict) else outputs.pred_masks
        )  # (batch_size, num_queries, height, width)
        presence_logits = (
            outputs["presence_logits"] if isinstance(outputs, dict) else outputs.presence_logits
        )  # (batch_size, 1) or None

        batch_size = pred_logits.shape[0]

        if target_sizes is not None and len(target_sizes) != batch_size:
            msg = "Make sure that you pass in as many target sizes as images"
            raise ValueError(msg)

        # Compute scores: combine pred_logits with presence_logits if available
        batch_scores = pred_logits.sigmoid()
        if presence_logits is not None:
            presence_scores = presence_logits.sigmoid()  # (batch_size, 1)
            batch_scores *= presence_scores  # Broadcast multiplication

        # Apply sigmoid to mask logits
        batch_masks = pred_masks.sigmoid()

        # Boxes are already in xyxy format from the model
        batch_boxes = pred_boxes

        # Scale boxes to target sizes if provided
        if target_sizes is not None:
            batch_boxes = _scale_boxes(batch_boxes, target_sizes)

        results = []
        for idx, (scores, boxes, masks) in enumerate(zip(batch_scores, batch_boxes, batch_masks, strict=False)):
            # Filter by score threshold
            keep = scores > threshold
            scores = scores[keep]
            boxes = boxes[keep]
            masks = masks[keep]  # (num_keep, height, width)

            # Resize masks to target size if provided
            if target_sizes is not None:
                target_size = target_sizes[idx]
                if len(masks) > 0:
                    masks = torch.nn.functional.interpolate(
                        masks.unsqueeze(0),  # (1, num_keep, height, width)
                        size=target_size,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(0)  # (num_keep, target_height, target_width)

            # Binarize masks
            masks = (masks > mask_threshold).to(torch.long)

            results.append({"scores": scores, "boxes": boxes, "masks": masks})

        return results
