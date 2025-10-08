# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM decoder."""

from itertools import zip_longest
from logging import getLogger

import torch
from segment_anything_hq.predictor import SamPredictor as SamHQPredictor
from torch import nn
from torchvision.ops import masks_to_boxes, nms

from getiprompt.types import Boxes, Image, Masks, Points, Priors, Similarities
from getiprompt.utils import ResizeLongestSide

logger = getLogger("Geti Prompt")


class SamDecoder(nn.Module):
    """This Segmenter uses SAM to create masks based on points.

    Examples:
        >>> from getiprompt.models.models import load_sam_model
        >>> from getiprompt.processes.segmenters import SamDecoder
        >>> from getiprompt.types import Image, Masks, Points, Priors, Similarities
        >>> import torch
        >>> import numpy as np
        >>> sam_predictor = load_sam_model(backbone_name="MobileSAM")
        >>> segmenter = SamDecoder(sam_predictor=sam_predictor)
        >>> image = Image(np.zeros((1024, 1024, 3), dtype=np.uint8))
        >>> priors = Priors()
        >>> points = torch.tensor([[512, 512, 0.9, 1], [100, 100, 0.8, 0]]) # fg, bg
        >>> priors.points.add(points, class_id=1)
        >>> similarities = Similarities()
        >>> similarities.add(torch.ones(1, 1024, 1024), class_id=1)
        >>> masks, used_points = segmenter(
        ...     images=[image],
        ...     priors=[priors],
        ...     similarities=[similarities],
        ... )
        >>> isinstance(masks, list) and isinstance(masks[0], Masks) and len(masks[0].get(1)) == 1
        True
        >>> isinstance(used_points, list) and isinstance(used_points[0], Points) and len(used_points[0].get(1)[0]) == 2
        True
    """

    def __init__(
        self,
        sam_predictor: SamHQPredictor,
        mask_similarity_threshold: float = 0.38,
        nms_iou_threshold: float = 0.1,
    ) -> None:
        """This Segmenter uses SAM to create masks based on points.

        Args:
            sam_predictor: The SAM predictor.
            mask_similarity_threshold: The similarity threshold for the mask.
            nms_iou_threshold: The IoU threshold for the NMS.
        """
        super().__init__()
        self.predictor = sam_predictor
        self.mask_similarity_threshold = mask_similarity_threshold
        self.nms_iou_threshold = nms_iou_threshold

        if hasattr(self.predictor.model.image_encoder, "img_size"):
            img_size = self.predictor.model.image_encoder.img_size
        elif hasattr(self.predictor.model, "image_size"):
            img_size = self.predictor.model.image_size
        else:
            # fallback to 1024
            logger.warning("Image size not found in the model. Using 1024 as default.")
            img_size = 1024

        self.transform = ResizeLongestSide(img_size)

    def preprocess_inputs(
        self,
        images: list[Image],
        priors: list[Priors] | None = None,
    ) -> tuple[list[torch.Tensor], list[dict[int, torch.Tensor]], list[tuple[int, int]]]:
        """Preprocess the inputs.

        Args:
            images: The images to preprocess.
            priors: The priors to preprocess.

        Returns:
            A tuple of preprocessed images, preprocessed points, and original sizes.

        TODO(Eugene): Unwrap getiprompt.Priors and getiprompt.Image into pure tensors for the SAM predictor.
        Consider moving this to a dedicated preprocessing module once the data flow is finalized.
        https://github.com/open-edge-platform/geti-prompt/issues/174

        """
        preprocessed_images = []
        preprocessed_points = []
        preprocessed_boxes = []
        original_sizes = []
        device = self.predictor.device

        for image, priors_per_image in zip_longest(images, priors, fillvalue=None):
            # Preprocess image using SamPredictor transform
            input_image = self.transform.apply_image(image.data)
            input_image_torch = torch.as_tensor(input_image, device=device)
            input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]
            ori_size = image.data.shape[:2]
            preprocessed_images.append(input_image_torch)
            original_sizes.append(ori_size)
            class_points = {}
            class_boxes = {}

            # Preprocess points for each class
            for class_id, points_per_class in priors_per_image.points.data.items():
                if len(points_per_class) != 1:
                    msg = (
                        f"Each class must have exactly one prior map (got {len(points_per_class)} for class {class_id})"
                    )
                    raise ValueError(msg)
                points = points_per_class[0].to(device)
                coords = points[:, :2]
                transformed_coords = self.transform.apply_coords_torch(coords, ori_size)
                transformed_points = points.clone()
                transformed_points[:, :2] = transformed_coords
                class_points[class_id] = transformed_points

            # Preprocess boxes for each class
            for class_id, boxes_per_class in priors_per_image.boxes.data.items():
                if len(boxes_per_class) != 1:
                    msg = (
                        f"Each class must have exactly one prior map (got {len(boxes_per_class)} for class {class_id})"
                    )
                    raise ValueError(msg)
                boxes = boxes_per_class[0].to(device)
                box_coords = boxes[:, :4]
                transformed_boxes = self.transform.apply_boxes_torch(box_coords, ori_size)
                transformed_boxes = torch.cat([transformed_boxes, boxes[:, 4:]], dim=1)
                class_boxes[class_id] = transformed_boxes

            preprocessed_points.append(class_points)
            preprocessed_boxes.append(class_boxes)

        # find unique labels
        unique_labels = set()
        for class_id in class_points:
            unique_labels.add(class_id)
        for class_id in class_boxes:
            unique_labels.add(class_id)
        unique_labels = sorted(unique_labels)

        return (
            preprocessed_images,
            preprocessed_points,
            preprocessed_boxes,
            unique_labels,
            original_sizes,
        )

    @torch.inference_mode()
    def forward(
        self,
        images: list[Image],
        priors: list[Priors] | None = None,
        similarities: list[Similarities] | None = None,
    ) -> tuple[list[Masks], list[Points], list[Boxes]]:
        """Forward pass.

        Args:
            images: The images to predict masks from.
            priors: The priors to predict masks from.
            similarities: The similarities to predict masks from.
        """
        if similarities is None:
            similarities = []
        masks_per_image: list[Masks] = []
        points_per_image: list[Points] = []
        boxes_per_image: list[Boxes] = []

        (
            preprocessed_images,
            preprocessed_points,
            preprocessed_boxes,
            labels,
            original_sizes,
        ) = self.preprocess_inputs(images, priors)

        for (
            preprocessed_image,
            preprocessed_points_per_image,
            preprocessed_boxes_per_image,
            similarities_per_image,
            original_size,
        ) in zip_longest(
            preprocessed_images,
            preprocessed_points,
            preprocessed_boxes,
            similarities,
            original_sizes,
            fillvalue=None,
        ):
            if preprocessed_points_per_image is None:
                continue

            # Set the preprocessed image in the predictor
            self.predictor.set_torch_image(preprocessed_image, original_size)
            if preprocessed_points_per_image or preprocessed_boxes_per_image:
                masks, points_used, boxes_used = self.predict_single(
                    preprocessed_points_per_image,
                    preprocessed_boxes_per_image,
                    labels,
                    similarities_per_image,
                    original_size,
                )
                points_per_image.append(points_used)
                masks_per_image.append(masks)
                boxes_per_image.append(boxes_used)
            else:
                points_per_image.append(Points())
                masks_per_image.append(Masks())
                boxes_per_image.append(Boxes())
        return masks_per_image, points_per_image, boxes_per_image

    @staticmethod
    def point_preprocess(
        points: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Preprocess the points.

        Pair a positive point with all negative points.

        Args:
            points: The points to preprocess.
            labels: The labels to preprocess.
            scores: The scores to preprocess.

        Returns:
            The preprocessed points (with scores in last dimension) and labels
        """
        # Separate Positive and Negative Points ---
        positive_mask = (labels == 1).squeeze(1)
        negative_mask = (labels == 0).squeeze(1)

        # Get the corresponding coordinates and scores
        positive_coords = points[positive_mask]
        negative_coords = points[negative_mask]
        positive_scores = scores[positive_mask]
        negative_scores = scores[negative_mask]

        # Get the counts
        num_positive = positive_coords.shape[0]
        num_negative = negative_coords.shape[0]

        if num_negative == 0:
            final_point_coords = torch.cat([positive_coords, positive_scores.unsqueeze(-1)], dim=-1)
            return positive_coords, labels

        # Combine each positive point with all negative points
        expanded_negative_coords = negative_coords.squeeze(1).expand(num_positive, -1, -1)
        expanded_negative_scores = negative_scores.squeeze(1).unsqueeze(-1).expand(num_positive, -1, -1)

        # Concatenate the positive coordinates with the expanded negative coordinates
        final_point_coords_2d = torch.cat([positive_coords, expanded_negative_coords], dim=1)

        # Expand positive scores to match the dimension of expanded_negative_scores
        # positive_scores shape: [num_positive, 1] -> unsqueeze -> [num_positive, 1, 1]
        expanded_positive_scores = positive_scores.unsqueeze(1)

        # Concatenate the positive scores with the expanded negative scores
        final_point_scores = torch.cat([expanded_positive_scores, expanded_negative_scores], dim=1)
        final_point_coords = torch.cat([final_point_coords_2d, final_point_scores], dim=-1)

        positive_label = torch.tensor([1], device=points.device, dtype=torch.float32)
        negative_labels = torch.zeros(num_negative, device=points.device, dtype=torch.float32)
        single_group_labels = torch.cat([positive_label, negative_labels])
        final_point_labels = single_group_labels.expand(num_positive, -1)

        return final_point_coords, final_point_labels

    @staticmethod
    def remap_preprocessed_points(preprocessed_points: torch.Tensor) -> torch.Tensor:
        """Remap preprocessed points from grouped format to flat format.

        Args:
            preprocessed_points: Tensor of shape [num_positive_points, 1_positive + N_negative, 3]
                                where last dimension is [x, y, score]

        Returns:
            Tensor of shape [num_positive_points + num_negative_points, 4]
            where last dimension is [x, y, score, label]
        """
        num_pos, total_num, last_dim = preprocessed_points.shape
        num_neg = total_num - 1
        remapped_points = preprocessed_points.new_zeros(
            num_pos + num_neg,
            last_dim + 1,  # last dimension is [x, y, score, label]
        )
        positive_points = preprocessed_points[:, 0]
        positive_labels = torch.ones(num_pos, device=preprocessed_points.device, dtype=torch.float32).unsqueeze(-1)
        positive_points = torch.cat([positive_points, positive_labels], dim=-1)

        negative_points = preprocessed_points[0, 1:]
        negative_labels = torch.zeros(num_neg, device=preprocessed_points.device, dtype=torch.float32).unsqueeze(-1)
        negative_points = torch.cat([negative_points, negative_labels], dim=-1)

        remapped_points[:num_pos, :] = positive_points
        remapped_points[num_pos:, :] = negative_points
        return remapped_points

    def predict_single(
        self,
        class_points: dict[int, torch.Tensor],
        class_boxes: dict[int, torch.Tensor],
        labels: list[int],
        similarities: Similarities | None = None,
        original_size: tuple[int, int] | None = None,
    ) -> tuple[Masks, Points]:
        """Predict masks from a list of points.

        Args:
            class_points: The points to predict masks from.
            class_boxes: The boxes to predict masks from.
            labels: The labels to predict masks from.
            similarities: The class-specific similaritie maps to predict masks from.
            original_size: The original size of the image.
        """
        all_masks = Masks()
        all_used_points = Points()
        all_used_boxes = Boxes()

        similarity_maps = (
            [[] for _ in labels] if similarities is None else [similarities.data[label] for label in labels]
        )
        class_points_list = [class_points.get(label) for label in labels]
        class_boxes_list = [class_boxes.get(label) for label in labels]

        for label, points_per_class, boxes_per_class, similarity_map in zip(
            labels, class_points_list, class_boxes_list, similarity_maps, strict=True
        ):
            final_masks, final_points, final_boxes = self.predict(
                points_per_class=points_per_class,  # Extract only x, y coordinates for SAM predictor
                boxes_per_class=boxes_per_class,
                similarity_map=similarity_map,
                original_size=original_size,
            )

            if len(final_masks):
                for final_mask in final_masks:
                    all_masks.add(final_mask, label)

                # Apply inverse coordinate transformation only to x, y coordinates
                if final_points is not None and len(final_points) > 0:
                    final_points[:, :2] = self.transform.apply_inverse_coords_torch(
                        final_points[:, :2],  # Just the x, y coordinates
                        original_size,
                    )
                    # Remap from [total_points, 3] to [total_points, 4] where last dim is [x, y, score, label]
                    remapped_points = self.remap_preprocessed_points(final_points)
                    all_used_points.add(remapped_points, label)

                if final_boxes is not None and len(final_boxes) > 0:
                    final_boxes[:, :4] = self.transform.apply_inverse_coords_torch(
                        final_boxes[:, :4],
                        original_size,
                    )
                    all_used_boxes.add(final_boxes, label)
            else:
                # TODO(Eugene): This part feels inconsistent.
                # It only adds empty points, but not empty masks or boxes.
                # As a result, len(all_masks), len(all_used_points), and len(all_used_boxes) end up mismatched.
                # Returning variables with inconsistent lengths is undesirable.
                # https://github.com/open-edge-platform/geti-prompt/issues/174
                all_used_points.add(torch.tensor([]), label)

        return all_masks, all_used_points, all_used_boxes

    def predict(
        self,
        points_per_class: torch.Tensor,
        boxes_per_class: torch.Tensor,
        similarity_map: torch.Tensor,
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict masks using SAMPredictor.

        Args:
            points_per_class: The points to predict masks from.
            boxes_per_class: The boxes to predict masks from.
            similarity_map: The similarity map to predict masks from.
            original_size: The original size of the image.
        """
        input_coords, input_labels, input_boxes = None, None, None
        if points_per_class is not None:
            if (points_per_class[:, 3] == 1).any():
                point_coords = points_per_class[:, :2].unsqueeze(1)
                point_scores = points_per_class[:, 2].unsqueeze(1)
                point_labels = points_per_class[:, 3].unsqueeze(1)
                input_coords, input_labels = self.point_preprocess(point_coords, point_labels, point_scores)
            else:
                return (
                    torch.empty((0, *original_size)),
                    torch.empty((0, 1, 3)),
                    torch.empty((0, 6)),
                )

        if boxes_per_class is not None:
            input_boxes = boxes_per_class

        masks, _, low_res_logits = self.predictor.predict_torch(
            point_coords=input_coords[:, :, :2] if input_coords is not None else None,
            boxes=input_boxes[:, :4] if input_boxes is not None else None,
            point_labels=input_labels,
            multimask_output=True,
        )

        # Only refine masks if points are used
        if input_coords is not None:
            final_masks, input_coords = self.mask_refinement(
                masks=masks,
                low_res_logits=low_res_logits,
                input_coords=input_coords,
                input_labels=input_labels,
                similarity_map=similarity_map,
                original_size=original_size,
                score_threshold=self.mask_similarity_threshold,
                nms_iou_threshold=self.nms_iou_threshold,
            )
        else:
            final_masks = masks.squeeze(1)

        return (
            final_masks,
            input_coords,
            input_boxes,
        )

    def mask_refinement(
        self,
        masks: torch.Tensor,
        low_res_logits: torch.Tensor,
        input_coords: torch.Tensor,
        input_labels: torch.Tensor,
        original_size: tuple[int, int],
        similarity_map: torch.Tensor | None = None,
        score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Refine the masks.

        Args:
            masks: The masks to refine.
            low_res_logits: The low-res logits to refine.
            input_coords: The point coordinates to refine.
            input_labels: The point labels to refine.
            original_size: The original size of the image.
            similarity_map: The similarity map to postprocess.
            score_threshold: The score threshold to postprocess.
            nms_iou_threshold: The IoU threshold for the NMS.

        Returns:
            The postprocessed masks, point coordinates.
        """
        keep = masks.squeeze(1).sum(dim=(-1, -2)) > 0
        if not keep.any():
            return (
                masks.new_zeros((0, *masks.shape[-2:])),
                input_coords.new_zeros(0),
            )

        masks = masks[keep]
        low_res_logits = low_res_logits[keep]
        input_coords = input_coords[keep]
        input_labels = input_labels[keep]

        # refine masks with boxes
        boxes = masks_to_boxes(masks.squeeze(1))
        boxes = self.transform.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        boxes = boxes.reshape(-1, 4)
        boxes = boxes.unsqueeze(1)

        masks, mask_weights, _ = self.predictor.predict_torch(
            point_coords=input_coords[:, :, :2],  # Extract only x, y coordinates for SAM predictor
            point_labels=input_labels,
            boxes=boxes,
            mask_input=low_res_logits,
        )

        # nms the masks
        nms_indices = nms(
            boxes.squeeze(1),
            mask_weights.squeeze(1),
            iou_threshold=nms_iou_threshold,
        )

        masks = masks[nms_indices]
        mask_weights = mask_weights[nms_indices]
        low_res_logits = low_res_logits[nms_indices]
        boxes = boxes[nms_indices]
        input_coords = input_coords[nms_indices]
        input_labels = input_labels[nms_indices]

        masks = masks.squeeze(1)
        # TODO(Eugene): in GroundedDINO similarity map is None, in PerDino it's an emppty list
        # Refactor this to use a more consistent approach.
        # https://github.com/open-edge-platform/geti-prompt/issues/174
        if similarity_map is None or len(similarity_map) == 0:
            return masks, input_coords

        mask_sum = (similarity_map[0] * masks).sum(dim=(1, 2))
        mask_area = masks.sum(dim=(1, 2))
        mask_scores = mask_sum / (mask_area + 1e-6)
        weighted_scores = (mask_scores * mask_weights.T).squeeze(0)
        keep = weighted_scores > score_threshold
        if not keep.any():
            return (
                masks.new_zeros((0, *masks.shape[-2:])),
                input_coords.new_zeros(0),
            )

        return masks[keep], input_coords[keep]
