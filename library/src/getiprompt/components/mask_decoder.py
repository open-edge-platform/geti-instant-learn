# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""SAM decoder."""

from logging import getLogger

import torch
from torch import nn
from torchvision import tv_tensors
from torchvision.ops import masks_to_boxes, nms

from getiprompt.components.sam import OpenVINOSAMPredictor, PyTorchSAMPredictor
from getiprompt.data import ResizeLongestSide

logger = getLogger("Geti Prompt")


class SamDecoder(nn.Module):
    """This Segmenter uses SAM to create masks based on points.

    Examples:
        >>> from getiprompt.models.models import load_sam_model
        >>> from getiprompt.processes.segmenters import SamDecoder
        >>> from getiprompt.types import Masks, Points, Priors
        >>> from torchvision import tv_tensors
        >>> import torch
        >>> import numpy as np
        >>> sam_predictor = load_sam_model(backbone_name="MobileSAM")
        >>> segmenter = SamDecoder(sam_predictor=sam_predictor)
        >>> image = tv_tensors.Image(np.zeros((3, 1024, 1024), dtype=np.uint8))
        >>> point_prompts = {1: torch.tensor([[512, 512, 0.9, 1], [100, 100, 0.8, 0]])} # fg, bg
        >>> points = torch.tensor([[512, 512, 0.9, 1], [100, 100, 0.8, 0]]) # fg, bg
        >>> similarities = {}
        >>> similarities[1] = torch.ones(1, 1024, 1024)
        >>> masks, point_prompts_used, box_prompts_used = segmenter(
        ...     images=[image],
        ...     point_prompts=[point_prompts],
        ...     similarities=[similarities],
        ... )
    """

    def __init__(
        self,
        sam_predictor: PyTorchSAMPredictor | OpenVINOSAMPredictor,
        target_length: int = 1024,
        mask_similarity_threshold: float = 0.38,
        nms_iou_threshold: float = 0.1,
        use_mask_refinement: bool = False,
    ) -> None:
        """This Segmenter uses SAM to create masks based on points.

        Args:
            sam_predictor: The SAM predictor (any backend: PyTorch, OpenVINO, etc.).
            target_length: Target length for the longest side of the image (for image transform).
            mask_similarity_threshold: The similarity threshold for the mask.
            nms_iou_threshold: The IoU threshold for the NMS.
            use_mask_refinement: Whether to use 2-stage mask refinement. Defaults to False for better FPS.
        """
        super().__init__()
        self.predictor = sam_predictor
        self.mask_similarity_threshold = mask_similarity_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.use_mask_refinement = use_mask_refinement

        self.transform = ResizeLongestSide(target_length)
        self.device = sam_predictor.device

    def preprocess_inputs(
        self,
        images: list[tv_tensors.Image],
        point_prompts: list[dict[int, torch.Tensor]] | None = None,
        box_prompts: list[dict[int, torch.Tensor]] | None = None,
    ) -> tuple[list[torch.Tensor], list[int], list[tuple[int, int]]]:
        """Preprocess the inputs.

        Args:
            images(list[tv_tensors.Image]): The images to preprocess.
            point_prompts(list[dict[int, torch.Tensor]]): The point prompts to preprocess.
            box_prompts(list[dict[int, torch.Tensor]]): The box prompts to preprocess.

        Returns:
            A tuple of preprocessed images, unique labels, and original sizes.

        """
        preprocessed_images = []
        original_sizes = []
        for image, class_point_prompts, class_box_prompts in zip(images, point_prompts, box_prompts, strict=True):
            # Preprocess image using SamPredictor transform
            input_image = self.transform.apply_image_torch(image).to(self.device)
            ori_size = image.shape[-2:]
            preprocessed_images.append(input_image)
            original_sizes.append(ori_size)

            # Preprocess points for each class in place
            for class_id, points in class_point_prompts.items():
                transformed_coords = self.transform.apply_coords_torch(points[:, :2], ori_size)
                transformed_points = points.clone()
                transformed_points[:, :2] = transformed_coords
                class_point_prompts[class_id] = transformed_points

            # Preprocess boxes for each class
            for class_id, boxes in class_box_prompts.items():
                box_coords = boxes[:, :4]
                transformed_boxes = self.transform.apply_boxes_torch(box_coords, ori_size)
                transformed_boxes = torch.cat([transformed_boxes, boxes[:, 4:]], dim=1)
                class_box_prompts[class_id] = transformed_boxes

        # find unique labels
        unique_labels = sorted(set(class_point_prompts.keys()) | set(class_box_prompts.keys()))

        return preprocessed_images, unique_labels, original_sizes

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
        negative_mask = (labels == -1).squeeze(1)

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
            return final_point_coords, labels

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
        negative_labels = -torch.ones(num_negative, device=points.device, dtype=torch.float32)
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
        negative_labels = -torch.ones(num_neg, device=preprocessed_points.device, dtype=torch.float32).unsqueeze(-1)
        negative_points = torch.cat([negative_points, negative_labels], dim=-1)

        remapped_points[:num_pos, :] = positive_points
        remapped_points[num_pos:, :] = negative_points
        return remapped_points

    def predict_single(
        self,
        class_points: dict[int, torch.Tensor],
        class_boxes: dict[int, torch.Tensor],
        labels: list[int],
        similarities: dict[int, torch.Tensor],
        original_size: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        """Predict masks from a list of points and boxes.

        Args:
            class_points: The points to predict masks from.
            class_boxes: The boxes to predict masks from.
            labels: The labels to predict masks from.
            similarities: The class-specific similaritie maps to predict masks from.
            original_size: The original size of the image.

        Returns:
            A dictionary of predictions:
                "pred_masks": torch.Tensor of shape [num_masks, H, W]
                "pred_points": torch.Tensor of shape [num_points, 4] with last dimension [x, y, score, fg_label]
                "pred_boxes": torch.Tensor of shape [num_boxes, 5] with last dimension [x1, y1, x2, y2, score]
                "pred_labels": torch.Tensor of shape [num_masks]
        """
        prediction = {
            "pred_masks": torch.empty((0, *original_size)).to(self.device),
            "pred_points": torch.empty((0, 4)).to(self.device),
            "pred_boxes": torch.empty((0, 5)).to(self.device),
            "pred_labels": torch.empty((0,), dtype=torch.long).to(self.device),
        }

        similarity_maps = [similarities[label] for label in labels] if len(similarities) else [[] for _ in labels]
        class_points_list = [class_points.get(label) for label in labels]
        class_boxes_list = [class_boxes.get(label) for label in labels]

        for label, points_per_class, boxes_per_class, similarity_map in zip(
            labels,
            class_points_list,
            class_boxes_list,
            similarity_maps,
            strict=True,
        ):
            final_masks, final_points, final_boxes = self.predict(
                points_per_class=points_per_class,  # Extract only x, y coordinates for SAM predictor
                boxes_per_class=boxes_per_class,
                similarity_map=similarity_map,
                original_size=original_size,
            )

            if len(final_masks):
                # Apply inverse coordinate transformation only to x, y coordinates
                if final_points is not None and len(final_points) > 0:
                    # Remap from [total_points, 3] to [total_points, 4] where last dim is [x, y, score, label]
                    remapped_points = self.remap_preprocessed_points(final_points)
                    remapped_points[:, :2] = self.transform.apply_inverse_coords_torch(
                        remapped_points[:, :2],
                        original_size,
                    )
                    prediction["pred_points"] = remapped_points

                if final_boxes is not None and len(final_boxes) > 0:
                    final_boxes[:, :4] = self.transform.apply_inverse_boxes(final_boxes[:, :4], original_size)
                    prediction["pred_boxes"] = final_boxes

                prediction["pred_masks"] = final_masks
                prediction["pred_labels"] = torch.full(
                    (len(final_masks),),
                    label,
                    device=self.device,
                    dtype=torch.long,
                )

        return prediction

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

        Returns:
            A tuple of masks, point coordinates, and box coordinates.
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

        masks, mask_weights, low_res_logits = self.predictor.predict(
            point_coords=input_coords[:, :, :2] if input_coords is not None else None,
            boxes=input_boxes[:, :4] if input_boxes is not None else None,
            point_labels=input_labels,
            multimask_output=True,
        )

        # Apply mask refinement (NMS + similarity filtering), optionally with 2nd SAM prediction
        if input_coords is not None:
            final_masks, input_coords = self.mask_refinement(
                masks=masks,
                mask_weights=mask_weights,
                low_res_logits=low_res_logits,
                input_coords=input_coords,
                input_labels=input_labels,
                similarity_map=similarity_map,
                original_size=original_size,
                score_threshold=self.mask_similarity_threshold,
                nms_iou_threshold=self.nms_iou_threshold,
                use_box_refinement=self.use_mask_refinement,
            )
        else:
            final_masks = masks.squeeze(1)

        final_masks = (final_masks.sum(0) > 0).unsqueeze(0)
        return (
            final_masks,
            input_coords,
            input_boxes,
        )

    def mask_refinement(
        self,
        masks: torch.Tensor,
        mask_weights: torch.Tensor,
        low_res_logits: torch.Tensor,
        input_coords: torch.Tensor,
        input_labels: torch.Tensor,
        original_size: tuple[int, int],
        similarity_map: torch.Tensor | None = None,
        score_threshold: float = 0.45,
        nms_iou_threshold: float = 0.1,
        use_box_refinement: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Refine the masks.

        Args:
            masks: The masks to refine.
            mask_weights: The mask weights/scores from SAM prediction.
            low_res_logits: The low-res logits to refine.
            input_coords: The point coordinates to refine.
            input_labels: The point labels to refine.
            original_size: The original size of the image.
            similarity_map: The similarity map to postprocess.
            score_threshold: The score threshold to postprocess.
            nms_iou_threshold: The IoU threshold for the NMS.
            use_box_refinement: Whether to use 2nd SAM prediction with box prompts.

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
        mask_weights = mask_weights[keep]
        low_res_logits = low_res_logits[keep]
        input_coords = input_coords[keep]
        input_labels = input_labels[keep]

        # Compute boxes from masks for NMS (and optionally for 2nd SAM prediction)
        boxes = masks_to_boxes(masks.squeeze(1))
        boxes = self.transform.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        boxes = boxes.reshape(-1, 4)
        boxes = boxes.unsqueeze(1)

        # Optionally refine masks with 2nd SAM prediction using box prompts
        if use_box_refinement:
            masks, mask_weights, _ = self.predictor.predict(
                point_coords=input_coords[:, :, :2],  # Extract only x, y coordinates for SAM predictor
                point_labels=input_labels,
                boxes=boxes,
                mask_input=low_res_logits,
            )

        # NOTE: torchvision NMS requires float32 inputs
        nms_indices = nms(
            boxes.squeeze(1).to(torch.float32),
            mask_weights.squeeze(1).to(torch.float32),
            iou_threshold=nms_iou_threshold,
        )

        masks = masks[nms_indices]
        mask_weights = mask_weights[nms_indices]
        low_res_logits = low_res_logits[nms_indices]
        boxes = boxes[nms_indices]
        input_coords = input_coords[nms_indices]
        input_labels = input_labels[nms_indices]

        masks = masks.squeeze(1)
        if len(similarity_map) == 0:
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

    @torch.inference_mode()
    def forward(
        self,
        images: list[tv_tensors.Image],
        point_prompts: list[dict[int, torch.Tensor]] | None = None,
        box_prompts: list[dict[int, torch.Tensor]] | None = None,
        similarities: list[dict[int, torch.Tensor]] | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            images(list[tv_tensors.Image]): The images to predict masks from.
            point_prompts(list[dict[int, torch.Tensor]]): The point prompts to predict masks from.
            box_prompts(list[dict[int, torch.Tensor]]): The box prompts to predict masks from.
            similarities(list[dict[int, torch.Tensor]]): The similarities to predict masks from.

        Returns:
            predictions(list[dict[str, torch.Tensor | None]]): The predictions per image.
                Each element in the list is a dictionary containing the following keys:
                    "pred_masks": torch.Tensor of shape [num_masks, H, W]
                    "pred_points": torch.Tensor of shape [num_points, 4], [x, y, score, fg_label]
                    "pred_boxes": torch.Tensor of shape [num_boxes, 5], [x1, y1, x2, y2, score]
                    "pred_labels": torch.Tensor of shape [num_masks]
        """
        predictions: list[dict[str, torch.Tensor | None]] = []

        # default to empty lists if not provided
        if similarities is None:
            similarities = [{}] * len(images)
        if box_prompts is None:
            box_prompts = [{}] * len(images)
        if point_prompts is None:
            point_prompts = [{}] * len(images)

        images, labels, original_sizes = self.preprocess_inputs(images, point_prompts, box_prompts)

        for (
            image,
            class_point_prompts,
            class_box_prompts,
            similarities_per_image,
            original_size,
        ) in zip(
            images,
            point_prompts,
            box_prompts,
            similarities,
            original_sizes,
            strict=True,
        ):
            # Set the preprocessed image in the predictor
            self.predictor.set_image(image, original_size)
            prediction = self.predict_single(
                class_point_prompts,
                class_box_prompts,
                labels,
                similarities_per_image,
                original_size,
            )
            predictions.append(prediction)
        return predictions
