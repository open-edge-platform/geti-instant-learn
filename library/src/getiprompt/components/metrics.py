# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Segmentation metrics calculator."""

from logging import getLogger
from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix
from torch import nn

from getiprompt.types import Masks

logger = getLogger("Geti Prompt")


class SegmentationMetrics(nn.Module):
    """This class handles metrics calculations.

    Examples:
        >>> from getiprompt.processes.calculators import SegmentationMetrics
        >>> from getiprompt.types import Masks
        >>> import torch
        >>>
        >>> calculator = SegmentationMetrics(categories=["car"])
        >>>
        >>> # Create a perfect match scenario for prediction and reference masks.
        >>> mask_tensor = torch.zeros((1, 10, 10), dtype=torch.bool)
        >>> mask_tensor[0, 2:8, 2:8] = True
        >>> reference = Masks()
        >>> reference.add(mask_tensor, class_id=0)
        >>> prediction = Masks()
        >>> prediction.add(mask_tensor.clone(), class_id=0)
        >>>
        >>> calculator(
        ...     predictions=[prediction],
        ...     references=[reference],
        ...     mapping={0: "car"},
        ... )
        >>> metrics = calculator.get_metrics()
        >>>
        >>> # For a perfect match, IoU should be 1.0.
        >>> metrics["category"][0]
        'car'
        >>> round(metrics["iou"][0], 1)
        1.0
    """

    def __init__(self, categories: list[str]) -> None:
        """Ths class handles metrics calculations.

        Args:
            categories: A list of category names that will be used.
                Note: The background should not be present and is added automatically.
        """
        super().__init__()
        self.n_samples = 0
        self.tp_count = self.fp_count = self.fn_count = self.tn_count = 0
        self.confusion: dict[str : np.ndarray] = {}  # category: binary confusion summation
        if categories is not None:
            self.categories = list({"background"}.union(set(categories)))

    def get_metrics(self) -> dict[str, list[Any]]:
        """This method calculates the metrics."""
        d = {
            "category": [],
            "iou": [],
            "f1score": [],
            "precision": [],
            "recall": [],
            "accuracy": [],
            "true_positives": [],
            "true_negatives": [],
            "false_positives": [],
            "false_negatives": [],
            "jaccard": [],
            "dice": [],
        }
        for cat_name, confusion in self.confusion.items():
            if confusion.shape != (2, 2):
                logger.warning(
                    f"Warning: Confusion matrix for {cat_name} is not 2x2 ({confusion.shape}). Skipping metrics."
                )
                continue

            tn = confusion[0, 0]  # True Negative (BG correctly predicted)
            fp = confusion[0, 1]  # False Positive (BG predicted as FG)
            fn = confusion[1, 0]  # False Negative (FG predicted as BG)
            tp = confusion[1, 1]  # True Positive (FG correctly predicted)

            # Add epsilon for numerical stability
            epsilon = 1e-6

            # Correct Metrics Calculations
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
            f1score = 2 * (precision * recall) / (precision + recall + epsilon)

            # Correct Jaccard (IoU)
            union = tp + fp + fn
            jaccard = tp / (union + epsilon)
            iou = jaccard

            # Dice is equivalent to F1
            dice = f1score

            d["category"].append(cat_name)
            d["true_positives"].append(int(tp))
            d["true_negatives"].append(int(tn))
            d["false_positives"].append(int(fp))
            d["false_negatives"].append(int(fn))
            d["precision"].append(precision)
            d["recall"].append(recall)
            d["f1score"].append(f1score)
            d["jaccard"].append(jaccard)
            d["iou"].append(iou)
            d["dice"].append(dice)
            d["accuracy"].append(accuracy)
        return d

    def forward(
        self,
        predictions: list[Masks] | None = None,
        references: list[Masks] | None = None,
        mapping: dict[int, str] | None = None,
    ) -> None:
        """This class compares predicted and reference masks.

        Individual instances are merged into one mask.
        If no mapping is provided then a mapping is created according to indices in self.categories.
        Currently, this implementation supports only binary masks per class.

        Args:
            predictions: List of predicted masks
            references: List of reference masks
            mapping: Dictionary mapping class ids to class names.
        """
        if mapping is None:
            mapping = {}
        if references is None:
            references = []
        if predictions is None:
            predictions = []
        if mapping is None:
            mapping = dict(enumerate(self.categories))
        else:
            # For internal calculations include the background class
            mapping = {idx + 1: name for idx, name in mapping.items()}
            mapping[0] = "background"
        class_ids = sorted(mapping.keys())
        if len(class_ids) > 2:
            msg = "Multiple classes per image not yet supported."
            raise NotImplementedError(msg)
        class_name = mapping[class_ids[-1]]

        # Start metric calculation
        for prediction, reference in zip(predictions, references, strict=False):
            # Create a mask where each pixel value represents the class id
            pred_mask = np.zeros([*reference.mask_shape])  # pred shape can be empty
            ref_mask = np.zeros([*reference.mask_shape])
            for class_id in class_ids:
                if (
                    class_id - 1 not in reference.class_ids() and class_id - 1 not in prediction.class_ids()
                ) or class_id == 0:
                    continue
                if class_id - 1 in reference.class_ids():
                    ref = reference.to_numpy(class_id - 1)
                    if len(ref) > 0:
                        ref_mask[np.max(ref, axis=0) > 0] = class_id

                if class_id - 1 in prediction.class_ids():
                    pred = prediction.to_numpy(class_id - 1)
                    pred_mask[np.max(pred, axis=0) > 0] = class_id

            # Calculate confusion matrix of this image
            conf = confusion_matrix(
                y_true=ref_mask.flatten(),
                y_pred=pred_mask.flatten(),
                labels=class_ids,
            )
            if class_name in self.confusion:
                self.confusion[class_name] = self.confusion[class_name] + conf
            else:
                self.confusion[class_name] = conf
