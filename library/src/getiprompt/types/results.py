# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from getiprompt.types.annotations import Annotations
from getiprompt.types.boxes import Boxes
from getiprompt.types.masks import Masks
from getiprompt.types.points import Points
from getiprompt.types.priors import Priors
from getiprompt.types.similarities import Similarities


class Results:
    """The class containing results from models."""

    def __init__(self) -> None:
        self._masks: list[Masks] | None = None
        self._priors: list[Priors] | None = None
        self._annotations: list[Annotations] | None = None
        self._used_points: list[Points] | None = None
        self._used_boxes: list[Boxes] | None = None
        self._similarities: list[Similarities] | None = None
        self._duration: float | None = None

    @property
    def masks(self) -> list[Masks]:
        """Returns masks produced by the latest run of model."""
        return self._masks if self._masks is not None else []

    @masks.setter
    def masks(self, masks: list[Masks]) -> None:
        """Sets the masks."""
        self._masks = masks

    @property
    def priors(self) -> list[Priors]:
        """Returns priors produced by the latest run of model."""
        return self._priors if self._priors is not None else []

    @priors.setter
    def priors(self, priors: list[Priors]) -> None:
        """Sets the priors."""
        self._priors = priors

    @property
    def used_points(self) -> list[Points]:
        """Returns used points produced by the latest run of model."""
        return self._used_points if self._used_points is not None else []

    @used_points.setter
    def used_points(self, used_points: list[Points]) -> None:
        """Sets the used points."""
        self._used_points = used_points

    @property
    def used_boxes(self) -> list[Boxes]:
        """Returns used boxes produced by the latest run of model."""
        return self._used_boxes if self._used_boxes is not None else []

    @used_boxes.setter
    def used_boxes(self, used_boxes: list[Boxes]) -> None:
        """Sets the used boxes."""
        self._used_boxes = used_boxes

    @property
    def annotations(self) -> list[Annotations]:
        """Returns annotations produced by the latest run of model."""
        return self._annotations if self._annotations is not None else []

    @annotations.setter
    def annotations(self, annotations: list[Annotations]) -> None:
        """Sets the annotations."""
        self._annotations = annotations

    @property
    def similarities(self) -> list[Similarities]:
        """Returns similarities produced by the latest run of model."""
        return self._similarities if self._similarities is not None else []

    @similarities.setter
    def similarities(self, similarities: list[Similarities]) -> None:
        """Sets the similarities."""
        self._similarities = similarities

    @property
    def duration(self) -> float:
        """Returns the duration of the latest run of model."""
        return self._duration if self._duration is not None else 0.0

    @duration.setter
    def duration(self, duration: float) -> None:
        """Sets the duration."""
        self._duration = duration
