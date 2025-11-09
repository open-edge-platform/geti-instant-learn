# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Results type for Geti Prompt."""

from getiprompt.types.boxes import Boxes
from getiprompt.types.masks import Masks
from getiprompt.types.points import Points
from getiprompt.types.similarities import Similarities


class Results:
    """The class containing results from models."""

    def __init__(self) -> None:
        """Initializes the Results."""
        self._masks: list[Masks] | None = None
        self._used_points: list[Points] | None = None
        self._used_boxes: list[Boxes] | None = None
        self._similarities: list[Similarities] | None = None
        self._duration: float | None = None
        self._box_prompts: list[Boxes] | None = None
        self._point_prompts: list[Points] | None = None

    @property
    def masks(self) -> list[Masks]:
        """Returns masks produced by the latest run of model."""
        return self._masks if self._masks is not None else []

    @masks.setter
    def masks(self, masks: list[Masks]) -> None:
        """Sets the masks."""
        self._masks = masks

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

    @property
    def box_prompts(self) -> list[Boxes]:
        """Returns box prompts produced by the latest run of model."""
        return self._box_prompts if self._box_prompts is not None else []

    @box_prompts.setter
    def box_prompts(self, box_prompts: list[Boxes]) -> None:
        """Sets the box prompts."""
        self._box_prompts = box_prompts

    @property
    def point_prompts(self) -> list[Points]:
        """Returns point prompts produced by the latest run of model."""
        return self._point_prompts if self._point_prompts is not None else []

    @point_prompts.setter
    def point_prompts(self, point_prompts: list[Points]) -> None:
        """Sets the point prompts."""
        self._point_prompts = point_prompts
