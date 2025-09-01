# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from getiprompt.types.boxes import Boxes
from getiprompt.types.data import Data
from getiprompt.types.masks import Masks
from getiprompt.types.points import Points
from getiprompt.types.prompts import Prompt
from getiprompt.types.text import Text


class Priors(Data):
    """This class represents priors for a single image.

    These can contain points, boxes, masks or polygons. They mainly serve as input for Segmentation models.
    """

    def __init__(
        self,
        points: Points | None = None,
        boxes: Boxes | None = None,
        masks: Masks | None = None,
        polygons: Prompt | None = None,
        text: Text | None = None,
    ) -> None:
        self._points: Points = points if points is not None else Points()
        self._boxes: Boxes = boxes if boxes is not None else Boxes()
        self._masks: Masks = masks if masks is not None else Masks()
        self._polygons: Prompt = polygons if polygons is not None else Prompt()
        self._text: Text = text if text is not None else Text()

    @property
    def points(self) -> Points:
        """Get the points."""
        return self._points

    @property
    def boxes(self) -> Boxes:
        """Get the boxes."""
        return self._boxes

    @boxes.setter
    def boxes(self, boxes: Boxes) -> None:
        """Set the boxes."""
        self._boxes = boxes

    @points.setter
    def points(self, points: Points) -> None:
        """Set the points."""
        self._points = points

    @property
    def masks(self) -> Masks:
        """Get the masks."""
        return self._masks

    @masks.setter
    def masks(self, masks: Masks) -> None:
        """Set the masks."""
        self._masks = masks

    @property
    def polygons(self) -> Prompt:
        """Get the polygons."""
        return self._polygons

    @property
    def text(self) -> Text:
        """Get the text."""
        return self._text
