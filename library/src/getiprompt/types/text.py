# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Text prompt type for Geti Prompt."""

from difflib import SequenceMatcher

from getiprompt.types.prompts import Prompt


class Text(Prompt):
    """This class represent text prompts for a single image."""

    def __init__(
        self,
        data: dict[int, str] | None = None,
    ) -> None:
        """Initializes the Text prompt."""
        self._data = data if data is not None else {}

    def add(self, data: str, class_id: int = 0) -> None:
        """Adds data for a given class by extending the list."""
        self._data[class_id] = data

    def get(self, class_id: int = 0) -> str:
        """Get the data for a given class."""
        if class_id in self._data:
            return self._data[class_id]
        return ""

    def find(self, text: str) -> int:
        """Find the class_id associated with text. Throws exception if not found."""
        return int(next(key for key, value in self._data.items() if value == text))

    @staticmethod
    def match(a: str, b: str) -> float:
        """Returns the similarity between two strings."""
        return SequenceMatcher(None, a, b).ratio()

    def find_best(self, text: str) -> int:
        """Find the class_id associated with text. Always returns the closest match."""
        sims = {round(self.match(value, text), 10): cid for cid, value in self._data.items()}
        best_match = max(sims.keys())
        return sims[best_match]
