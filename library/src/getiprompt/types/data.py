# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import BinaryIO


class Data:
    """This is a base class for all data types.

    It provides a way to save and load the data to and from a file.
    """

    def __init__(self) -> None:
        """Initialize the data."""
        self._data = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the shape of the data."""
        return self._data.shape

    @property
    def data(self) -> None:
        """Get the data."""
        return self._data

    def save(self, f: BinaryIO) -> None:
        """Save the data to a file."""

    def load(self, f: BinaryIO) -> None:
        """Load the data from a file."""
