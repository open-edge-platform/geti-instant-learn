# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""InstantLearn data utilities.

This module provides utility functions for data handling in InstantLearn.
"""

from .image import read_image, read_mask

__all__ = ["read_image", "read_mask"]
