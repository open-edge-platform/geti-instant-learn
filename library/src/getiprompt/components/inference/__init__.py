# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference components.

Note: ExportableSAMPredictor has moved to getiprompt.components.sam.exportable
This import is maintained for backward compatibility.
"""

# Backward compatibility - re-export from new location
from getiprompt.components.sam import ExportableSAMPredictor

__all__ = ["ExportableSAMPredictor"]
