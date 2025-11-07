# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
class PipelineManagerError(Exception):
    """Base exception for PipelineManager errors."""


class PipelineNotActiveError(PipelineManagerError):
    """Exception raised when there is no active pipeline to register to."""


class PipelineProjectMismatchError(PipelineManagerError):
    """Exception raised when the project ID does not match the active pipeline's project ID."""
