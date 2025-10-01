#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Any

from core.components.schemas.processor import ModelConfig


class ModelFactory:
    """
    A factory for creating Model instances based on a configuration.
    """

    @classmethod
    def create(cls, config: ModelConfig) -> Any:
        # Any here is a placeholder for a vision prompt model instance
        pass
