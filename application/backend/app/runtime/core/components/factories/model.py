#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Any

from runtime.core.components.schemas.processor import ModelConfig


class ModelFactory:
    """
    A factory for creating Model instances based on a configuration.
    """

    @classmethod
    def create(cls, config: ModelConfig) -> Any:
        match config:
            case _:
                return None
