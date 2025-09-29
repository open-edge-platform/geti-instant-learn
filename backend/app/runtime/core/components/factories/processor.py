# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from runtime.core.components.base import Processor
from runtime.core.components.schemas.processor import InputData, OutputData, ProcessorConfig


class ProcessorFactory:
    """
    A factory for creating Processor instances based on a configuration.
    """

    @classmethod
    def create(cls, config: ProcessorConfig) -> Processor[InputData, OutputData]:
        pass
