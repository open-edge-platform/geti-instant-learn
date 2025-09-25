# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from runtime.core.components.base import Processor
from runtime.core.components.schemas.processor import ProcessorConfig, InputData, OutputData


class ProcessorFactory:

    @classmethod
    def create(cls, config: ProcessorConfig) -> Processor[InputData, OutputData]:
        pass
