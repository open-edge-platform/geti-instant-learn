#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from runtime.core.components.base import Processor
from runtime.core.components.schemas.processor import InputData, OutputData


class NoOpProcessor(Processor[InputData, OutputData]):

    def process(self, input_data: InputData) -> OutputData:
        return OutputData(frame=input_data.frame)
