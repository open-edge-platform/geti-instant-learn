#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from backend.app.runtime.core.base import Processor
from backend.app.runtime.schemas.pipeline import InputData, OutputData


class NoOpProcessor(Processor[InputData, OutputData]):

    def process(self, input_data: InputData) -> OutputData:
        return OutputData(frame=input_data.frame)
