#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from runtime.core.components.base import Processor
from runtime.core.components.schemas.processor import InputData, OutputData


class NoOpProcessor(Processor[InputData, OutputData]):
    """
    A 'no-operation' implementation of the Processor interface.

    This processor bypasses any actual processing and simply repackages the
    input frame into the output data structure. It serves as a default placeholder
    in a pipeline when no specific algorithm has been configured by the user.
    """

    def process(self, input_data: InputData) -> OutputData:
        return OutputData(frame=input_data.frame)
