#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from backend.app.runtime.core.base import Processor
from backend.app.runtime.core.types import ConfigDict, IN, OUT, InputData


class NoOpProcessor(Processor[InputData, InputData]):

    def get_config(self) -> ConfigDict:
        return {}

    def process(self, input_data: IN) -> OUT:
        return input_data
