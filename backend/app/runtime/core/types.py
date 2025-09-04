#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from typing import TypeVar

from pydantic import BaseModel


# Generic type variables for pipeline data flow

class InputData(BaseModel):
    pass


class OutputData(BaseModel):
    pass


IN = TypeVar("IN")
OUT = TypeVar("OUT")
