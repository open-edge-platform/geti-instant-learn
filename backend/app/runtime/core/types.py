#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from typing import Any, TypeVar

# Generic type variables for pipeline data flow
IN = TypeVar("IN")
OUT = TypeVar("OUT")

ConfigDict = dict[str, Any]
