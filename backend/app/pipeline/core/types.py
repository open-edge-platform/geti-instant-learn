#  Copyright (C) 2022-2025 Intel Corporation
#  LIMITED EDGE SOFTWARE DISTRIBUTION LICENSE

from typing import Any, TypeVar

# Generic type variables for pipeline data flow
IN = TypeVar("IN")
OUT = TypeVar("OUT")

ConfigDict = dict[str, Any]
