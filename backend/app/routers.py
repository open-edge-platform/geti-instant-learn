# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

pipelines_router = APIRouter(prefix="/pipelines", tags=["Pipelines"])
state_router = APIRouter(prefix="/state", tags=["State"])
