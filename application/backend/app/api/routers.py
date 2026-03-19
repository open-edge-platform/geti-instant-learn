# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

projects_router = APIRouter(prefix="/projects")
webrtc_router = APIRouter(prefix="/webrtc")
source_types_router = APIRouter(prefix="/source-types")
license_router = APIRouter(prefix="/license")

supported_models_router = APIRouter(prefix="/supported-models")
