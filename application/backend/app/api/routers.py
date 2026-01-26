# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from fastapi import APIRouter

projects_router = APIRouter(prefix="/projects")
webrtc_router = APIRouter(prefix="/webrtc")
source_types_router = APIRouter(prefix="/source-types")
