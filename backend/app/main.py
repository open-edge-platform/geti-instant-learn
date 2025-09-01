# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from fastapi import FastAPI

import rest.endpoints  # noqa: F401, pylint: disable=unused-import  # Importing for endpoint registration
from routers import pipelines_router, state_router

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(pipelines_router, prefix="/api/v1")
app.include_router(state_router, prefix="/api/v1")
logger.info("Application has started")
