# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

import rest.endpoints  # noqa: F401, pylint: disable=unused-import  # Importing for endpoint registration
from dependencies import run_db_migrations
from routers import pipelines_router, state_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore # noqa: ARG001, ANN201
    """
    Defines startup and shutdown of the FastAPI app
    """
    # Startup actions
    run_db_migrations()
    yield

    # Shutdown actions


app = FastAPI(lifespan=lifespan)
app.include_router(pipelines_router, prefix="/api/v1")
app.include_router(state_router, prefix="/api/v1")
logger.info("Application has started")
