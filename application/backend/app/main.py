# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import rest.endpoints  # noqa: F401, pylint: disable=unused-import  # Importing for endpoint registration
from dependencies import ensure_default_active_project, run_db_migrations
from routers import projects_router
from settings import get_settings

settings = get_settings()

logging.basicConfig(
    level=logging.INFO if not settings.debug else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:  # noqa: ARG001
    """FastAPI lifespan context manager"""
    # Startup actions
    logger.info(f"Starting {settings.app_name} application...")
    run_db_migrations(settings)

    # TODO remove later, we'll require explicit project creation via a UI form when no projects exist
    ensure_default_active_project()

    logger.info("Application startup completed")
    yield

    # Shutdown actions
    logger.info(f"Shutting down {settings.app_name} application...")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    openapi_url=settings.openapi_url,
    redoc_url=None,
    lifespan=lifespan,
    # TODO add contact info
    # TODO add license
)

app.add_middleware(  # TODO restrict settings in production
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(projects_router, prefix="/api/v1")

if (
    settings.static_files_dir
    and os.path.isdir(settings.static_files_dir)
    and next(os.scandir(settings.static_files_dir), None) is not None
):
    app.mount("/html", StaticFiles(directory=settings.static_files_dir), name="static")

    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = "") -> FileResponse:  # noqa: ARG001
        """
        Serve the Single Page Application (SPA) index.html file for any path
        """
        index_path = os.path.join(settings.static_files_dir, "index.html")
        return FileResponse(index_path)


def main() -> None:
    """Main application entry point"""
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
