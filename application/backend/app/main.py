# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

import api.endpoints  # noqa: F401, pylint: disable=unused-import  # Importing for endpoint registration
from api.error_handler import custom_exception_handler
from api.routers import projects_router
from domain.db.engine import get_session_factory, run_db_migrations
from domain.dispatcher import ConfigChangeDispatcher
from runtime.pipeline_manager import PipelineManager
from runtime.webrtc.manager import WebRTCManager
from settings import get_settings

settings = get_settings()

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup actions
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )
    logger.info(f"Starting {settings.app_name} application...")
    run_db_migrations()

    app.state.config_dispatcher = ConfigChangeDispatcher()
    app.state.pipeline_manager = PipelineManager(
        event_dispatcher=app.state.config_dispatcher, session_factory=get_session_factory()
    )
    app.state.pipeline_manager.start()

    # Initialize WebRTC Manager
    app.state.webrtc_manager = WebRTCManager(pipeline_manager=app.state.pipeline_manager)

    logger.info("Application startup completed")
    yield

    # Shutdown actions
    logger.info(f"Shutting down {settings.app_name} application...")
    app.state.config_dispatcher.shutdown()
    await app.state.webrtc_manager.cleanup()
    app.state.pipeline_manager.stop()


fastapi_app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description=settings.description,
    openapi_url=settings.openapi_url,
    redoc_url=None,
    lifespan=lifespan,
    # TODO add contact info
    # TODO add license
)

fastapi_app.add_exception_handler(Exception, custom_exception_handler)
fastapi_app.add_exception_handler(RequestValidationError, custom_exception_handler)


@fastapi_app.get(path="/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Health check endpoint"""
    return {"status": "ok"}


fastapi_app.include_router(projects_router, prefix="/api/v1")

if (
    settings.static_files_dir
    and os.path.isdir(settings.static_files_dir)
    and next(os.scandir(settings.static_files_dir), None) is not None
):
    fastapi_app.mount(
        os.getenv("ASSET_PREFIX", "/html"), StaticFiles(directory=settings.static_files_dir), name="static"
    )

    @fastapi_app.get("/", include_in_schema=False)
    @fastapi_app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = "") -> FileResponse:  # noqa: ARG001
        """
        Serve the Single Page Application (SPA) index.html file for any path
        """
        index_path = os.path.join(settings.static_files_dir, "index.html")
        return FileResponse(index_path)


app = CORSMiddleware(  # TODO restrict settings in production
    app=fastapi_app,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def main() -> None:
    """Main application entry point"""
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="debug" if settings.debug else "info",
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
