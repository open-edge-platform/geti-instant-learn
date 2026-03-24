# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import Response

import api.endpoints  # noqa: F401, pylint: disable=unused-import  # Importing for endpoint registration
from api.error_handler import custom_exception_handler
from api.routers import (
    license_router,
    projects_router,
    source_types_router,
    supported_models_router,
    system_router,
    webrtc_router,
)
from dependencies import LicenseServiceDep
from domain.db.engine import get_session_factory, run_db_migrations
from domain.dispatcher import ConfigChangeDispatcher
from domain.services.schemas.base import Pagination
from domain.services.schemas.dataset import DatasetsListSchema
from domain.services.schemas.health import HealthCheckSchema, HealthStatus
from runtime.errors import DatasetNotFoundError
from runtime.pipeline_manager import PipelineManager
from runtime.services.dataset_discovery import scan_datasets
from runtime.webrtc.manager import WebRTCManager
from runtime.webrtc.sdp_handler import SDPHandler
from settings import get_settings

settings = get_settings()
settings.logs_dir.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """FastAPI lifespan context manager"""
    # Startup actions
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(filename=settings.log_file, encoding="utf8")
    logging.basicConfig(
        handlers=[console_handler, file_handler],
        level=getattr(logging, settings.log_level.upper()),
        format=settings.log_format,
        force=True,
    )
    logger.info(f"Starting {settings.app_name} application...")
    logger.info(settings.format_for_logging())
    run_db_migrations()

    app.state.config_dispatcher = ConfigChangeDispatcher()
    app.state.pipeline_manager = PipelineManager(
        event_dispatcher=app.state.config_dispatcher, session_factory=get_session_factory()
    )
    app.state.pipeline_manager.start()

    # Initialize WebRTC Manager
    app.state.sdp_handler = SDPHandler()
    app.state.webrtc_manager = WebRTCManager(
        pipeline_manager=app.state.pipeline_manager, sdp_handler=app.state.sdp_handler
    )

    # Dataset cache is startup-static by design and refreshed only on app restart.
    datasets = DatasetsListSchema(
        datasets=[],
        pagination=Pagination(count=0, total=0, offset=0, limit=0),
    )
    dataset_paths: dict = {}
    try:
        datasets, dataset_paths = scan_datasets(settings.template_dataset_dir)
    except DatasetNotFoundError as e:
        logger.error("Dataset discovery failed during startup: %s", str(e))

    app.state.available_datasets = datasets
    app.state.dataset_paths = dataset_paths

    if dataset_paths:
        dataset_lines = "\n".join(
            f"- {dataset.name}: {dataset.id} -> {dataset_paths[dataset.id]}" for dataset in datasets.datasets
        )
        logger.debug("Dataset cache entries:\n%s", dataset_lines)
    logger.info("Cached %d dataset(s) during startup", len(dataset_paths))

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
def health_check(license_service: LicenseServiceDep) -> HealthCheckSchema:
    """Health check endpoint"""
    return HealthCheckSchema(status=HealthStatus.OK, license_accepted=license_service.is_accepted())


fastapi_app.include_router(projects_router, prefix="/api/v1")
fastapi_app.include_router(source_types_router, prefix="/api/v1")
fastapi_app.include_router(webrtc_router, prefix="/api/v1")
fastapi_app.include_router(license_router, prefix="/api/v1")
fastapi_app.include_router(system_router, prefix="/api/v1")
fastapi_app.include_router(supported_models_router, prefix="/api/v1")


if (
    settings.static_files_dir
    and os.path.isdir(settings.static_files_dir)
    and next(os.scandir(settings.static_files_dir), None) is not None
):
    asset_prefix = os.getenv("ASSET_PREFIX", "/html")
    logger.info("Serving static files from %s by context %s", settings.static_files_dir, asset_prefix)
    fastapi_app.mount(asset_prefix, StaticFiles(directory=settings.static_files_dir), name="static")

    @fastapi_app.get("/", include_in_schema=False)
    @fastapi_app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str = "") -> FileResponse:  # noqa: ARG001
        """
        Serve the Single Page Application (SPA) index.html file for any path
        """
        index_path = os.path.join(settings.static_files_dir, "index.html")
        return FileResponse(index_path)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware that injects COEP and COOP headers into every response."""

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Cross-Origin-Embedder-Policy", "require-corp")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        return response


fastapi_app.add_middleware(SecurityHeadersMiddleware)

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
    log_config["formatters"]["default"]["fmt"] = settings.log_format
    log_config["formatters"]["access"]["fmt"] = settings.log_format

    logger.info(f"Starting {settings.app_name} in {settings.environment} mode")
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        log_config=log_config,
    )


if __name__ == "__main__":
    main()
