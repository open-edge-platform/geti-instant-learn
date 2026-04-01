# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from dependencies import get_license_service


@pytest.fixture
def mock_license_service():
    return MagicMock()


@pytest.fixture
def app(mock_license_service):
    """Create a test FastAPI app with the license endpoint."""
    app = FastAPI()
    app.add_exception_handler(Exception, custom_exception_handler)

    # Override the dependency before importing endpoints
    app.dependency_overrides[get_license_service] = lambda: mock_license_service

    # Import and register endpoints after override is in place
    from api.endpoints import license as _  # noqa: F401
    from api.routers import system_router

    app.include_router(system_router, prefix="/api/v1")

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestAcceptLicense:
    """Tests for POST /system/license/accept endpoint."""

    def test_accept_license_first_time(self, client, mock_license_service):
        mock_license_service.is_accepted.return_value = False

        resp = client.post("/api/v1/system/license/accept")

        assert resp.status_code == 200
        assert resp.json() == {"accepted": True}
        mock_license_service.accept.assert_called_once()

    def test_accept_license_already_accepted(self, client, mock_license_service):
        mock_license_service.is_accepted.return_value = True

        resp = client.post("/api/v1/system/license/accept")

        assert resp.status_code == 200
        assert resp.json() == {"accepted": True}
        mock_license_service.accept.assert_not_called()

    def test_accept_license_persist_failure(self, client, mock_license_service):
        mock_license_service.is_accepted.return_value = False
        mock_license_service.accept.side_effect = OSError("Cannot create consent file")

        resp = client.post("/api/v1/system/license/accept")

        assert resp.status_code == 500
        assert "internal server error" in resp.json()["detail"].lower()
        mock_license_service.accept.assert_called_once()
