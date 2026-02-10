# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from api.routers import license_router


@pytest.fixture
def app():
    from api.endpoints import license as _  # noqa: F401

    app = FastAPI()
    app.include_router(license_router, prefix="/api/v1")
    app.add_exception_handler(Exception, custom_exception_handler)
    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestAcceptLicense:
    """Tests for POST /license/accept endpoint."""

    def test_accept_license_first_time(self, client):
        """License accepted for the first time."""
        mock_service = MagicMock()
        mock_service.is_accepted.return_value = False

        with patch("api.endpoints.license.LicenseService", return_value=mock_service):
            resp = client.post("/api/v1/license/accept")

        assert resp.status_code == 200
        assert resp.json() == {"accepted": True}
        mock_service.is_accepted.assert_called_once()
        mock_service.accept.assert_called_once()

    def test_accept_license_already_accepted(self, client):
        """License already accepted - idempotent behavior."""
        mock_service = MagicMock()
        mock_service.is_accepted.return_value = True

        with patch("api.endpoints.license.LicenseService", return_value=mock_service):
            resp = client.post("/api/v1/license/accept")

        assert resp.status_code == 200
        assert resp.json() == {"accepted": True}
        mock_service.is_accepted.assert_called_once()
        mock_service.accept.assert_not_called()

    def test_accept_license_persist_failure(self, client):
        """License acceptance fails due to persistence error."""
        mock_service = MagicMock()
        mock_service.is_accepted.return_value = False
        mock_service.accept.side_effect = OSError("Cannot create consent file")

        with patch("api.endpoints.license.LicenseService", return_value=mock_service):
            resp = client.post("/api/v1/license/accept")

        assert resp.status_code == 500
        assert "internal server error" in resp.json()["detail"].lower()
