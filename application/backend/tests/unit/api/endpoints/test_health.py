# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from domain.services.schemas.health import HealthCheckSchema


@pytest.fixture
def app():
    app = FastAPI()

    @app.get("/health")
    async def health_check() -> HealthCheckSchema:
        from instantlearn_license.service import LicenseService

        license_service = LicenseService()
        return HealthCheckSchema(status="ok", license_accepted=license_service.is_accepted())

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_health_check_license_accepted(self, client):
        """Health check returns license_accepted=True when license is accepted."""
        mock_service = MagicMock()
        mock_service.is_accepted.return_value = True

        with patch("instantlearn_license.service.LicenseService", return_value=mock_service):
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["license_accepted"] is True

    def test_health_check_license_not_accepted(self, client):
        """Health check returns license_accepted=False when license is not accepted."""
        mock_service = MagicMock()
        mock_service.is_accepted.return_value = False

        with patch("instantlearn_license.service.LicenseService", return_value=mock_service):
            resp = client.get("/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["license_accepted"] is False

