# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest
from fastapi import status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from dependencies import get_discovery_service
from domain.services.schemas.reader import SourceType, UsbCameraConfig
from main import fastapi_app


@pytest.fixture
def fxt_client():
    fastapi_app.add_exception_handler(Exception, custom_exception_handler)
    fastapi_app.add_exception_handler(RequestValidationError, custom_exception_handler)
    return TestClient(fastapi_app, raise_server_exceptions=False)


@pytest.fixture
def fxt_discovery_service() -> MagicMock:
    discovery_service = MagicMock()
    fastapi_app.dependency_overrides[get_discovery_service] = lambda: discovery_service
    return discovery_service


@pytest.fixture
def fxt_usb_camera_sources() -> list[UsbCameraConfig]:
    return [
        UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=0, name="Camera 0"),
        UsbCameraConfig(source_type=SourceType.USB_CAMERA, device_id=1, name="Camera 1"),
    ]


class TestSourceTypeEndpoints:
    def test_get_available_sources_success(self, fxt_discovery_service, fxt_usb_camera_sources, fxt_client):
        fxt_discovery_service.list_available_sources.return_value = fxt_usb_camera_sources

        response = fxt_client.get(f"/api/v1/source-types/{SourceType.USB_CAMERA}/sources")

        assert response.status_code == status.HTTP_200_OK
        fxt_discovery_service.list_available_sources.assert_called_once_with(SourceType.USB_CAMERA)
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0]["source_type"] == "usb_camera"
        assert data[0]["device_id"] == 0
        assert data[0]["name"] == "Camera 0"
        assert data[1]["device_id"] == 1
        assert data[1]["name"] == "Camera 1"

    def test_get_available_sources_empty_list(self, fxt_discovery_service, fxt_client):
        fxt_discovery_service.list_available_sources.return_value = []

        response = fxt_client.get(f"/api/v1/source-types/{SourceType.USB_CAMERA}/sources")

        assert response.status_code == status.HTTP_200_OK
        fxt_discovery_service.list_available_sources.assert_called_once_with(SourceType.USB_CAMERA)
        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_get_available_sources_not_supported(self, fxt_discovery_service, fxt_client):
        fxt_discovery_service.list_available_sources.side_effect = ValueError(
            f"Discovery not supported for source type: {SourceType.USB_CAMERA}"
        )

        response = fxt_client.get(f"/api/v1/source-types/{SourceType.USB_CAMERA}/sources")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        fxt_discovery_service.list_available_sources.assert_called_once_with(SourceType.USB_CAMERA)
        assert "detail" in response.json()

    def test_get_available_sources_error(self, fxt_discovery_service, fxt_client):
        fxt_discovery_service.list_available_sources.side_effect = RuntimeError("Failed to enumerate devices")

        response = fxt_client.get(f"/api/v1/source-types/{SourceType.USB_CAMERA}/sources")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        fxt_discovery_service.list_available_sources.assert_called_once_with(SourceType.USB_CAMERA)
        assert "detail" in response.json()
