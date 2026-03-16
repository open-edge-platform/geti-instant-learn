# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from domain.services.schemas.health import HealthStatus
from main import health_check


@pytest.fixture
def mock_license_service():
    return MagicMock()


class TestHealthEndpoint:
    def test_health_check_license_accepted(self, mock_license_service):
        mock_license_service.is_accepted.return_value = True

        result = health_check(license_service=mock_license_service)

        assert result.status == HealthStatus.OK
        assert result.license_accepted is True
        mock_license_service.is_accepted.assert_called_once()

    def test_health_check_license_not_accepted(self, mock_license_service):
        mock_license_service.is_accepted.return_value = False

        result = health_check(license_service=mock_license_service)

        assert result.status == HealthStatus.OK
        assert result.license_accepted is False
        mock_license_service.is_accepted.assert_called_once()
