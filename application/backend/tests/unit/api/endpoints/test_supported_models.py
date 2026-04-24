# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from domain.repositories.supported_model import SupportedModelRepository
from domain.services.schemas.processor import (
    ModelType,
    SupportedModelsListSchema,
)


@pytest.fixture
def app():
    from api.endpoints import models as _  # noqa: F401
    from api.routers import system_router

    test_app = FastAPI()
    test_app.add_exception_handler(Exception, custom_exception_handler)
    test_app.add_exception_handler(RequestValidationError, custom_exception_handler)
    test_app.include_router(system_router, prefix="/api/v1")

    return test_app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


class TestGetSupportedModels:
    """Tests for GET /api/v1/system/supported-models."""

    def test_returns_200(self, client):
        response = client.get("/api/v1/system/supported-models")

        assert response.status_code == status.HTTP_200_OK

    def test_response_is_valid_schema(self, client):
        response = client.get("/api/v1/system/supported-models")

        body = response.json()
        assert "models" in body
        assert isinstance(body["models"], list)

    def test_response_has_pagination(self, client):
        response = client.get("/api/v1/system/supported-models")
        body = response.json()
        assert "pagination" in body
        assert body["pagination"]["total"] == 4
        assert body["pagination"]["count"] == 4

    def test_response_is_parseable_by_schema(self, client):
        response = client.get("/api/v1/system/supported-models")
        parsed = SupportedModelsListSchema.model_validate(response.json())
        assert len(parsed.models) == 4

    def test_full_payload_matches_repository(self, client):
        """Endpoint response matches the serialized repository data exactly."""
        repo = SupportedModelRepository()
        expected_models = [m.model_dump(mode="json") for m in repo.get_all()]

        response = client.get("/api/v1/system/supported-models")
        actual_models = response.json()["models"]

        assert actual_models == expected_models


class TestSupportedModelRepository:
    """Unit tests for the SupportedModelRepository itself."""

    def test_get_all_returns_all_model_types(self):
        repo = SupportedModelRepository()
        all_models = repo.get_all()
        model_types = {m.default_config.model_type for m in all_models}
        assert model_types == {ModelType.MATCHER, ModelType.PERDINO, ModelType.SOFT_MATCHER, ModelType.SAM3}

    def test_get_by_model_type_returns_none_for_unknown(self):
        repo = SupportedModelRepository()
        assert repo.get_by_model_type("nonexistent") is None  # type: ignore[arg-type]

    def test_get_supported_annotation_types(self):
        from domain.services.schemas.annotation import AnnotationType

        repo = SupportedModelRepository()
        assert AnnotationType.POLYGON in repo.get_supported_annotation_types(ModelType.MATCHER)
        assert AnnotationType.RECTANGLE in repo.get_supported_annotation_types(ModelType.SAM3)
        assert AnnotationType.POLYGON not in repo.get_supported_annotation_types(ModelType.SAM3)
