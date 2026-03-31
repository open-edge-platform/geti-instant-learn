# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from api.error_handler import custom_exception_handler
from domain.services.schemas.processor import (
    MatcherConfig,
    ModelType,
    PerDinoConfig,
    Sam3Config,
    SoftMatcherConfig,
    SupportedModelsListSchema,
    SupportedPromptType,
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
        # Validates the top-level structure
        assert "models" in body
        assert isinstance(body["models"], list)

    def test_all_models_returned(self, client):
        response = client.get("/api/v1/system/supported-models")

        model_types = {m["default_config"]["model_type"] for m in response.json()["models"]}
        assert model_types == {
            ModelType.MATCHER,
            ModelType.PERDINO,
            ModelType.SOFT_MATCHER,
            # ModelType.SAM3,
        }

    def test_each_model_has_required_fields(self, client):
        response = client.get("/api/v1/system/supported-models")

        for model in response.json()["models"]:
            assert "default_config" in model
            assert "supported_prompt_types" in model
            assert isinstance(model["supported_prompt_types"], list)
            assert len(model["supported_prompt_types"]) > 0

    def test_response_has_pagination(self, client):
        response = client.get("/api/v1/system/supported-models")
        body = response.json()
        assert "pagination" in body
        assert body["pagination"]["total"] == 3
        assert body["pagination"]["count"] == 3

    def test_response_is_parseable_by_schema(self, client):
        response = client.get("/api/v1/system/supported-models")
        parsed = SupportedModelsListSchema.model_validate(response.json())
        assert len(parsed.models) == 3


class TestMatcherModel:
    """Verify Matcher model metadata."""

    def _get_matcher(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]
        return next(m for m in models if m["default_config"]["model_type"] == ModelType.MATCHER)

    def test_matcher_prompt_types(self, client):
        matcher = self._get_matcher(client)

        assert matcher["supported_prompt_types"] == [SupportedPromptType.VISUAL_POLYGON]

    def test_matcher_default_config_matches_class_defaults(self, client):
        matcher = self._get_matcher(client)
        expected = MatcherConfig()

        assert matcher["default_config"]["confidence_threshold"] == expected.confidence_threshold
        assert matcher["default_config"]["num_foreground_points"] == expected.num_foreground_points
        assert matcher["default_config"]["num_background_points"] == expected.num_background_points
        assert matcher["default_config"]["precision"] == expected.precision
        assert matcher["default_config"]["sam_model"] == expected.sam_model
        assert matcher["default_config"]["encoder_model"] == expected.encoder_model
        assert matcher["default_config"]["use_mask_refinement"] == expected.use_mask_refinement


class TestPerDinoModel:
    """Verify PerDINO model metadata."""

    def _get_perdino(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]
        return next(m for m in models if m["default_config"]["model_type"] == ModelType.PERDINO)

    def test_perdino_prompt_types(self, client):
        perdino = self._get_perdino(client)

        assert perdino["supported_prompt_types"] == [SupportedPromptType.VISUAL_POLYGON]

    def test_perdino_default_config_matches_class_defaults(self, client):
        perdino = self._get_perdino(client)
        expected = PerDinoConfig()

        assert perdino["default_config"]["confidence_threshold"] == expected.confidence_threshold
        assert perdino["default_config"]["num_foreground_points"] == expected.num_foreground_points
        assert perdino["default_config"]["num_background_points"] == expected.num_background_points
        assert perdino["default_config"]["num_grid_cells"] == expected.num_grid_cells
        assert perdino["default_config"]["point_selection_threshold"] == expected.point_selection_threshold
        assert perdino["default_config"]["precision"] == expected.precision
        assert perdino["default_config"]["sam_model"] == expected.sam_model
        assert perdino["default_config"]["encoder_model"] == expected.encoder_model


class TestSoftMatcherModel:
    """Verify SoftMatcher model metadata."""

    def _get_soft_matcher(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]
        return next(m for m in models if m["default_config"]["model_type"] == ModelType.SOFT_MATCHER)

    def test_soft_matcher_prompt_types(self, client):
        soft_matcher = self._get_soft_matcher(client)

        assert soft_matcher["supported_prompt_types"] == [SupportedPromptType.VISUAL_POLYGON]

    def test_soft_matcher_default_config_matches_class_defaults(self, client):
        soft_matcher = self._get_soft_matcher(client)
        expected = SoftMatcherConfig()

        assert soft_matcher["default_config"]["confidence_threshold"] == expected.confidence_threshold
        assert soft_matcher["default_config"]["num_foreground_points"] == expected.num_foreground_points
        assert soft_matcher["default_config"]["num_background_points"] == expected.num_background_points
        assert soft_matcher["default_config"]["use_sampling"] == expected.use_sampling
        assert soft_matcher["default_config"]["use_spatial_sampling"] == expected.use_spatial_sampling
        assert soft_matcher["default_config"]["approximate_matching"] == expected.approximate_matching
        assert soft_matcher["default_config"]["softmatching_score_threshold"] == expected.softmatching_score_threshold
        assert soft_matcher["default_config"]["softmatching_bidirectional"] == expected.softmatching_bidirectional
        assert soft_matcher["default_config"]["precision"] == expected.precision
        assert soft_matcher["default_config"]["sam_model"] == expected.sam_model
        assert soft_matcher["default_config"]["encoder_model"] == expected.encoder_model


class TestSam3Model:
    """Verify SAM3 model metadata."""

    def _get_sam3(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]
        return next(m for m in models if m["default_config"]["model_type"] == ModelType.SAM3)

    def test_sam3_prompt_types(self, client):
        sam3 = self._get_sam3(client)

        assert set(sam3["supported_prompt_types"]) == {
            SupportedPromptType.TEXT,
            SupportedPromptType.VISUAL_RECTANGLE,
        }

    def test_sam3_does_not_support_visual_polygon(self, client):
        sam3 = self._get_sam3(client)

        assert SupportedPromptType.VISUAL_POLYGON not in sam3["supported_prompt_types"]

    def test_sam3_default_config_matches_class_defaults(self, client):
        sam3 = self._get_sam3(client)
        expected = Sam3Config()

        assert sam3["default_config"]["confidence_threshold"] == expected.confidence_threshold
        assert sam3["default_config"]["resolution"] == expected.resolution
        assert sam3["default_config"]["precision"] == expected.precision

    def test_sam3_has_no_sam_or_encoder_fields(self, client):
        """SAM3 uses its own base — it has no sam_model / encoder_model fields."""
        sam3 = self._get_sam3(client)

        assert "sam_model" not in sam3["default_config"]
        assert "encoder_model" not in sam3["default_config"]


class TestPromptTypeCoverage:
    """Cross-cutting checks on prompt type assignments."""

    def test_visual_polygon_models(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]

        polygon_models = {
            m["default_config"]["model_type"]
            for m in models
            if SupportedPromptType.VISUAL_POLYGON in m["supported_prompt_types"]
        }
        assert polygon_models == {ModelType.MATCHER, ModelType.PERDINO, ModelType.SOFT_MATCHER}

    def test_text_prompt_models(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]

        text_models = {
            m["default_config"]["model_type"] for m in models if SupportedPromptType.TEXT in m["supported_prompt_types"]
        }
        assert text_models == {ModelType.SAM3}

    def test_visual_rectangle_models(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]

        rect_models = {
            m["default_config"]["model_type"]
            for m in models
            if SupportedPromptType.VISUAL_RECTANGLE in m["supported_prompt_types"]
        }
        assert rect_models == {ModelType.SAM3}

    def test_no_unknown_prompt_types(self, client):
        response = client.get("/api/v1/system/supported-models")
        models = response.json()["models"]

        known = {pt.value for pt in SupportedPromptType}
        for model in models:
            for pt in model["supported_prompt_types"]:
                assert pt in known, f"Unknown prompt type '{pt}' for model '{model['model_type']}'"
