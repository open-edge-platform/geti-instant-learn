# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from domain.repositories.supported_model import SupportedModelRepository
from domain.services.schemas.processor import ModelType


class TestSupportedModelRepository:
    """Unit tests for the SupportedModelRepository"""

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
