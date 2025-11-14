# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from domain.db.constraints import UniqueConstraintName
from domain.db.models import ProcessorDB, ProjectDB
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
)
from domain.services.model import ModelService
from domain.services.schemas.processor import (
    ProcessorCreateSchema,
    ProcessorListSchema,
    ProcessorSchema,
    ProcessorUpdateSchema,
)


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session."""
    session = Mock(spec=Session)
    return session


@pytest.fixture
def mock_processor_repository():
    """Mock ProcessorRepository."""
    return Mock()


@pytest.fixture
def mock_project_repository():
    """Mock ProjectRepository."""
    return Mock()


@pytest.fixture
def mock_dispatcher():
    """Mock ConfigChangeDispatcher."""
    return Mock()


@pytest.fixture
def service(mock_session, mock_processor_repository, mock_project_repository, mock_dispatcher):
    """Create ModelService instance with mocked dependencies."""
    return ModelService(
        session=mock_session,
        processor_repository=mock_processor_repository,
        project_repository=mock_project_repository,
        config_change_dispatcher=mock_dispatcher,
    )


@pytest.fixture
def sample_project_id():
    """Sample project UUID."""
    return uuid4()


@pytest.fixture
def sample_model_id():
    """Sample model configuration UUID."""
    return uuid4()


@pytest.fixture
def sample_project_db(sample_project_id):
    """Sample ProjectDB instance."""
    project = Mock(spec=ProjectDB)
    project.id = sample_project_id
    return project


@pytest.fixture
def sample_processor_db(sample_model_id, sample_project_id):
    """Sample ProcessorDB instance."""
    processor = Mock(spec=ProcessorDB)
    processor.id = sample_model_id
    processor.project_id = sample_project_id
    processor.name = "test_processor"
    processor.active = True
    processor.config = {
        "mask_similarity_threshold": 0.38,
        "model_type": "matcher",
        "num_background_points": 2,
        "num_foreground_points": 40,
        "precision": "bf16",
    }
    return processor


class TestListModels:
    """Tests for list_models method."""

    def test_list_models_success(
        self, service, mock_project_repository, mock_processor_repository, sample_project_id, sample_project_db
    ):
        """Test successfully listing models."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_paginated.return_value = ([], 0)

        result = service.list_models(sample_project_id, offset=0, limit=20)

        assert isinstance(result, ProcessorListSchema)
        assert result.models == []
        assert result.pagination.total == 0
        assert result.pagination.count == 0
        mock_project_repository.get_by_id.assert_called_once_with(sample_project_id)
        mock_processor_repository.get_paginated.assert_called_once_with(
            project_id=sample_project_id, offset=0, limit=20
        )

    def test_list_models_project_not_found(self, service, mock_project_repository, sample_project_id):
        """Test listing models when project does not exist."""
        mock_project_repository.get_by_id.return_value = None

        with pytest.raises(ResourceNotFoundError) as exc_info:
            service.list_models(sample_project_id)

        assert exc_info.value.resource_type == ResourceType.PROJECT


class TestGetModel:
    """Tests for get_model method."""

    def test_get_model_success(
        self,
        service,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_model_id,
        sample_project_db,
        sample_processor_db,
    ):
        """Test successfully retrieving a model configuration."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_by_id_and_project.return_value = sample_processor_db

        with patch("domain.services.model.processor_db_to_schema") as mock_mapper:
            mock_mapper.return_value = Mock(spec=ProcessorSchema)
            result = service.get_model(sample_project_id, sample_model_id)

            assert result is not None
            mock_processor_repository.get_by_id_and_project.assert_called_once_with(
                processor_id=sample_model_id, project_id=sample_project_id
            )

    def test_get_model_not_found(
        self,
        service,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_model_id,
        sample_project_db,
    ):
        """Test getting model configuration that does not exist."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_by_id_and_project.return_value = None

        with pytest.raises(ResourceNotFoundError) as exc_info:
            service.get_model(sample_project_id, sample_model_id)

        assert exc_info.value.resource_type == ResourceType.PROCESSOR


class TestCreateModel:
    """Tests for create_model method."""

    def test_create_model_success(
        self,
        service,
        mock_session,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_project_db,
    ):
        """Test successfully creating a model configuration."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_activated_in_project.return_value = None

        create_data = Mock(spec=ProcessorCreateSchema)
        create_data.name = "new_processor"
        create_data.active = False
        create_data.config = Mock()
        create_data.config.model_type.value = "test_type"

        with (
            patch("domain.services.model.processor_schema_to_db") as mock_schema_to_db,
            patch("domain.services.model.processor_db_to_schema") as mock_db_to_schema,
        ):
            new_processor = Mock(spec=ProcessorDB)
            new_processor.id = uuid4()
            new_processor.config = {"model_type": "test_type"}
            new_processor.active = False
            mock_schema_to_db.return_value = new_processor
            mock_db_to_schema.return_value = Mock(spec=ProcessorSchema)

            result = service.create_model(sample_project_id, create_data)

            assert result is not None
            mock_processor_repository.add.assert_called_once_with(new_processor)
            mock_session.commit.assert_called_once()
            mock_session.refresh.assert_called_once_with(new_processor)

    def test_create_model_deactivates_existing_active(
        self,
        service,
        mock_session,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_project_db,
    ):
        """Test creating active model configuration deactivates existing active one."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        existing_active = Mock(spec=ProcessorDB)
        existing_active.id = uuid4()
        existing_active.active = True
        mock_processor_repository.get_activated_in_project.return_value = existing_active

        create_data = Mock(spec=ProcessorCreateSchema)
        create_data.name = "new_processor"
        create_data.active = True
        create_data.id = uuid4()
        create_data.config = Mock()
        create_data.config.model_type.value = "test_type"

        with (
            patch("domain.services.model.processor_schema_to_db") as mock_schema_to_db,
            patch("domain.services.model.processor_db_to_schema"),
        ):
            mock_schema_to_db.return_value = Mock(spec=ProcessorDB, id=uuid4(), active=True)

            service.create_model(sample_project_id, create_data)

            assert existing_active.active is False

    def test_create_model_integrity_error(
        self,
        service,
        mock_session,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_project_db,
    ):
        """Test handling of IntegrityError during creation."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_activated_in_project.return_value = None

        create_data = Mock(spec=ProcessorCreateSchema)
        create_data.name = "duplicate_processor"
        create_data.active = False
        create_data.config = Mock()
        create_data.config.model_type.value = "test_type"

        mock_session.commit.side_effect = IntegrityError("statement", {}, Exception("unique constraint"))

        with (
            patch("domain.services.model.processor_schema_to_db"),
            patch(
                "domain.services.model.extract_constraint_name",
                return_value=UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT,
            ),
        ):
            with pytest.raises(ResourceAlreadyExistsError):
                service.create_model(sample_project_id, create_data)

            mock_session.rollback.assert_called_once()


class TestGetActiveModel:
    """Tests for get_active_model method."""

    def test_get_active_model_success(
        self,
        service,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_project_db,
        sample_processor_db,
    ):
        """Test successfully retrieving active model configuration."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_activated_in_project.return_value = sample_processor_db

        with patch("domain.services.model.processor_db_to_schema") as mock_mapper:
            mock_mapper.return_value = Mock(spec=ProcessorSchema)
            result = service.get_active_model(sample_project_id)

            assert result is not None
            mock_processor_repository.get_activated_in_project.assert_called_once_with(sample_project_id)

    def test_get_active_model_not_found(
        self, service, mock_project_repository, mock_processor_repository, sample_project_id, sample_project_db
    ):
        """Test getting active model configuration when none exists."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_activated_in_project.return_value = None

        with pytest.raises(ResourceNotFoundError) as exc_info:
            service.get_active_model(sample_project_id)

        assert exc_info.value.resource_type == ResourceType.PROCESSOR


class TestUpdateModel:
    """Tests for update_model method."""

    def test_update_model_success(
        self,
        service,
        mock_session,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_model_id,
        sample_project_db,
        sample_processor_db,
    ):
        """Test successfully updating a model configuration."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_by_id_and_project.return_value = sample_processor_db

        # Ensure the mock processor has a valid UUID for its id
        sample_processor_db.id = uuid4()

        # Mock the active model returned by get_activated_in_project
        active_model = Mock(spec=ProcessorDB)
        active_model.id = uuid4()
        mock_processor_repository.get_activated_in_project.return_value = active_model

        update_data = Mock(spec=ProcessorUpdateSchema)
        update_data.name = "updated_name"
        update_data.active = True
        update_data.config = Mock()
        update_data.config.model_dump.return_value = {"model_type": "updated_type"}

        with patch("domain.services.model.processor_db_to_schema") as mock_mapper:
            mock_mapper.return_value = Mock(spec=ProcessorSchema)
            result = service.update_model(sample_project_id, sample_model_id, update_data)

            assert result is not None
            assert sample_processor_db.name == "updated_name"
            assert sample_processor_db.active is True
            mock_session.commit.assert_called_once()

    def test_update_model_not_found(
        self,
        service,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_model_id,
        sample_project_db,
    ):
        """Test updating model configuration that does not exist."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_by_id_and_project.return_value = None

        update_data = Mock(spec=ProcessorUpdateSchema)

        with pytest.raises(ResourceNotFoundError):
            service.update_model(sample_project_id, sample_model_id, update_data)


class TestDeleteModel:
    """Tests for delete_model method."""

    def test_delete_model_success(
        self,
        service,
        mock_session,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_model_id,
        sample_project_db,
        sample_processor_db,
    ):
        """Test successfully deleting a model configuration."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_by_id_and_project.return_value = sample_processor_db

        service.delete_model(sample_project_id, sample_model_id)

        mock_processor_repository.delete.assert_called_once_with(sample_processor_db)
        mock_session.commit.assert_called_once()

    def test_delete_model_not_found(
        self,
        service,
        mock_project_repository,
        mock_processor_repository,
        sample_project_id,
        sample_model_id,
        sample_project_db,
    ):
        """Test deleting model configuration that does not exist."""
        mock_project_repository.get_by_id.return_value = sample_project_db
        mock_processor_repository.get_by_id_and_project.return_value = None

        with pytest.raises(ResourceNotFoundError) as exc_info:
            service.delete_model(sample_project_id, sample_model_id)

        assert exc_info.value.resource_type == ResourceType.PROCESSOR


class TestHandleSourceIntegrityError:
    """Tests for _handle_source_integrity_error method."""

    def test_handle_foreign_key_error(self, service, sample_model_id, sample_project_id):
        """Test handling foreign key constraint violation."""
        exc = IntegrityError("statement", {}, Exception("foreign key constraint"))

        with patch("domain.services.model.extract_constraint_name", return_value=None):
            with pytest.raises(ResourceNotFoundError) as exc_info:
                service._handle_source_integrity_error(exc, sample_model_id, sample_project_id, "test")

            assert "project does not exist" in str(exc_info.value).lower()

    def test_handle_unique_name_error(self, service, sample_model_id, sample_project_id):
        """Test handling unique name constraint violation."""
        exc = IntegrityError("statement", {}, Exception("unique constraint"))

        with patch(
            "domain.services.model.extract_constraint_name",
            return_value=UniqueConstraintName.PROCESSOR_NAME_PER_PROJECT,
        ):
            with pytest.raises(ResourceAlreadyExistsError) as exc_info:
                service._handle_source_integrity_error(exc, sample_model_id, sample_project_id, "duplicate_name")

            assert exc_info.value.field == "name"

    def test_handle_unmapped_constraint_error(self, service, sample_model_id, sample_project_id):
        """Test handling unmapped constraint violation."""
        exc = IntegrityError("statement", {}, Exception("some other constraint"))

        with patch("domain.services.model.extract_constraint_name", return_value=None):
            with pytest.raises(ValueError) as exc_info:
                service._handle_source_integrity_error(exc, sample_model_id, sample_project_id, "test")

            assert "constraint violation" in str(exc_info.value).lower()
