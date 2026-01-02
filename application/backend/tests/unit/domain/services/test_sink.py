# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from domain.db.models import ProjectDB, SinkDB
from domain.dispatcher import ComponentConfigChangeEvent, ComponentType, ConfigChangeDispatcher
from domain.errors import (
    ResourceAlreadyExistsError,
    ResourceNotFoundError,
    ResourceType,
    ResourceUpdateConflictError,
)
from domain.repositories.project import ProjectRepository
from domain.repositories.sink import SinkRepository
from domain.services.schemas.sink import (
    SinkCreateSchema,
    SinkSchema,
    SinksListSchema,
    SinkUpdateSchema,
)
from domain.services.schemas.writer import WriterType
from domain.services.sink import SinkService


@pytest.fixture
def mock_session():
    """Mock SQLAlchemy session."""
    session = Mock(spec=Session)
    session.commit = Mock()
    session.rollback = Mock()
    session.refresh = Mock()
    return session


@pytest.fixture
def mock_sink_repository():
    """Mock SinkRepository."""
    return Mock(spec=SinkRepository)


@pytest.fixture
def mock_project_repository():
    """Mock ProjectRepository."""
    return Mock(spec=ProjectRepository)


@pytest.fixture
def mock_dispatcher():
    """Mock ConfigChangeDispatcher."""
    return Mock(spec=ConfigChangeDispatcher)


@pytest.fixture
def sink_service(mock_session, mock_sink_repository, mock_project_repository, mock_dispatcher):
    """Create SinkService instance with mocked dependencies."""
    return SinkService(
        session=mock_session,
        sink_repository=mock_sink_repository,
        project_repository=mock_project_repository,
        config_change_dispatcher=mock_dispatcher,
    )


@pytest.fixture
def project_id():
    """Sample project ID."""
    return uuid4()


@pytest.fixture
def sink_id():
    """Sample sink ID."""
    return uuid4()


@pytest.fixture
def mock_project(project_id):
    """Mock ProjectDB entity."""
    project = Mock(spec=ProjectDB)
    project.id = project_id
    project.name = "Test Project"
    return project


@pytest.fixture
def mock_sink(sink_id, project_id):
    """Mock SinkDB entity."""
    sink = Mock(spec=SinkDB)
    sink.id = sink_id
    sink.project_id = project_id
    sink.active = False
    sink.config = {"sink_type": WriterType.MQTT, "broker_host": "localhost"}
    return sink


@pytest.fixture
def sink_create_data():
    """Sample SinkCreateSchema."""
    sink_id = uuid4()
    config_mock = Mock()
    config_mock.sink_type = WriterType.MQTT
    config_mock.name = "test-sink"
    config_mock.model_dump = Mock(return_value={"sink_type": WriterType.MQTT, "broker_host": "localhost"})

    schema = Mock(spec=SinkCreateSchema)
    schema.id = sink_id
    schema.active = False
    schema.config = config_mock
    return schema


@pytest.fixture
def sink_update_data():
    """Sample SinkUpdateSchema."""
    config_mock = Mock()
    config_mock.sink_type = WriterType.MQTT
    config_mock.name = "updated-sink"
    config_mock.model_dump = Mock(return_value={"sink_type": WriterType.MQTT, "broker_host": "newhost"})

    schema = Mock(spec=SinkUpdateSchema)
    schema.id = uuid4()
    schema.active = False
    schema.config = config_mock
    return schema


class TestListSinks:
    """Tests for list_sinks method."""

    def test_list_sinks_success(
        self, sink_service, mock_project_repository, mock_sink_repository, project_id, mock_project, mock_sink
    ):
        """Test successfully listing all sinks in a project."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.list_with_pagination_by_project.return_value = ([mock_sink], 1)

        # Act
        result = sink_service.list_sinks(project_id)

        # Assert
        assert isinstance(result, SinksListSchema)
        mock_project_repository.get_by_id.assert_called_once_with(project_id)
        mock_sink_repository.list_with_pagination_by_project.assert_called_once()

    def test_list_sinks_empty_list(
        self, sink_service, mock_project_repository, mock_sink_repository, project_id, mock_project
    ):
        """Test listing sinks when project has no sinks."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.list_with_pagination_by_project.return_value = ([], 0)

        # Act
        result = sink_service.list_sinks(project_id)

        # Assert
        assert isinstance(result, SinksListSchema)
        assert len(result.sinks) == 0

    def test_list_sinks_project_not_found(self, sink_service, mock_project_repository, project_id):
        """Test listing sinks when project does not exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.list_sinks(project_id)

        assert exc_info.value.resource_type == ResourceType.PROJECT
        assert exc_info.value.resource_id == str(project_id)


class TestGetSink:
    """Tests for get_sink method."""

    def test_get_sink_success(
        self, sink_service, mock_project_repository, mock_sink_repository, project_id, sink_id, mock_project, mock_sink
    ):
        """Test successfully retrieving a sink."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink

        # Act
        result = sink_service.get_sink(project_id, sink_id)

        # Assert
        assert isinstance(result, SinkSchema)
        mock_sink_repository.get_by_id_and_project.assert_called_once_with(sink_id, project_id)

    def test_get_sink_not_found(
        self, sink_service, mock_project_repository, mock_sink_repository, project_id, sink_id, mock_project
    ):
        """Test getting a sink that doesn't exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.get_sink(project_id, sink_id)

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.resource_id == str(sink_id)

    def test_get_sink_project_not_found(self, sink_service, mock_project_repository, project_id, sink_id):
        """Test getting a sink when project doesn't exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.get_sink(project_id, sink_id)

        assert exc_info.value.resource_type == ResourceType.PROJECT


class TestCreateSink:
    """Tests for create_sink method."""

    def test_create_sink_success(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        mock_dispatcher,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test successfully creating a sink."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        new_sink = Mock(spec=SinkDB)
        new_sink.id = uuid4()
        new_sink.config = {"sink_type": WriterType.MQTT}
        new_sink.active = False
        mock_sink_repository.add.return_value = None
        mock_session.refresh.side_effect = lambda x: setattr(x, "id", new_sink.id)

        # Act
        result = sink_service.create_sink(project_id, sink_create_data)

        # Assert
        assert isinstance(result, SinkSchema)
        mock_sink_repository.add.assert_called_once()
        mock_session.commit.assert_called_once()
        mock_dispatcher.dispatch.assert_called_once()

    def test_create_sink_connected_disconnects_existing(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test creating a active sink disconnects existing active sink."""
        # Arrange
        sink_create_data.active = True
        mock_project_repository.get_by_id.return_value = mock_project
        existing_connected_sink = Mock(spec=SinkDB)
        existing_connected_sink.active = True
        existing_connected_sink.id = uuid4()
        mock_sink_repository.get_active_in_project.return_value = existing_connected_sink

        # Act
        sink_service.create_sink(project_id, sink_create_data)

        # Assert
        assert existing_connected_sink.active is False
        mock_sink_repository.get_active_in_project.assert_called_once_with(project_id)

    def test_create_sink_project_not_found(self, sink_service, mock_project_repository, project_id, sink_create_data):
        """Test creating a sink when project doesn't exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.create_sink(project_id, sink_create_data)

        assert exc_info.value.resource_type == ResourceType.PROJECT

    def test_create_sink_duplicate_name(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test creating a sink with duplicate name in same project."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_session.commit.side_effect = IntegrityError(
            "statement", "params", orig=Exception("unique constraint uq_sink_name_per_project")
        )

        # Act & Assert
        with pytest.raises(ResourceAlreadyExistsError) as exc_info:
            sink_service.create_sink(project_id, sink_create_data)

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.field == "name"
        mock_session.rollback.assert_called_once()

    def test_create_sink_duplicate_type(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test creating a sink with duplicate type in same project."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_session.commit.side_effect = IntegrityError(
            "statement", "params", orig=Exception("unique constraint uq_sink_type_per_project")
        )

        # Act & Assert
        with pytest.raises(ResourceAlreadyExistsError) as exc_info:
            sink_service.create_sink(project_id, sink_create_data)

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.field == "sink_type"

    def test_create_sink_multiple_active(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test creating a second active sink violates constraint."""
        # Arrange
        sink_create_data.active = True
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_active_in_project.return_value = None
        mock_session.commit.side_effect = IntegrityError(
            "statement", "params", orig=Exception("unique constraint uq_single_active_sink_per_project")
        )

        # Act & Assert
        with pytest.raises(ResourceAlreadyExistsError) as exc_info:
            sink_service.create_sink(project_id, sink_create_data)

        assert exc_info.value.field == "active"

    def test_create_sink_foreign_key_violation(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test creating a sink with invalid foreign key reference."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_session.commit.side_effect = IntegrityError(
            "statement", "params", orig=Exception("foreign key constraint failed")
        )

        # Act & Assert
        with pytest.raises(ResourceNotFoundError):
            sink_service.create_sink(project_id, sink_create_data)


class TestUpdateSink:
    """Tests for update_sink method."""

    def test_update_sink_success(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        mock_dispatcher,
        project_id,
        sink_id,
        mock_project,
        mock_sink,
        sink_update_data,
    ):
        """Test successfully updating a sink."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink
        mock_sink_repository.update.return_value = mock_sink

        # Act
        result = sink_service.update_sink(project_id, sink_id, sink_update_data)

        # Assert
        assert isinstance(result, SinkSchema)
        assert mock_sink.config == sink_update_data.config.model_dump()
        mock_session.commit.assert_called_once()
        mock_dispatcher.dispatch.assert_called_once()

    def test_update_sink_connect_disconnects_existing(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        sink_id,
        mock_project,
        mock_sink,
        sink_update_data,
    ):
        """Test updating sink to active disconnects other active sink."""
        # Arrange
        sink_update_data.active = True
        mock_sink.active = False
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink
        mock_sink_repository.update.return_value = mock_sink

        existing_connected = Mock(spec=SinkDB)
        existing_connected.active = True
        existing_connected.id = uuid4()
        mock_sink_repository.get_active_in_project.return_value = existing_connected

        # Act
        sink_service.update_sink(project_id, sink_id, sink_update_data)

        # Assert
        assert existing_connected.active is False
        assert mock_sink.active is True

    def test_update_sink_not_found(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        project_id,
        sink_id,
        mock_project,
        sink_update_data,
    ):
        """Test updating a sink that doesn't exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.update_sink(project_id, sink_id, sink_update_data)

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.resource_id == str(sink_id)

    def test_update_sink_type_change_forbidden(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        project_id,
        sink_id,
        mock_project,
        mock_sink,
        sink_update_data,
    ):
        """Test updating sink_type is forbidden."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink.config = {"sink_type": WriterType.MQTT}
        sink_update_data.config.sink_type = "HTTP"
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink

        # Act & Assert
        with pytest.raises(ResourceUpdateConflictError) as exc_info:
            sink_service.update_sink(project_id, sink_id, sink_update_data)

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.field == "sink_type"

    def test_update_sink_duplicate_name(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        sink_id,
        mock_project,
        mock_sink,
        sink_update_data,
    ):
        """Test updating sink with duplicate name in same project."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink
        mock_sink_repository.update.return_value = mock_sink
        mock_session.commit.side_effect = IntegrityError(
            "statement", "params", orig=Exception("unique constraint uq_sink_name_per_project")
        )

        # Act & Assert
        with pytest.raises(ResourceAlreadyExistsError) as exc_info:
            sink_service.update_sink(project_id, sink_id, sink_update_data)

        assert exc_info.value.field == "name"
        mock_session.rollback.assert_called_once()

    def test_update_sink_project_not_found(
        self, sink_service, mock_project_repository, project_id, sink_id, sink_update_data
    ):
        """Test updating sink when project doesn't exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.update_sink(project_id, sink_id, sink_update_data)

        assert exc_info.value.resource_type == ResourceType.PROJECT


class TestDeleteSink:
    """Tests for delete_sink method."""

    def test_delete_sink_success(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        mock_dispatcher,
        project_id,
        sink_id,
        mock_project,
        mock_sink,
    ):
        """Test successfully deleting a sink."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink

        # Act
        sink_service.delete_sink(project_id, sink_id)

        # Assert
        mock_sink_repository.delete.assert_called_once_with(mock_sink.id)
        mock_session.commit.assert_called_once()
        mock_dispatcher.dispatch.assert_called_once()

    def test_delete_sink_not_found(
        self, sink_service, mock_project_repository, mock_sink_repository, project_id, sink_id, mock_project
    ):
        """Test deleting a sink that doesn't exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.delete_sink(project_id, sink_id)

        assert exc_info.value.resource_type == ResourceType.SINK
        assert exc_info.value.resource_id == str(sink_id)

    def test_delete_sink_project_not_found(self, sink_service, mock_project_repository, project_id, sink_id):
        """Test deleting sink when project doesn't exist."""
        # Arrange
        mock_project_repository.get_by_id.return_value = None

        # Act & Assert
        with pytest.raises(ResourceNotFoundError) as exc_info:
            sink_service.delete_sink(project_id, sink_id)

        assert exc_info.value.resource_type == ResourceType.PROJECT


class TestEventDispatcher:
    """Tests for event dispatcher integration."""

    def test_create_sink_emits_event(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_dispatcher,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test that creating a sink emits a component change event."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        expected_sink_id = sink_create_data.id

        # Act
        sink_service.create_sink(project_id, sink_create_data)

        # Assert
        mock_dispatcher.dispatch.assert_called_once()
        event = mock_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ComponentConfigChangeEvent)
        assert event.project_id == project_id
        assert event.component_type == ComponentType.SINK
        assert event.component_id == expected_sink_id

    def test_update_sink_emits_event(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_dispatcher,
        project_id,
        sink_id,
        mock_project,
        mock_sink,
        sink_update_data,
    ):
        """Test that updating a sink emits a component change event."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink
        mock_sink_repository.update.return_value = mock_sink

        # Act
        sink_service.update_sink(project_id, sink_id, sink_update_data)

        # Assert
        mock_dispatcher.dispatch.assert_called_once()
        event = mock_dispatcher.dispatch.call_args[0][0]
        assert event.component_type == ComponentType.SINK
        assert event.component_id == sink_id

    def test_delete_sink_emits_event(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_dispatcher,
        project_id,
        sink_id,
        mock_project,
        mock_sink,
    ):
        """Test that deleting a sink emits a component change event."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_sink_repository.get_by_id_and_project.return_value = mock_sink

        # Act
        sink_service.delete_sink(project_id, sink_id)

        # Assert
        mock_dispatcher.dispatch.assert_called_once()
        event = mock_dispatcher.dispatch.call_args[0][0]
        assert isinstance(event, ComponentConfigChangeEvent)
        assert event.component_type == ComponentType.SINK

    def test_no_dispatcher_does_not_raise(
        self, mock_session, mock_sink_repository, mock_project_repository, project_id, mock_project, sink_create_data
    ):
        """Test that operations work without a dispatcher."""
        # Arrange
        service = SinkService(
            session=mock_session,
            sink_repository=mock_sink_repository,
            project_repository=mock_project_repository,
            config_change_dispatcher=None,
        )
        mock_project_repository.get_by_id.return_value = mock_project

        # Act & Assert - should not raise
        service.create_sink(project_id, sink_create_data)


class TestEdgeCases:
    """Tests for various edge cases."""

    def test_unknown_constraint_violation(
        self,
        sink_service,
        mock_project_repository,
        mock_sink_repository,
        mock_session,
        project_id,
        mock_project,
        sink_create_data,
    ):
        """Test handling of unknown constraint violations."""
        # Arrange
        mock_project_repository.get_by_id.return_value = mock_project
        mock_session.commit.side_effect = IntegrityError(
            "statement", "params", orig=Exception("some_unknown_constraint_error")
        )

        # Act & Assert
        with pytest.raises(ValueError) as exc_info:
            sink_service.create_sink(project_id, sink_create_data)

        assert "Database constraint violation" in str(exc_info.value)

    def test_disconnect_when_no_connected_sink(self, sink_service, mock_sink_repository, project_id):
        """Test disconnecting when there's no currently active sink."""
        # Arrange
        mock_sink_repository.get_active_in_project.return_value = None

        # Act & Assert - should not raise
        sink_service._disconnect_existing_active_sink(project_id)

        mock_sink_repository.get_active_in_project.assert_called_once_with(project_id)
