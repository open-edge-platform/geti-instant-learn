# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Sequence
from uuid import uuid4

import pytest

from domain.db.models import ProjectDB, SinkDB
from domain.repositories.sink import SinkRepository
from domain.services.schemas.writer import WriterType


@pytest.fixture
def repository(fxt_session):
    """Create a SinkRepository instance."""
    return SinkRepository(fxt_session)


@pytest.fixture
def clean_after(request, fxt_clean_table):
    request.addfinalizer(lambda: fxt_clean_table(SinkDB))
    request.addfinalizer(lambda: fxt_clean_table(ProjectDB))


@pytest.fixture
def sample_project_id(fxt_session):
    """Generate a sample project ID and create the project in the database."""
    project_id = uuid4()
    project = ProjectDB(
        id=project_id,
        name=f"Test Project {project_id}",
        active=False,
    )
    fxt_session.add(project)
    fxt_session.commit()
    return project_id


@pytest.fixture
def other_project_id(fxt_session):
    """Generate another project ID and create the project in the database."""
    project_id = uuid4()
    project = ProjectDB(
        id=project_id,
        name=f"Other Project {project_id}",
        active=False,
    )
    fxt_session.add(project)
    fxt_session.commit()
    return project_id


@pytest.fixture
def sample_sink(sample_project_id):
    """Create a sample SinkDB entity."""
    return SinkDB(
        id=uuid4(),
        project_id=sample_project_id,
        active=False,
        config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
    )


class TestSinkRepositoryAdd:
    """Tests for the add method."""

    def test_add_sink(self, repository, sample_sink, fxt_session):
        """Test adding a sink successfully."""
        repository.add(sample_sink)
        fxt_session.commit()

        result = repository.get_by_id(sample_sink.id)
        assert result is not None
        assert result.id == sample_sink.id
        assert result.project_id == sample_sink.project_id


class TestSinkRepositoryGetById:
    """Tests for the get_by_id method."""

    def test_get_by_id(self, repository, sample_sink, fxt_session):
        """Test retrieving a sink by ID."""
        fxt_session.add(sample_sink)
        fxt_session.commit()

        result = repository.get_by_id(sample_sink.id)
        assert result is not None
        assert result.id == sample_sink.id
        assert result.project_id == sample_sink.project_id

    def test_get_by_id_not_found(self, repository):
        """Test retrieving a non-existent sink."""
        non_existent_id = uuid4()
        result = repository.get_by_id(non_existent_id)
        assert result is None


class TestSinkRepositoryGetByIdAndProject:
    """Tests for the get_by_id_and_project method."""

    def test_get_by_id_and_project(self, repository, sample_sink, fxt_session):
        """Test retrieving a sink by ID and project ID."""
        fxt_session.add(sample_sink)
        fxt_session.commit()

        result = repository.get_by_id_and_project(sample_sink.id, sample_sink.project_id)
        assert result is not None
        assert result.id == sample_sink.id
        assert result.project_id == sample_sink.project_id

    def test_get_by_id_and_project_wrong_project(self, repository, sample_sink, other_project_id, fxt_session):
        """Test retrieving a sink with wrong project ID returns None."""
        fxt_session.add(sample_sink)
        fxt_session.commit()

        result = repository.get_by_id_and_project(sample_sink.id, other_project_id)
        assert result is None

    def test_get_by_id_and_project_not_found(self, repository, sample_project_id):
        """Test retrieving a non-existent sink."""
        non_existent_id = uuid4()
        result = repository.get_by_id_and_project(non_existent_id, sample_project_id)
        assert result is None


class TestSinkRepositoryGetAllByProject:
    """Tests for the list_all_by_project method."""

    def test_get_all_by_project(self, repository, sample_project_id, fxt_session):
        """Test retrieving all sinks for a project."""
        sink1 = SinkDB(
            id=uuid4(),
            project_id=sample_project_id,
            active=False,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )
        sink2 = SinkDB(
            id=uuid4(),
            project_id=sample_project_id,
            active=False,
            config={"sink_type": "kafka", "bootstrap_servers": "localhost:9092"},
        )

        fxt_session.add_all([sink1, sink2])
        fxt_session.commit()

        results = repository.list_all_by_project(sample_project_id)
        assert len(results) == 2
        assert {r.id for r in results} == {sink1.id, sink2.id}

    def test_get_all_by_project_empty(self, repository, sample_project_id):
        """Test retrieving sinks when project has none."""
        results = repository.list_all_by_project(sample_project_id)
        assert len(results) == 0
        assert isinstance(results, Sequence)

    def test_get_all_by_project_filters_other_projects(
        self, repository, sample_project_id, other_project_id, fxt_session
    ):
        """Test that only sinks from specified project are returned."""
        sink1 = SinkDB(
            id=uuid4(),
            project_id=sample_project_id,
            active=False,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )
        sink2 = SinkDB(
            id=uuid4(),
            project_id=other_project_id,
            active=False,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )

        fxt_session.add_all([sink1, sink2])
        fxt_session.commit()

        results = repository.list_all_by_project(sample_project_id)
        assert len(results) == 1
        assert results[0].id == sink1.id


class TestSinkRepositoryDelete:
    """Tests for the delete method."""

    def test_delete(self, repository, sample_sink, fxt_session):
        """Test deleting a sink."""
        fxt_session.add(sample_sink)
        fxt_session.commit()

        result = repository.delete(sample_sink.id)
        fxt_session.commit()

        assert result is True
        assert fxt_session.get(SinkDB, sample_sink.id) is None

    def test_delete_multiple_sinks(self, repository, sample_project_id, fxt_session):
        """Test deleting one sink while keeping others."""
        sink1 = SinkDB(
            id=uuid4(),
            project_id=sample_project_id,
            active=False,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )
        sink2 = SinkDB(
            id=uuid4(),
            project_id=sample_project_id,
            active=False,
            config={"sink_type": "kafka", "broker_host": "localhost"},
        )

        fxt_session.add_all([sink1, sink2])
        fxt_session.commit()

        repository.delete(sink1.id)
        fxt_session.commit()

        assert fxt_session.get(SinkDB, sink1.id) is None
        assert fxt_session.get(SinkDB, sink2.id) is not None


class TestSinkRepositoryGetConnectedInProject:
    """Tests for the get_active_in_project method."""

    def test_get_active_in_project(self, repository, fxt_session, clean_after):
        """Test retrieving the active sink in an active project."""
        project = ProjectDB(name=f"Active Project {uuid4()}", active=True)
        fxt_session.add(project)
        fxt_session.commit()

        # Use different sink types to avoid unique constraint violation
        active_sink = SinkDB(
            id=uuid4(),
            project_id=project.id,
            active=True,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )
        deactivated_sink = SinkDB(
            id=uuid4(),
            project_id=project.id,
            active=False,
            config={"sink_type": "kafka", "broker_host": "localhost"},
        )

        fxt_session.add_all([active_sink, deactivated_sink])
        fxt_session.commit()

        result = repository.get_active_in_project(project.id)
        assert result is not None
        assert result.id == active_sink.id
        assert result.active is True

    def test_get_active_in_project_none_active(self, repository, fxt_session, clean_after):
        """Test when no sinks are active."""
        project = ProjectDB(name=f"Active Project No Sinks {uuid4()}", active=True)
        fxt_session.add(project)
        fxt_session.commit()

        sink = SinkDB(
            id=uuid4(),
            project_id=project.id,
            active=False,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )

        fxt_session.add(sink)
        fxt_session.commit()

        result = repository.get_active_in_project(project.id)
        assert result is None

    def test_get_active_in_project_empty_project(self, repository, fxt_session, clean_after):
        """Test when project has no sinks at all."""
        project = ProjectDB(name=f"Active Project Empty {uuid4()}", active=True)
        fxt_session.add(project)
        fxt_session.commit()

        result = repository.get_active_in_project(project.id)
        assert result is None

    def test_get_active_in_project_filters_other_projects(
        self, repository, sample_project_id, fxt_session, clean_after
    ):
        """Test that active sinks from other projects are not returned."""
        other_project = ProjectDB(name=f"Other Active Project {uuid4()}", active=True)
        fxt_session.add(other_project)
        fxt_session.commit()

        connected_sink_other = SinkDB(
            id=uuid4(),
            project_id=other_project.id,
            active=True,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )
        disconnected_sink = SinkDB(
            id=uuid4(),
            project_id=sample_project_id,
            active=False,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )

        fxt_session.add_all([connected_sink_other, disconnected_sink])
        fxt_session.commit()

        result = repository.get_active_in_project(sample_project_id)
        assert result is None


class TestSinkRepositoryExceptions:
    def test_unique_active_constraint(self, repository, fxt_session, clean_after):
        """Test that only one sink can be active per project (business rule)."""
        project = ProjectDB(name=f"Unique Active Project {uuid4()}", active=True)
        fxt_session.add(project)
        fxt_session.commit()

        # Create first active sink
        sink1 = SinkDB(
            id=uuid4(),
            project_id=project.id,
            active=True,
            config={"sink_type": WriterType.MQTT, "broker_host": "localhost"},
        )
        fxt_session.add(sink1)
        fxt_session.commit()

        # Verify first sink is active
        active = repository.get_active_in_project(project.id)
        assert active is not None
        assert active.id == sink1.id

        # Try to create second active sink - should violate constraint
        sink2 = SinkDB(
            id=uuid4(),
            project_id=project.id,
            active=True,
            config={"sink_type": "kafka", "broker_host": "localhost"},
        )
        fxt_session.add(sink2)

        # This should raise IntegrityError due to UNIQUE(project_id, active) constraint
        with pytest.raises(Exception):  # IntegrityError from SQLAlchemy
            fxt_session.commit()

        fxt_session.rollback()
