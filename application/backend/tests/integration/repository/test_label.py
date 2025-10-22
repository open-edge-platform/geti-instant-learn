# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import pytest

from db.models import LabelDB
from repositories.label import LabelRepository

PROJECT_ID = uuid4()
OTHER_PROJECT_ID = uuid4()


@pytest.fixture
def label_repository(fxt_session):
    """Create a LabelRepository instance."""
    return LabelRepository(session=fxt_session)


@pytest.fixture
def clean_after(request, fxt_clean_table):
    request.addfinalizer(lambda: fxt_clean_table(LabelDB))


@pytest.fixture
def project_id(fxt_session, fxt_clean_table):
    """Create a project and return its ID."""
    from db.models import ProjectDB

    project = ProjectDB(id=PROJECT_ID, name="test_project", active=False)
    other_project = ProjectDB(id=OTHER_PROJECT_ID, name="other_project", active=False)
    fxt_session.add(project)
    fxt_session.add(other_project)
    fxt_session.commit()
    yield project.id

    # Cleanup will happen via clean_after fixture for LabelDB
    # Then clean ProjectDB
    fxt_clean_table(ProjectDB)


@pytest.fixture
def sample_label(project_id):
    """Create a sample label."""
    return LabelDB(id=uuid4(), name="test_label", project_id=project_id, color="#FF0000")


class TestLabelRepository:
    """Test suite for LabelRepository."""

    def test_add_label(self, label_repository, sample_label, fxt_session, clean_after):
        """Test adding a label to the session."""
        label_repository.add(sample_label)
        fxt_session.commit()

        result = label_repository.get_by_id(project_id=sample_label.project_id, label_id=sample_label.id)
        assert result is not None
        assert result.name == "test_label"
        assert result.color is not None
        assert result.id == sample_label.id

    def test_get_by_id_exists(self, label_repository, sample_label, project_id, fxt_session, clean_after):
        """Test retrieving an existing label by ID."""
        fxt_session.add(sample_label)
        fxt_session.commit()

        result = label_repository.get_by_id(project_id, sample_label.id)
        assert result is not None
        assert result.id == sample_label.id
        assert result.name == "test_label"

    def test_get_by_id_not_exists(self, label_repository, project_id, clean_after):
        """Test retrieving a non-existent label by ID."""
        result = label_repository.get_by_id(project_id, uuid4())
        assert result is None

    def test_get_by_id_wrong_project(self, label_repository, sample_label, fxt_session, clean_after):
        """Test retrieving a label with wrong project ID."""
        label_repository.add(sample_label)
        fxt_session.commit()

        result = label_repository.get_by_id(uuid4(), sample_label.id)
        assert result is None

    def test_get_all_empty(self, label_repository, project_id, clean_after):
        """Test retrieving all labels when none exist."""
        result = label_repository.get_all(project_id)
        assert len(result) == 0

    def test_get_all_multiple_labels(self, label_repository, project_id, fxt_session, clean_after):
        """Test retrieving all labels."""
        labels = [LabelDB(id=uuid4(), name=f"label_{i}", project_id=project_id, color="#000000") for i in range(3)]
        for label in labels:
            label_repository.add(label)
        fxt_session.commit()

        result = label_repository.get_all(project_id)
        assert len(result) == 3

    def test_get_all_filters_by_project(self, label_repository, project_id, fxt_session, clean_after):
        """Test that get_all only returns labels for the specified project."""

        label_repository.add(LabelDB(id=uuid4(), name="label_1", project_id=PROJECT_ID, color="#000000"))
        label_repository.add(LabelDB(id=uuid4(), name="label_2", project_id=OTHER_PROJECT_ID, color="#000000"))
        fxt_session.commit()

        result = label_repository.get_all(PROJECT_ID)
        assert len(result) == 1
        assert result[0].name == "label_1"

    def test_exists_by_name_true(self, label_repository, sample_label, fxt_session, clean_after):
        """Test checking existence by name when label exists."""
        label_repository.add(sample_label)
        fxt_session.commit()

        assert label_repository.exists_by_name("test_label") is True

    def test_exists_by_name_false(self, label_repository, clean_after):
        """Test checking existence by name when label doesn't exist."""
        assert label_repository.exists_by_name("nonexistent") is False

    def test_exists_by_id_true(self, label_repository, sample_label, fxt_session, clean_after):
        """Test checking existence by ID when label exists."""
        label_repository.add(sample_label)
        fxt_session.commit()

        assert label_repository.exists_by_id(sample_label.id) is True

    def test_exists_by_id_false(self, label_repository, clean_after):
        """Test checking existence by ID when label doesn't exist."""
        assert label_repository.exists_by_id(uuid4()) is False

    def test_delete_label(self, label_repository, sample_label, project_id, fxt_session, clean_after):
        """Test deleting a label."""
        label_repository.add(sample_label)
        fxt_session.commit()

        label_repository.delete(project_id, sample_label)
        fxt_session.commit()

        result = label_repository.get_by_id(project_id=project_id, label_id=sample_label.id)
        assert result is None

    def test_delete_wrong_project(self, label_repository, project_id, sample_label, fxt_session, clean_after):
        """Test that delete doesn't remove label from wrong project."""
        label_repository.add(sample_label)
        fxt_session.commit()

        label_repository.delete(OTHER_PROJECT_ID, sample_label)
        fxt_session.commit()

        result = label_repository.get_by_id(project_id=project_id, label_id=sample_label.id)
        assert result is not None

    def test_get_paginated_empty(self, label_repository, project_id, clean_after):
        """Test pagination with no labels."""
        labels, total_count = label_repository.get_paginated(project_id)
        assert len(labels) == 0
        assert total_count == 0

    def test_get_paginated_first_page(self, label_repository, project_id, fxt_session, clean_after):
        """Test getting the first page of results."""
        for i in range(25):
            label_repository.add(LabelDB(id=uuid4(), name=f"label_{i:02d}", project_id=project_id, color="#000000"))
        fxt_session.commit()

        labels, total_count = label_repository.get_paginated(project_id, offset=0, limit=20)
        assert len(labels) == 20
        assert total_count == 25

    def test_get_paginated_second_page(self, label_repository, project_id, fxt_session, clean_after):
        """Test getting the second page of results."""
        for i in range(25):
            label_repository.add(LabelDB(id=uuid4(), name=f"label_{i:02d}", project_id=project_id, color="#000000"))
        fxt_session.commit()

        labels, total_count = label_repository.get_paginated(project_id, offset=20, limit=20)
        assert len(labels) == 5
        assert total_count == 25

    def test_get_paginated_ordered_by_name(self, label_repository, project_id, fxt_session, clean_after):
        """Test that pagination returns results ordered by name."""
        names = ["zebra", "alpha", "beta"]
        for name in names:
            label_repository.add(LabelDB(id=uuid4(), name=name, project_id=project_id, color="#000000"))
        fxt_session.commit()

        labels, _ = label_repository.get_paginated(project_id, offset=0, limit=10)
        assert labels[0].name == "alpha"
        assert labels[1].name == "beta"
        assert labels[2].name == "zebra"

    def test_get_paginated_filters_by_project(self, label_repository, project_id, fxt_session, clean_after):
        """Test that pagination only returns labels for the specified project."""

        label_repository.add(LabelDB(id=uuid4(), name="label_1", project_id=PROJECT_ID, color="#000000"))
        label_repository.add(LabelDB(id=uuid4(), name="label_2", project_id=OTHER_PROJECT_ID, color="#000000"))
        fxt_session.commit()

        labels, total_count = label_repository.get_paginated(project_id)
        assert len(labels) == 1
        assert total_count == 1
