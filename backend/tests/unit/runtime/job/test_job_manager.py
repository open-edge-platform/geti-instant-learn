#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from runtime.job.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher, ProjectActivationEvent
from runtime.job.job_manager import DummyProjectRepo, JobManager
from runtime.job.schemas.project import ProjectConfig


@pytest.fixture
def mock_dispatcher():
    return Mock(spec=ConfigChangeDispatcher)


@pytest.fixture
def mock_project_repo():
    return Mock(spec=DummyProjectRepo)


class TestJobManager:
    @patch("runtime.job.job_manager.Job")
    def test_start_initializes_and_starts_job(self, mock_job_class, mock_project_repo, mock_dispatcher):
        mock_project_repo.get_active_project.return_value = "active-project-01"
        mock_config = Mock(spec=ProjectConfig)
        mock_project_repo.get_project_configuration.return_value = mock_config
        mock_job_instance = mock_job_class.return_value

        job_manager = JobManager(mock_dispatcher, mock_project_repo)

        job_manager.start()

        mock_project_repo.get_active_project.assert_called_once()
        mock_project_repo.get_project_configuration.assert_called_once_with("active-project-01")

        mock_job_class.assert_called_once_with(mock_config)
        mock_job_instance.start.assert_called_once()
        mock_dispatcher.subscribe.assert_called_once_with(job_manager.on_config_change)

    @patch("runtime.job.job_manager.Job")
    def test_on_config_change_project_activation_stops_old_job_and_starts_new(
        self, mock_job_class, mock_project_repo, mock_dispatcher
    ):
        old_job = Mock()
        job_manager = JobManager(mock_dispatcher, mock_project_repo)

        job_manager._job = old_job

        new_config = Mock(spec=ProjectConfig)
        mock_project_repo.get_project_configuration.return_value = new_config
        new_job_instance = mock_job_class.return_value
        event = ProjectActivationEvent(project_id="new-project-02")

        job_manager.on_config_change(event)

        old_job.stop.assert_called_once()
        mock_project_repo.get_project_configuration.assert_called_once_with("new-project-02")
        mock_job_class.assert_called_once_with(new_config)
        new_job_instance.start.assert_called_once()
        assert job_manager._job == new_job_instance

    def test_on_config_change_component_update_for_matching_project(self, mock_project_repo, mock_dispatcher):
        running_job = Mock()
        running_job.config.project_id = "project-123"

        job_manager = JobManager(mock_dispatcher, mock_project_repo)
        job_manager._job = running_job

        new_config = Mock(spec=ProjectConfig)
        mock_project_repo.get_project_configuration.return_value = new_config
        event = ComponentConfigChangeEvent(
            project_id="project-123", component_type="processor", component_id="nn-model"
        )

        job_manager.on_config_change(event)

        mock_project_repo.get_project_configuration.assert_called_once_with("project-123")
        running_job.update_config.assert_called_once_with(new_config)
        running_job.stop.assert_not_called()

    def test_on_config_change_component_update_ignores_mismatched_project(self, mock_project_repo, mock_dispatcher):
        running_job = Mock()
        running_job.config.project_id = "project-123"
        job_manager = JobManager(mock_dispatcher, mock_project_repo)
        job_manager._job = running_job

        event = ComponentConfigChangeEvent(
            project_id="PROJECT-567", component_type="processor", component_id="nn-model"
        )

        job_manager.on_config_change(event)

        mock_project_repo.get_project_configuration.assert_called_once_with("PROJECT-567")
        running_job.update_config.assert_not_called()
        running_job.stop.assert_not_called()

    def test_stop_stops_running_job(self, mock_project_repo, mock_dispatcher):
        running_job = Mock()
        job_manager = JobManager(mock_dispatcher, mock_project_repo)
        job_manager._job = running_job

        job_manager.stop()

        running_job.stop.assert_called_once()
