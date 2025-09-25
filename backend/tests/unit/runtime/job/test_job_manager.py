#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, patch

from runtime.job.dispatcher import ConfigChangeDispatcher, ProjectActivationEvent, ComponentConfigChangeEvent
from runtime.job.job_manager import JobManager, DummyProjectRepo
from runtime.job.schemas.project import ProjectConfig


class TestJobManager(unittest.TestCase):

    def setUp(self):
        self.mock_dispatcher = Mock(spec=ConfigChangeDispatcher)
        self.mock_repo = Mock(spec=DummyProjectRepo)
        self.job_manager = JobManager(self.mock_dispatcher, self.mock_repo)

    @patch('runtime.job.job_manager.Job')
    def test_start_initializes_and_starts_job(self, mock_job_class):
        self.mock_repo.get_active_project.return_value = "active-project-01"
        mock_config = Mock(spec=ProjectConfig)
        self.mock_repo.get_project_configuration.return_value = mock_config
        mock_job_instance = mock_job_class.return_value

        self.job_manager.start()

        self.mock_repo.get_active_project.assert_called_once()
        self.mock_repo.get_project_configuration.assert_called_once_with("active-project-01")

        mock_job_class.assert_called_once_with(mock_config)
        mock_job_instance.start.assert_called_once()
        self.mock_dispatcher.subscribe.assert_called_once_with(self.job_manager.on_config_change)

    @patch('runtime.job.job_manager.Job')
    def test_on_config_change_project_activation_stops_old_job_and_starts_new(self, mock_job_class):
        old_job = Mock()
        self.job_manager._job = old_job

        new_config = Mock(spec=ProjectConfig)
        self.mock_repo.get_project_configuration.return_value = new_config
        new_job_instance = mock_job_class.return_value
        event = ProjectActivationEvent(project_id="new-project-02")

        self.job_manager.on_config_change(event)

        old_job.stop.assert_called_once()
        self.mock_repo.get_project_configuration.assert_called_once_with("new-project-02")
        mock_job_class.assert_called_once_with(new_config)
        new_job_instance.start.assert_called_once()
        self.assertEqual(self.job_manager._job, new_job_instance)

    def test_on_config_change_component_update_for_matching_project(self):
        running_job = Mock()
        running_job.config.project_id = "project-123"
        self.job_manager._job = running_job
        new_config = Mock(spec=ProjectConfig)
        self.mock_repo.get_project_configuration.return_value = new_config
        event = ComponentConfigChangeEvent(
            project_id="project-123", component_type="processor", component_id="nn-model"
        )

        self.job_manager.on_config_change(event)

        self.mock_repo.get_project_configuration.assert_called_once_with("project-123")
        running_job.update_config.assert_called_once_with(new_config)
        running_job.stop.assert_not_called()

    def test_on_config_change_component_update_ignores_mismatched_project(self):
        running_job = Mock()
        running_job.config.project_id = "project-123"
        self.job_manager._job = running_job
        event = ComponentConfigChangeEvent(
            project_id="PROJECT-567", component_type="processor", component_id="nn-model"
        )

        self.job_manager.on_config_change(event)

        self.mock_repo.get_project_configuration.assert_called_once_with("PROJECT-567")
        running_job.update_config.assert_not_called()
        running_job.stop.assert_not_called()

    def test_stop_stops_running_job(self):
        running_job = Mock()
        self.job_manager._job = running_job

        self.job_manager.stop()

        running_job.stop.assert_called_once()
