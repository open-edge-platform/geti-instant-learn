#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from core.runtime.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher, ProjectActivationEvent
from core.runtime.pipeline_manager import DummyProjectRepo, PipelineManager
from core.runtime.schemas.pipeline import PipelineConfig


@pytest.fixture
def mock_dispatcher():
    return Mock(spec=ConfigChangeDispatcher)


@pytest.fixture
def mock_project_repo():
    return Mock(spec=DummyProjectRepo)


class TestJobManager:
    @patch("core.runtime.pipeline_manager.Pipeline")
    def test_start_initializes_and_starts_job(self, mock_pipeline_class, mock_project_repo, mock_dispatcher):
        mock_project_repo.get_active_project.return_value = "active-project-01"
        mock_config = Mock(spec=PipelineConfig)
        mock_project_repo.get_project_configuration.return_value = mock_config
        mock_pipeline_instance = mock_pipeline_class.return_value

        pipeline_manager = PipelineManager(mock_dispatcher, mock_project_repo)

        pipeline_manager.start()

        mock_project_repo.get_active_project.assert_called_once()
        mock_project_repo.get_project_configuration.assert_called_once_with("active-project-01")

        mock_pipeline_class.assert_called_once_with(mock_config)
        mock_pipeline_instance.start.assert_called_once()
        mock_dispatcher.subscribe.assert_called_once_with(pipeline_manager.on_config_change)

    @patch("core.runtime.pipeline_manager.Pipeline")
    def test_on_config_change_project_activation_stops_old_job_and_starts_new(
        self, mock_pipeline_class, mock_project_repo, mock_dispatcher
    ):
        old_pipeline = Mock()
        pipeline_manager = PipelineManager(mock_dispatcher, mock_project_repo)

        pipeline_manager._pipeline = old_pipeline

        new_config = Mock(spec=PipelineConfig)
        mock_project_repo.get_project_configuration.return_value = new_config
        new_pipeline_instance = mock_pipeline_class.return_value
        event = ProjectActivationEvent(project_id="new-project-02")

        pipeline_manager.on_config_change(event)

        old_pipeline.stop.assert_called_once()
        mock_project_repo.get_project_configuration.assert_called_once_with("new-project-02")
        mock_pipeline_class.assert_called_once_with(new_config)
        new_pipeline_instance.start.assert_called_once()
        assert pipeline_manager._pipeline == new_pipeline_instance

    def test_on_config_change_component_update_for_matching_project(self, mock_project_repo, mock_dispatcher):
        running_pipeline = Mock()
        running_pipeline.config.project_id = "project-123"

        pipeline_manager = PipelineManager(mock_dispatcher, mock_project_repo)
        pipeline_manager._pipeline = running_pipeline

        new_config = Mock(spec=PipelineConfig)
        mock_project_repo.get_project_configuration.return_value = new_config
        event = ComponentConfigChangeEvent(
            project_id="project-123", component_type="processor", component_id="nn-model"
        )

        pipeline_manager.on_config_change(event)

        mock_project_repo.get_project_configuration.assert_called_once_with("project-123")
        running_pipeline.update_config.assert_called_once_with(new_config)
        running_pipeline.stop.assert_not_called()

    def test_on_config_change_component_update_ignores_mismatched_project(self, mock_project_repo, mock_dispatcher):
        running_pipeline = Mock()
        running_pipeline.config.project_id = "project-123"
        pipeline_manager = PipelineManager(mock_dispatcher, mock_project_repo)
        pipeline_manager._pipeline = running_pipeline

        event = ComponentConfigChangeEvent(
            project_id="PROJECT-567", component_type="processor", component_id="nn-model"
        )

        pipeline_manager.on_config_change(event)

        mock_project_repo.get_project_configuration.assert_called_once_with("PROJECT-567")
        running_pipeline.update_config.assert_not_called()
        running_pipeline.stop.assert_not_called()

    def test_stop_stops_running_job(self, mock_project_repo, mock_dispatcher):
        running_pipeline = Mock()
        pipeline_manager = PipelineManager(mock_dispatcher, mock_project_repo)
        pipeline_manager._pipeline = running_pipeline

        pipeline_manager.stop()

        running_pipeline.stop.assert_called_once()
