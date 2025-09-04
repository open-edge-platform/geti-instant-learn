#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
from typing import Optional

from backend.app.runtime.core.job import Job
from backend.app.runtime.schemas.pipeline import PipelineConfig
from backend.app.runtime.schemas.project import ProjectConfig
from backend.app.runtime.schemas.sink import SinkConfig
from backend.app.runtime.schemas.source import SourceConfig
from backend.app.services.config_events import ConfigChangeDispatcher, ConfigChangeEvent, ProjectActivationEvent, \
    ComponentConfigChangeEvent


class JobManager:
    """Glues the app configuration and runtime layers together"""

    def __init__(self, event_dispatcher: ConfigChangeDispatcher):
        self._event_dispatcher = event_dispatcher
        self._job: Optional[Job] = None

    def start(self):
        # retrieve an active pipeline configuration and run it.
        # subscribe for updates
        self._event_dispatcher.subscribe(self.on_config_change)

    def on_config_change(self, event: ConfigChangeEvent):
        match event:
            case ProjectActivationEvent() as event:
                if self._job:
                    self._job.stop()

                job = Job(self._get_project_configuration(event.project_id))
                job.start()
                self._job = job

            case ComponentConfigChangeEvent() as event:
                if self._job and self._job.config.project_id == event.project_id:
                    self._job.update_config(self._get_project_configuration(event.project_id))

    # todo: replace this dummy stub with getting the project config from the project repository/service:
    @staticmethod
    def _get_project_configuration(project_id) -> ProjectConfig:
        return ProjectConfig(
            project_id=project_id,
            source_config=SourceConfig(),
            pipeline_config=PipelineConfig(),
            sink_config=SinkConfig()
        )

    def stop(self):
        # gracefully stop the running pipeline
        if self._job:
            self._job.stop()
