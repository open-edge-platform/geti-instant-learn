# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from runtime.core.components.schemas.processor import ProcessorConfig
from runtime.core.components.schemas.reader import ReaderConfig
from runtime.core.components.schemas.writer import WriterConfig
from runtime.job.dispatcher import ConfigChangeDispatcher, ConfigChangeEvent, ProjectActivationEvent, \
    ComponentConfigChangeEvent
from runtime.job.job import Job
from runtime.job.schemas.project import ProjectConfig


# todo: replace this dummy stub with getting the project config from the project repository/service:
class DummyProjectRepo:

    def get_active_project(self) -> str:
        return 'the_active_project_id'

    def get_project_configuration(self, project_id) -> ProjectConfig:
        return ProjectConfig(
            project_id=project_id,
            processor=ProcessorConfig(),
            reader=ReaderConfig(),
            writer=WriterConfig()
        )


class JobManager:
    """
    Glues the app configuration and runtime layers together by managing the active Job.

    This class listens for configuration change events and translates them into
    lifecycle actions for the running Job instance, such as starting, stopping,
    or performing a live configuration update.
    """

    def __init__(self, event_dispatcher: ConfigChangeDispatcher, project_repo: DummyProjectRepo):
        self._event_dispatcher = event_dispatcher
        self._job: Optional[Job] = None
        self._project_repo = project_repo

    def start(self):
        project_id = self._project_repo.get_active_project()
        project_config = self._project_repo.get_project_configuration(project_id)
        job = Job(project_config)
        job.start()
        self._job = job

        # subscribe for updates
        self._event_dispatcher.subscribe(self.on_config_change)

    def on_config_change(self, event: ConfigChangeEvent):

        new_project_config = self._project_repo.get_project_configuration(event.project_id)

        match event:
            case ProjectActivationEvent() as event:
                if self._job:
                    self._job.stop()
                job = Job(new_project_config)
                job.start()
                self._job = job

            case ComponentConfigChangeEvent() as event:
                if self._job and self._job.config.project_id == event.project_id:
                    self._job.update_config(new_project_config)

    def stop(self):
        # gracefully stop the running pipeline
        if self._job:
            self._job.stop()
