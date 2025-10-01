#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from core.components.schemas.processor import ModelConfig
from core.components.schemas.reader import ReaderConfig
from core.components.schemas.writer import WriterConfig
from core.runtime.dispatcher import (
    ComponentConfigChangeEvent,
    ConfigChangeDispatcher,
    ConfigChangeEvent,
    ProjectActivationEvent,
)
from core.runtime.pipeline import Pipeline
from core.runtime.schemas.pipeline import PipelineConfig


# todo: replace this dummy stub with getting the project config from the project repository/service:
class DummyProjectRepo:
    def get_active_project(self) -> str:
        return "the_active_project_id"

    def get_project_configuration(self, project_id: str) -> PipelineConfig:
        return PipelineConfig(
            project_id=project_id, processor=ModelConfig(), reader=ReaderConfig(), writer=WriterConfig()
        )


class PipelineManager:
    """
    Glues the app configuration and runtime layers together by managing the active Pipeline.

    This class listens for configuration change events and translates them into
    lifecycle actions for the running Job instance, such as starting, stopping,
    or performing a live configuration update.
    """

    def __init__(self, event_dispatcher: ConfigChangeDispatcher, project_repo: DummyProjectRepo):
        self._event_dispatcher = event_dispatcher
        self._pipeline: Pipeline | None = None
        self._project_repo = project_repo

    def start(self) -> None:
        project_id = self._project_repo.get_active_project()
        project_config = self._project_repo.get_project_configuration(project_id)
        pipeline = Pipeline(project_config)
        pipeline.start()
        self._pipeline = pipeline

        # subscribe for updates
        self._event_dispatcher.subscribe(self.on_config_change)

    def on_config_change(self, event: ConfigChangeEvent) -> None:
        new_project_config = self._project_repo.get_project_configuration(event.project_id)

        match event:
            case ProjectActivationEvent() as event:
                if self._pipeline:
                    self._pipeline.stop()
                job = Pipeline(new_project_config)
                job.start()
                self._pipeline = job

            case ComponentConfigChangeEvent() as event:
                if self._pipeline and self._pipeline.config.project_id == event.project_id:
                    self._pipeline.update_config(new_project_config)

    def stop(self) -> None:
        # gracefully stop the running pipeline
        if self._pipeline:
            self._pipeline.stop()
