# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from queue import Queue
from threading import Thread
from typing import Type

from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.factories.components import DefaultComponentFactory
from runtime.core.components.pipeline import PipelineRunner
from runtime.core.components.sink import Sink
from runtime.core.components.source import Source
from runtime.job.schemas.project import ProjectConfig

logger = logging.getLogger(__name__)


class Job:
    """
    Orchestrates the job components lifecycle and runtime.
    """

    def __init__(self, project_conf: ProjectConfig, broadcaster=FrameBroadcaster(),
                 component_factory=DefaultComponentFactory()):

        self._broadcaster = broadcaster
        self.factory = component_factory
        self._in_queue = Queue(maxsize=5)
        self._config = project_conf
        self._threads: dict[Type, Thread] = {}

        self._components = {
            Source: self.factory.create_source(self._in_queue, project_conf.reader),
            PipelineRunner: self.factory.create_pipeline(self._in_queue, self._broadcaster, project_conf.processor),
            Sink: self.factory.create_sink(self._broadcaster, project_conf.writer)
        }
        logger.debug(f"A streaming job created for a project config: {project_conf}")

    @property
    def config(self) -> ProjectConfig:
        return self._config.model_copy(deep=True)

    def start(self) -> None:
        logger.debug(f"Starting the streaming job for project_id {self._config.project_id}")
        for name, component in self._components.items():
            thread = Thread(target=component)
            thread.start()
            self._threads[name] = thread
        logger.debug(f"The job has started for project_id {self._config.project_id}")

    def stop(self):
        # Stop components in order: source -> inference -> sink
        logger.debug(f"Stopping the streaming job, project_id {self._config.project_id}")

        for component_cls in [Source, PipelineRunner, Sink]:
            component = self._components.get(component_cls)
            if component:
                component.stop()
                thread = self._threads.get(component_cls)
                if thread and thread.is_alive():
                    thread.join(timeout=5)

        logger.debug(f"The streaming job has stopped, project_id {self._config.project_id}")

    def update_config(self, new_config: ProjectConfig):
        logger.debug(f"Updating the streaming job configuration for project_id {self._config.project_id}")

        if new_config.source_config != self._config.source_config:
            logger.info(f"Source configuration changed for project_id {self._config.project_id}. "
                        f"old config: {self._config.source_config}, new config: {new_config.source_config}. "
                        f"Restarting component.")
            new_source = self.factory.create_source(self._in_queue, new_config.reader)
            self._restart_component(Source, new_source)
            logger.info(f"Source configuration has been refreshed for project_id {self._config.project_id}.")

        if new_config.pipeline_config != self._config.pipeline_config:
            logger.info(f"Pipeline configuration changed for project_id {self._config.project_id}. "
                        f"old config: {self._config.pipeline_config}, new config: {new_config.pipeline_config}. "
                        f"Restarting component.")
            new_runner = self.factory.create_pipeline(self._in_queue, self._broadcaster, new_config.processor)
            self._restart_component(PipelineRunner, new_runner)
            logger.info(f"Pipeline configuration has been refreshed for project_id {self._config.project_id}.")

        if new_config.sink_config != self._config.sink_config:
            logger.info(f"Sink configuration changed for project_id {self._config.project_id}. "
                        f"old config: {self._config.sink_config}, new config: {new_config.sink_config}. "
                        f"Restarting component.")
            new_sink = self.factory.create_sink(self._broadcaster, new_config.writer)
            self._restart_component(Sink, new_sink)
            logger.info(f"Sink configuration has been refreshed for project_id {self._config.project_id}.")

        self._config = new_config

    def _restart_component(self, component_cls, new_component) -> None:
        self._components[component_cls].stop()
        thread = self._threads.get(component_cls)
        if thread and thread.is_alive():
            thread.join(timeout=5)

        self._components[component_cls] = new_component
        thread = Thread(target=new_component)
        thread.start()
        self._threads[component_cls] = thread
