#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
import logging
from queue import Queue
from threading import Thread
from typing import Type

from backend.app.runtime.core.components import Source, PipelineRunner, Sink, FrameBroadcaster
from backend.app.runtime.core.factories import StreamReaderFactory, ProcessorFactory, StreamWriterFactory
from backend.app.runtime.schemas.project import ProjectConfig

logger = logging.getLogger(__name__)


class Job:
    """
    Orchestrates the job components lifecycle and runtime.
    """

    def __init__(self, project_config: ProjectConfig):

        self._broadcaster = FrameBroadcaster()
        self._in_queue = Queue(maxsize=5)
        self._out_queue = self._broadcaster.register()

        self._config = project_config
        self._threads: dict[Type, Thread] = {}

        self._components = {
            Source: Source(
                self._in_queue,
                StreamReaderFactory.create(project_config.source_config)
            ),
            PipelineRunner: PipelineRunner(
                self._in_queue,
                self._broadcaster,
                ProcessorFactory.create(project_config.pipeline_config)
            ),
            Sink: Sink(
                self._out_queue,
                StreamWriterFactory.create(project_config.sink_config)
            )
        }
        logger.debug(f"A streaming job created for a project config: {project_config}")

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
            new_source = Source(self._in_queue, StreamReaderFactory.create(new_config.source_config))
            self._restart_component(Source, new_source)
            logger.info(f"Source configuration has been refreshed for project_id {self._config.project_id}.")

        if new_config.pipeline_config != self._config.pipeline_config:
            logger.info(f"Pipeline configuration changed for project_id {self._config.project_id}. "
                        f"old config: {self._config.pipeline_config}, new config: {new_config.pipeline_config}. "
                        f"Restarting component.")
            new_runner = PipelineRunner(self._in_queue, self._broadcaster,
                                        ProcessorFactory.create(new_config.pipeline_config))
            self._restart_component(PipelineRunner, new_runner)
            logger.info(f"Pipeline configuration has been refreshed for project_id {self._config.project_id}.")

        if new_config.sink_config != self._config.sink_config:
            logger.info(f"Sink configuration changed for project_id {self._config.project_id}. "
                        f"old config: {self._config.sink_config}, new config: {new_config.sink_config}. "
                        f"Restarting component.")
            new_sink = Sink(self._out_queue, StreamWriterFactory.create(new_config.sink_config))
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
