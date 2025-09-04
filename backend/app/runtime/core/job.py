#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0
import logging
from multiprocessing import Queue, Process
from typing import Type

from backend.app.runtime.core.components import Source, PipelineRunner, Sink
from backend.app.runtime.core.factories import StreamReaderFactory, ProcessorFactory, StreamWriterFactory
from backend.app.runtime.schemas.project import ProjectConfig

logger = logging.getLogger(__name__)


class Job:
    """
    Orchestrates the job lifecycle, including queues and components' management.
    """

    def __init__(self, project_config: ProjectConfig):

        self._in_queue = Queue(maxsize=5)
        self._out_queue = Queue(maxsize=5)
        self._config = project_config
        self._processes: dict[Type, Process] = {}

        self._components = {
            Source: Source(
                self._in_queue,
                StreamReaderFactory.create(project_config.source_config)
            ),
            PipelineRunner: PipelineRunner(
                self._in_queue,
                self._out_queue,
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
        """
        Public getter for the project configuration.
        Returns a deep copy to prevent modification of the internal state.
        """
        return self._config.model_copy(deep=True)

    def start(self) -> None:
        """Starts a process for each component."""
        logger.debug(f"Starting the streaming job for project_id {self._config.project_id}")
        for name, component in self._components.items():
            process = Process(target=component.run)
            process.start()
            self._processes[name] = process
        logger.debug(f"The job has started for project_id {self._config.project_id}")

    def stop(self):
        """Stops all components and their associated processes gracefully."""

        # Stop components in order: source -> inference -> sink
        logger.debug(f"Stopping the streaming job, project_id {self._config.project_id}")

        for component_cls in [Source, PipelineRunner, Sink]:
            component = self._components.get(component_cls)
            if component:
                component.stop()
                process = self._processes.get(component_cls)
                if process and process.is_alive():
                    process.join(timeout=5)

        logger.debug(f"The streaming job has stopped, project_id {self._config.project_id}")

    def update_config(self, new_config: ProjectConfig):
        """Update configuration for specific components."""

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
            new_runner = PipelineRunner(self._in_queue, self._out_queue,
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
        """Restart a specific component with new configuration."""

        self._components[component_cls].stop()
        process = self._processes.get(component_cls)
        if process and process.is_alive():
            process.join(timeout=5)

        self._components[component_cls] = new_component
        process = Process(target=new_component.run)
        process.start()
        self._processes[component_cls] = process
