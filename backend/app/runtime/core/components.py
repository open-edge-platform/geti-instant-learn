#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
import time
from multiprocessing import Queue, Process
from queue import Empty
from typing import Optional, Type

from backend.app.runtime.core.base import Processor, StreamReader, StreamWriter, PipelineComponent
from .factories import StreamReaderFactory, ProcessorFactory, StreamWriterFactory
from .types import ConfigDict

logger = logging.getLogger(__name__)


class Source(PipelineComponent):
    """Reads from a StreamReader and puts the data into the provided queue."""

    def __init__(self, in_queue: Queue, stream_reader: StreamReader):
        super().__init__()
        self._reader = stream_reader
        self._in_queue = in_queue

    def run(self) -> None:

        logger.debug(f"Entering a source loop: {self._reader.get_config()}")
        with self._reader:
            while not self._stop_event.is_set():
                data = self._reader.read()
                if data is None:
                    time.sleep(0.1)
                    continue
                self._in_queue.put(data)

            logger.debug(f"Existing the source loop: {self._reader.get_config()}")


class Sink(PipelineComponent):
    """Gets data from a queue and writes it using a StreamWriter."""

    def __init__(self, out_queue: Queue, stream_writer: StreamWriter):
        super().__init__()
        self._writer = stream_writer
        self._out_queue = out_queue

    def run(self) -> None:
        logger.debug(f"Enetring a sink loop: {self._writer.get_config()}")
        with self._writer:
            while not self._stop_event.is_set():
                try:
                    data = self._out_queue.get(timeout=0.1)
                    self._writer.write(data)
                except Empty:
                    continue
            logger.debug(f"Existing the sink loop: {self._writer.get_config()}")


class TaskRunner(PipelineComponent):
    """A component and a composite processor impl that runs a sequence of processors."""

    def __init__(self, in_queue: Queue, out_queue: Queue, processors: Processor):

        super().__init__()
        self._processor = processors
        self._in_queue = in_queue
        self._out_queue = out_queue

    def run(self) -> None:
        logger.debug(f"Entering a task runner loop: {self._processor.get_config()}")
        while not self._stop_event.is_set():
            try:
                data = self._in_queue.get(timeout=0.1)
                processed_data = self._processor.process(data)
                self._out_queue.put(processed_data)
            except Empty:
                continue
        logger.debug(f"Exiting the task runner loop: {self._processor.get_config()}")


class Pipeline:
    """
    Orchestrates the pipeline lifecycle, including queue and components' management.
    """

    def __init__(self,
                 pipeline_id: str,
                 source_conf: Optional[ConfigDict] = None,
                 task_runner_conf: Optional[ConfigDict] = None,
                 sink_conf: Optional[ConfigDict] = None):

        self._in_queue = Queue(maxsize=5)
        self._out_queue = Queue(maxsize=5)
        self._pipeline_id = pipeline_id
        self._processes: dict[Type, Process] = {}

        self._components = {
            Source: Source(self._in_queue, StreamReaderFactory.create(source_conf)),
            TaskRunner: TaskRunner(self._in_queue, self._out_queue, ProcessorFactory.create(task_runner_conf)),
            Sink: Sink(self._out_queue, StreamWriterFactory.create(sink_conf))
        }
        logger.debug(f"A pipeline created: pipeline id: {pipeline_id} source config: {source_conf}, "
                     f"task_runner config: {task_runner_conf}, sink config: {sink_conf}")

    def start(self) -> None:
        """Starts a process for each component."""
        logger.debug(f"Starting a pipline, pipline_id {self._pipeline_id}")
        for name, component in self._components.items():
            process = Process(target=component.run)
            process.start()
            self._processes[name] = process
        logger.debug(f"The pipline has started, pipline_id {self._pipeline_id}")

    def stop(self):
        """Stops all components and their associated processes gracefully."""

        # Stop components in order: source -> inference -> sink
        logger.debug(f"Stopping the pipeline, pipline_id {self._pipeline_id}")

        for component_cls in [Source, TaskRunner, Sink]:
            component = self._components.get(component_cls)
            if component:
                component.stop()
                process = self._processes.get(component_cls)
                if process and process.is_alive():
                    process.join(timeout=5)

        logger.debug(f"The pipeline has stopped, pipline_id {self._pipeline_id}")

    def update_config(self, source_config: Optional[ConfigDict] = None,
                      inference_config: Optional[ConfigDict] = None,
                      sink_config: Optional[ConfigDict] = None):
        """Update configuration for specific components."""

        logger.debug(f"Updating the pipeline configuration, pipline_id {self._pipeline_id}")
        updates = [
            (
                Source, source_config,
                lambda cfg: Source(self._in_queue, StreamReaderFactory.create(cfg))
            ),
            (
                TaskRunner, inference_config,
                lambda cfg: TaskRunner(self._in_queue, self._out_queue, ProcessorFactory.create(cfg))
            ),
            (
                Sink, sink_config,
                lambda cfg: Sink(self._out_queue, StreamWriterFactory.create(cfg))
            )
        ]

        for component_cls, config, factory in updates:
            if config is not None:
                logger.debug(f"Updating the pipeline component. Component: {component_cls},"
                             f"component config: {config}, pipline_id {self._pipeline_id}")

                self._restart_component(component_cls, factory(config))

                logger.debug(f"The pipeline component has been updated. Component: {component_cls},"
                             f"component config: {config}, pipline_id {self._pipeline_id}")

    def _restart_component(self, component_cls, new_component) -> None:
        """Restart a specific component with new configuration."""

        self._components[component_cls].stop()

        process = self._processes.get(component_cls)
        if process and process.is_alive():
            process.join(timeout=5)

        # Start new component
        self._components[component_cls] = new_component
        process = Process(target=new_component.run)
        process.start()
        self._processes[component_cls] = process
