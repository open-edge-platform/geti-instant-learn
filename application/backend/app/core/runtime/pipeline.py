#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Queue
from threading import Thread

from core.components.base import PipelineComponent
from core.components.broadcaster import FrameBroadcaster
from core.components.factories.components import ComponentFactory, DefaultComponentFactory
from core.components.processor import Processor
from core.components.schemas.processor import InputData, OutputData
from core.components.sink import Sink
from core.components.source import Source
from core.runtime.schemas.pipeline import PipelineConfig

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates a multithreaded streaming pipeline and manages its lifecycle.

    This class wires together the core components of a processing job: a Source,
    a Processor, and a Sink. Each component runs in a separate thread,
    communicating through broadcasters to form a processing pipeline:

    Source -> InboundBroadcaster -> Processor -> OutboundBroadcaster -> Sink

    The Pipeline class is responsible for starting, stopping, and gracefully shutting
    down all components. It also handles dynamic configuration updates by
    restarting the relevant component without interrupting the entire job.

    Args:
        pipeline_conf (PipelineConfig): The initial configuration for the job's
            components (reader, processor, and writer).
        inbound_broadcaster (FrameBroadcaster, optional): The broadcaster to distribute
            raw input frames. Defaults to a new FrameBroadcaster instance.
        outbound_broadcaster (FrameBroadcaster, optional): The broadcaster to distribute
            processed frames to sinks. Defaults to a new FrameBroadcaster instance.
        component_factory (DefaultComponentFactory, optional): The factory used
            to create the job's components. Defaults to a new
            DefaultComponentFactory instance.
    """

    def __init__(
        self,
        pipeline_conf: PipelineConfig,
        inbound_broadcaster: FrameBroadcaster[InputData] | None = None,
        outbound_broadcaster: FrameBroadcaster[OutputData] | None = None,
        component_factory: ComponentFactory | None = None,
    ):
        self._outbound_broadcaster = outbound_broadcaster or FrameBroadcaster[OutputData]()
        self._inbound_broadcaster = inbound_broadcaster or FrameBroadcaster[InputData]()
        self._factory = component_factory or DefaultComponentFactory()
        self._config = pipeline_conf
        self._threads: dict[type, Thread] = {}

        self._components: dict[type[PipelineComponent], PipelineComponent] = {
            Source: self._factory.create_source(pipeline_conf.reader, self._inbound_broadcaster),
            Processor: self._factory.create_processor(
                self._inbound_broadcaster, self._outbound_broadcaster, pipeline_conf.processor
            ),
            Sink: self._factory.create_sink(self._outbound_broadcaster, pipeline_conf.writer),
        }
        logger.debug(f"A streaming job created for a project config: {pipeline_conf}")

    def register_webrtc(self) -> Queue:
        return self._outbound_broadcaster.register()

    def unregister_webrtc(self, queue: Queue) -> None:
        return self._outbound_broadcaster.unregister(queue=queue)

    def register_inbound_consumer(self) -> Queue[InputData]:
        """Register a consumer for raw input frames from the source."""
        return self._inbound_broadcaster.register()

    def unregister_inbound_consumer(self, queue: Queue[InputData]) -> None:
        """Unregister a consumer for raw input frames."""
        self._inbound_broadcaster.unregister(queue)

    @property
    def config(self) -> PipelineConfig:
        return self._config.model_copy(deep=True)

    def start(self) -> None:
        logger.debug(f"Starting the streaming job for project_id {self._config.project_id}")
        for name, component in self._components.items():
            thread = Thread(target=component)
            thread.start()
            self._threads[name] = thread
        logger.debug(f"The job has started for project_id {self._config.project_id}")

    def stop(self) -> None:
        # Stop components in order: source -> inference -> sink
        logger.debug(f"Stopping the streaming job, project_id {self._config.project_id}")

        for component_cls in [Source, Processor, Sink]:
            component = self._components.get(component_cls)
            if component:
                component.stop()
                thread = self._threads.get(component_cls)
                if thread and thread.is_alive():
                    thread.join(timeout=5)

        logger.debug(f"The streaming job has stopped, project_id {self._config.project_id}")

    def update_config(self, new_config: PipelineConfig) -> None:
        logger.debug(f"Updating the streaming job configuration for project_id {self._config.project_id}")

        if new_config.reader != self._config.reader:
            logger.info(
                f"Source configuration changed for project_id {self._config.project_id}. "
                f"old config: {self._config.reader}, new config: {new_config.reader}. "
                f"Restarting component."
            )
            new_source = self._factory.create_source(new_config.reader, self._inbound_broadcaster)
            self._restart_component(Source, new_source)
            logger.info(f"Source configuration has been refreshed for project_id {self._config.project_id}.")

        if new_config.processor != self._config.processor:
            logger.info(
                f"Pipeline configuration changed for project_id {self._config.project_id}. "
                f"old config: {self._config.processor}, new config: {new_config.processor}. "
                f"Restarting component."
            )
            new_runner = self._factory.create_processor(
                self._inbound_broadcaster, self._outbound_broadcaster, new_config.processor
            )
            self._restart_component(Processor, new_runner)
            logger.info(f"Processor configuration has been refreshed for project_id {self._config.project_id}.")

        if new_config.writer != self._config.writer:
            logger.info(
                f"Sink configuration changed for project_id {self._config.project_id}. "
                f"old config: {self._config.writer}, new config: {new_config.writer}. "
                f"Restarting component."
            )
            new_sink = self._factory.create_sink(self._outbound_broadcaster, new_config.writer)
            self._restart_component(Sink, new_sink)
            logger.info(f"Sink configuration has been refreshed for project_id {self._config.project_id}.")

        self._config = new_config

    def _restart_component(self, component_cls: type[PipelineComponent], new_component: PipelineComponent) -> None:
        self._components[component_cls].stop()
        thread = self._threads.get(component_cls)
        if thread and thread.is_alive():
            thread.join(timeout=5)

        self._components[component_cls] = new_component
        thread = Thread(target=new_component)
        thread.start()
        self._threads[component_cls] = thread
