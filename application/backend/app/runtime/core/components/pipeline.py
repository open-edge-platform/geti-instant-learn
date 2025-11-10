#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import logging
from queue import Queue
from threading import Thread
from uuid import UUID

from runtime.core.components.base import PipelineComponent
from runtime.core.components.broadcaster import FrameBroadcaster
from runtime.core.components.processor import Processor
from runtime.core.components.schemas.processor import InputData, OutputData
from runtime.core.components.sink import Sink
from runtime.core.components.source import Source

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Orchestrates a multithreaded streaming pipeline and manages component lifecycle.

    This class manages the lifecycle of three core pipeline components (Source, Processor, Sink),
    each running in a separate thread and communicating through broadcasters:

    Source -> InboundBroadcaster -> Processor -> OutboundBroadcaster -> Sink

    The Pipeline is responsible for:
    - Starting/stopping all components
    - Gracefully replacing individual components at runtime
    - Managing thread lifecycle and broadcaster communication

    The caller (typically PipelineManager) is responsible for:
    - Configuration management and comparison
    - Creating component instances
    - Deciding when to update components

    Args:
        project_id (UUID): The project ID associated with this pipeline.
        source (Source): The source component for reading input frames.
        processor (Processor): The processor component for inference.
        sink (Sink): The sink component for writing output.
        inbound_broadcaster (FrameBroadcaster[InputData], optional): Broadcaster for raw frames.
            Defaults to a new instance.
        outbound_broadcaster (FrameBroadcaster[OutputData], optional): Broadcaster for processed frames.
            Defaults to a new instance.
    """

    def __init__(
        self,
        project_id: UUID,
        source: Source,
        processor: Processor,
        sink: Sink,
        inbound_broadcaster: FrameBroadcaster[InputData] = FrameBroadcaster[InputData](),
        outbound_broadcaster: FrameBroadcaster[OutputData] = FrameBroadcaster[OutputData](),
    ):
        # todo: remove project id from the pipeline as it is the application impl details
        self._project_id = project_id
        self._inbound_broadcaster = inbound_broadcaster
        self._outbound_broadcaster = outbound_broadcaster
        self._threads: dict[type[PipelineComponent], Thread] = {}

        self._components: dict[type[PipelineComponent], PipelineComponent] = {
            Source: source,
            Processor: processor,
            Sink: sink,
        }
        logger.debug(f"Pipeline created for project_id={project_id}")

    @property
    def project_id(self) -> UUID:
        """Get the project ID associated with this pipeline."""
        return self._project_id

    def register_webrtc(self) -> Queue:
        """Register a WebRTC consumer for processed output frames."""
        return self._outbound_broadcaster.register()

    def unregister_webrtc(self, queue: Queue) -> None:
        """Unregister a WebRTC consumer."""
        return self._outbound_broadcaster.unregister(queue=queue)

    def register_inbound_consumer(self) -> Queue[InputData]:
        """Register a consumer for raw input frames from the source."""
        return self._inbound_broadcaster.register()

    def unregister_inbound_consumer(self, queue: Queue[InputData]) -> None:
        """Unregister a consumer for raw input frames."""
        self._inbound_broadcaster.unregister(queue)

    def start(self) -> None:
        logger.debug(f"Starting pipeline for project_id={self._project_id}")
        for component_cls, component in self._components.items():
            thread = Thread(target=component, daemon=False)
            thread.start()
            self._threads[component_cls] = thread
        logger.debug(f"Pipeline started for project_id={self._project_id}")

    def stop(self) -> None:
        logger.debug(f"Stopping pipeline for project_id={self._project_id}")

        for component_cls in [Source, Processor, Sink]:
            component = self._components.get(component_cls)
            if component:
                component.stop()
                thread = self._threads.get(component_cls)
                if thread and thread.is_alive():
                    thread.join(timeout=5)

        logger.debug(f"Pipeline stopped for project_id={self._project_id}")

    def update_component(self, new_component: PipelineComponent) -> None:
        """
        A method to replace a component with a new one.

        Handles the stop/replace/start lifecycle for a single component.

        Args:
            new_component: The new component instance.
        """
        component_cls = new_component.__class__

        if component_cls not in self._components:
            raise ValueError(f"Unknown component type: {component_cls}")

        # Stop the current component
        current_component = self._components.get(component_cls)
        if current_component:
            current_component.stop()
            thread = self._threads.get(component_cls)
            if thread and thread.is_alive():
                thread.join(timeout=5)

        self._components[component_cls] = new_component
        thread = Thread(target=new_component, daemon=False)
        thread.start()
        self._threads[component_cls] = thread
