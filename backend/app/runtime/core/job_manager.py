#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

from typing import Protocol, Optional, Union

from pydantic import BaseModel

from backend.app.runtime.core.components import Job


class PipelineActivationEvent(BaseModel):
    """Event fired when a new pipeline should be activated."""
    pipeline_id: str


class ComponentConfigChangeEvent(BaseModel):
    """Event fired when a component of the active pipeline changes."""
    pipeline_id: str
    component_type: str
    component_id: str


ConfigChangeEvent = Union[PipelineActivationEvent, ComponentConfigChangeEvent]


class ConfigChangeListener(Protocol):
    """Defines the required signature for a configuration change listener."""

    def __call__(self, event: ConfigChangeEvent) -> None:
        ...


class ConfigChangeDispatcher:

    def __init__(self):
        self._listeners: list[ConfigChangeListener] = []

    def subscribe(self, listener: ConfigChangeListener):
        self._listeners.append(listener)

    def dispatch(self, event: ConfigChangeEvent):
        for listener in self._listeners:
            listener(event)


class JobManager:

    def __init__(self, event_dispatcher: ConfigChangeDispatcher):
        self._event_dispatcher = event_dispatcher
        self._job: Optional[Job] = None

    def start(self):
        # retrieve an active pipeline configuration and run it.
        # subscribe for updates
        self._event_dispatcher.subscribe(self.on_config_change)

    def on_config_change(self, event: ConfigChangeEvent):
        pass

    def stop(self):
        # gracefully stop the running pipeline
        if self._job:
            self._job.stop()
