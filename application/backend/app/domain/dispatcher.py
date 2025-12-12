# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import Protocol
from uuid import UUID

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ProjectActivationEvent(BaseModel):
    """Event fired when a new pipeline should be activated."""

    project_id: UUID


class ProjectDeactivationEvent(BaseModel):
    """Event fired when the current pipeline should be deactivated."""

    project_id: UUID


class ComponentType(StrEnum):
    """Enum for configurable types of pipeline components."""

    SOURCE = "source"
    PROCESSOR = "processor"
    SINK = "sink"


class ComponentConfigChangeEvent(BaseModel):
    """Event fired when a component of the active pipeline changes."""

    project_id: UUID
    component_type: ComponentType
    component_id: UUID


ConfigChangeEvent = ProjectActivationEvent | ProjectDeactivationEvent | ComponentConfigChangeEvent


class ConfigChangeListener(Protocol):
    """
    Defines a protocol for consumers that need to react to project activation or component configuration changes.
    """

    def __call__(self, event: ConfigChangeEvent) -> None: ...


class ConfigChangeDispatcher:
    """
    Manages and dispatches configuration change events to subscribed listeners.

    This class allows components to subscribe to configuration changes and be
    notified when an event occurs.
    Events are dispatched asynchronously to avoid blocking HTTP responses.
    """

    def __init__(self, max_workers: int = 2):
        self._listeners: list[ConfigChangeListener] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="config-dispatcher")

    def subscribe(self, listener: ConfigChangeListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def dispatch(self, event: ConfigChangeEvent) -> None:
        """
        Dispatch a configuration change event to all subscribed listeners.
        Events for sinks and sources are dispatched synchronously to avoid race conditions for the UI,
        the rest is dispatched asynchronously.

        """
        if isinstance(event, ComponentConfigChangeEvent) and event.component_type in (
            ComponentType.SOURCE,
            ComponentType.SINK,
        ):
            for listener in self._listeners:
                self._safe_notify(listener, event)
        else:
            for listener in self._listeners:
                self._executor.submit(self._safe_notify, listener, event)

    def _safe_notify(self, listener: ConfigChangeListener, event: ConfigChangeEvent) -> None:
        try:
            listener(event)
        except Exception:
            logger.exception(
                "Listener failed to process event: listener=%s, event=%s",
                listener.__class__.__name__,
                event.__class__.__name__,
            )

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the executor. Call during application shutdown."""
        self._executor.shutdown(wait=wait)
