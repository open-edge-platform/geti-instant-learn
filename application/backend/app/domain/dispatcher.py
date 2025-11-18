# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Protocol
from uuid import UUID

from pydantic import BaseModel


class ProjectActivationEvent(BaseModel):
    """Event fired when a new pipeline should be activated."""

    project_id: UUID


class ProjectDeactivationEvent(BaseModel):
    """Event fired when the current pipeline should be deactivated."""

    project_id: UUID


class ComponentConfigChangeEvent(BaseModel):
    """Event fired when a component of the active pipeline changes."""

    project_id: UUID
    component_type: str
    component_id: str


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
    """

    def __init__(self):
        self._listeners: list[ConfigChangeListener] = []

    def subscribe(self, listener: ConfigChangeListener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def dispatch(self, event: ConfigChangeEvent) -> None:
        for listener in self._listeners:
            listener(event)
