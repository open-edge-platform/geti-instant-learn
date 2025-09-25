# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Union, Protocol

from pydantic import BaseModel


class ProjectActivationEvent(BaseModel):
    """Event fired when a new pipeline should be activated."""
    project_id: str


class ComponentConfigChangeEvent(BaseModel):
    """Event fired when a component of the active pipeline changes."""
    project_id: str
    component_type: str
    component_id: str


ConfigChangeEvent = Union[ProjectActivationEvent, ComponentConfigChangeEvent]


class ConfigChangeListener(Protocol):

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
