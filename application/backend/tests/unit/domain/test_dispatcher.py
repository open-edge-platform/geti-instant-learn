#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import Mock
from uuid import uuid4

import pytest

from domain.dispatcher import ComponentConfigChangeEvent, ComponentType, ConfigChangeDispatcher, ProjectActivationEvent


@pytest.fixture
def dispatcher() -> ConfigChangeDispatcher:
    return ConfigChangeDispatcher()


@pytest.fixture
def mock_listener():
    return Mock()


@pytest.fixture
def sink_config_change_event():
    return ComponentConfigChangeEvent(project_id=uuid4(), component_type=ComponentType.SINK, component_id=uuid4())


@pytest.fixture
def source_config_change_event():
    return ComponentConfigChangeEvent(project_id=uuid4(), component_type=ComponentType.SOURCE, component_id=uuid4())


@pytest.fixture
def processor_config_change_event():
    return ComponentConfigChangeEvent(project_id=uuid4(), component_type=ComponentType.PROCESSOR, component_id=uuid4())


@pytest.fixture
def project_activation_event():
    return ProjectActivationEvent(project_id=uuid4())


class TestConfigChangeDispatcher:
    def test_subscribe_adds_listener(self, dispatcher, mock_listener):
        dispatcher.subscribe(mock_listener)

        assert mock_listener in dispatcher._listeners
        assert len(dispatcher._listeners) == 1

    def test_dispatch_sink_event_synchronously(self, dispatcher, mock_listener, sink_config_change_event):
        """Test that sink events are dispatched synchronously."""
        dispatcher.subscribe(mock_listener)
        dispatcher.dispatch(sink_config_change_event)

        # No need to wait - should be called immediately
        mock_listener.assert_called_once_with(sink_config_change_event)

    def test_dispatch_source_event_synchronously(self, dispatcher, mock_listener, source_config_change_event):
        """Test that source events are dispatched synchronously."""
        dispatcher.subscribe(mock_listener)
        dispatcher.dispatch(source_config_change_event)

        # No need to wait - should be called immediately
        mock_listener.assert_called_once_with(source_config_change_event)

    def test_dispatch_processor_event_asynchronously(self, dispatcher, mock_listener, processor_config_change_event):
        """Test that processor events are dispatched asynchronously."""
        dispatcher.subscribe(mock_listener)
        dispatcher.dispatch(processor_config_change_event)

        # Need to wait for async executor
        time.sleep(0.1)
        mock_listener.assert_called_once_with(processor_config_change_event)

    def test_dispatch_project_event_asynchronously(self, dispatcher, mock_listener, project_activation_event):
        """Test that project events are dispatched asynchronously."""
        dispatcher.subscribe(mock_listener)
        dispatcher.dispatch(project_activation_event)

        # Need to wait for async executor
        time.sleep(0.1)
        mock_listener.assert_called_once_with(project_activation_event)

    def test_dispatch_calls_multiple_listeners(self, dispatcher, sink_config_change_event):
        mock_listener_1 = Mock()
        mock_listener_2 = Mock()

        dispatcher.subscribe(mock_listener_1)
        dispatcher.subscribe(mock_listener_2)

        dispatcher.dispatch(sink_config_change_event)

        # Sink events are synchronous - no wait needed
        mock_listener_1.assert_called_once_with(sink_config_change_event)
        mock_listener_2.assert_called_once_with(sink_config_change_event)
