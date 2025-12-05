#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import time
from unittest.mock import Mock
from uuid import uuid4

import pytest

from domain.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher


@pytest.fixture
def dispatcher() -> ConfigChangeDispatcher:
    return ConfigChangeDispatcher()


@pytest.fixture
def mock_listener():
    return Mock()


@pytest.fixture
def config_change_event():
    return ComponentConfigChangeEvent(project_id=uuid4(), component_type="source", component_id="camera-01")


class TestConfigChangeDispatcher:
    def test_subscribe_adds_listener(self, dispatcher, mock_listener):
        dispatcher.subscribe(mock_listener)

        assert mock_listener in dispatcher._listeners
        assert len(dispatcher._listeners) == 1

    def test_dispatch_calls_multiple_listeners(self, dispatcher, config_change_event):
        mock_listener_1 = Mock()
        mock_listener_2 = Mock()

        dispatcher.subscribe(mock_listener_1)
        dispatcher.subscribe(mock_listener_2)

        dispatcher.dispatch(config_change_event)

        # wait for async executor to complete
        time.sleep(0.1)

        mock_listener_1.assert_called_once_with(config_change_event)
        mock_listener_2.assert_called_once_with(config_change_event)
