#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0


import unittest
from unittest.mock import Mock

from runtime.job.dispatcher import ComponentConfigChangeEvent, ConfigChangeDispatcher


class TestConfigChangeDispatcher(unittest.TestCase):
    def setUp(self):
        self.dispatcher = ConfigChangeDispatcher()

    def test_subscribe_adds_listener(self):
        mock_listener = Mock()
        self.dispatcher.subscribe(mock_listener)

        self.assertIn(mock_listener, self.dispatcher._listeners)
        self.assertEqual(len(self.dispatcher._listeners), 1)

    def test_dispatch_calls_multiple_listeners(self):
        mock_listener_1 = Mock()
        mock_listener_2 = Mock()

        self.dispatcher.subscribe(mock_listener_1)
        self.dispatcher.subscribe(mock_listener_2)

        event = ComponentConfigChangeEvent(project_id="project-beta", component_type="source", component_id="camera-01")
        self.dispatcher.dispatch(event)

        mock_listener_1.assert_called_once_with(event)
        mock_listener_2.assert_called_once_with(event)
