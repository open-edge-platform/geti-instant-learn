# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock
from uuid import uuid4

import pytest

from domain.dispatcher import (
    ComponentConfigChangeEvent,
    ComponentType,
    ConfigChangeDispatcher,
    ProjectActivationEvent,
    ProjectDeactivationEvent,
)
from domain.services.base import BaseService


@pytest.fixture
def mock_session():
    return Mock()


@pytest.fixture
def mock_dispatcher():
    return Mock(spec=ConfigChangeDispatcher)


class TestDispatchPendingEventsCoalescing:
    """Tests for BaseService._dispatch_pending_events event coalescing."""

    def test_duplicate_processor_events_coalesced_to_last(self, mock_session, mock_dispatcher):
        """When two PROCESSOR events for the same project are queued, only the last is dispatched."""
        svc = BaseService(mock_session, config_change_dispatcher=mock_dispatcher)
        pid = uuid4()
        old_id = uuid4()
        new_id = uuid4()

        svc._pending_events = [
            ComponentConfigChangeEvent(project_id=pid, component_type=ComponentType.PROCESSOR, component_id=old_id),
            ComponentConfigChangeEvent(project_id=pid, component_type=ComponentType.PROCESSOR, component_id=new_id),
        ]
        svc._dispatch_pending_events()

        mock_dispatcher.dispatch.assert_called_once()
        dispatched = mock_dispatcher.dispatch.call_args[0][0]
        assert dispatched.component_id == new_id

    def test_different_component_types_not_coalesced(self, mock_session, mock_dispatcher):
        """Events for different component types on the same project are preserved independently."""
        svc = BaseService(mock_session, config_change_dispatcher=mock_dispatcher)
        pid = uuid4()

        source_event = ComponentConfigChangeEvent(
            project_id=pid, component_type=ComponentType.SOURCE, component_id=uuid4()
        )
        processor_event = ComponentConfigChangeEvent(
            project_id=pid, component_type=ComponentType.PROCESSOR, component_id=uuid4()
        )

        svc._pending_events = [source_event, processor_event]
        svc._dispatch_pending_events()

        assert mock_dispatcher.dispatch.call_count == 2

    def test_different_projects_not_coalesced(self, mock_session, mock_dispatcher):
        """PROCESSOR events for different projects are dispatched independently."""
        svc = BaseService(mock_session, config_change_dispatcher=mock_dispatcher)

        event_a = ComponentConfigChangeEvent(
            project_id=uuid4(), component_type=ComponentType.PROCESSOR, component_id=uuid4()
        )
        event_b = ComponentConfigChangeEvent(
            project_id=uuid4(), component_type=ComponentType.PROCESSOR, component_id=uuid4()
        )

        svc._pending_events = [event_a, event_b]
        svc._dispatch_pending_events()

        assert mock_dispatcher.dispatch.call_count == 2

    def test_lifecycle_events_not_coalesced_with_component_events(self, mock_session, mock_dispatcher):
        """ProjectActivation/Deactivation events are dispatched independently from component events."""
        svc = BaseService(mock_session, config_change_dispatcher=mock_dispatcher)
        pid = uuid4()

        activation = ProjectActivationEvent(project_id=pid)
        deactivation = ProjectDeactivationEvent(project_id=pid)
        processor = ComponentConfigChangeEvent(
            project_id=pid, component_type=ComponentType.PROCESSOR, component_id=uuid4()
        )

        svc._pending_events = [activation, deactivation, processor]
        svc._dispatch_pending_events()

        assert mock_dispatcher.dispatch.call_count == 3

    def test_pending_events_cleared_after_dispatch(self, mock_session, mock_dispatcher):
        """Pending events list is cleared even after coalescing."""
        svc = BaseService(mock_session, config_change_dispatcher=mock_dispatcher)
        pid = uuid4()

        svc._pending_events = [
            ComponentConfigChangeEvent(project_id=pid, component_type=ComponentType.PROCESSOR, component_id=uuid4()),
            ComponentConfigChangeEvent(project_id=pid, component_type=ComponentType.PROCESSOR, component_id=uuid4()),
        ]
        svc._dispatch_pending_events()

        assert svc._pending_events == []

    def test_no_dispatcher_clears_events_without_dispatch(self, mock_session):
        """Without a dispatcher, pending events are silently cleared."""
        svc = BaseService(mock_session, config_change_dispatcher=None)
        svc._pending_events = [
            ComponentConfigChangeEvent(
                project_id=uuid4(), component_type=ComponentType.PROCESSOR, component_id=uuid4()
            ),
        ]
        svc._dispatch_pending_events()

        assert svc._pending_events == []

    def test_dispatch_error_does_not_prevent_other_events(self, mock_session, mock_dispatcher):
        """If dispatching one event fails, remaining events are still dispatched."""
        svc = BaseService(mock_session, config_change_dispatcher=mock_dispatcher)

        source_event = ComponentConfigChangeEvent(
            project_id=uuid4(), component_type=ComponentType.SOURCE, component_id=uuid4()
        )
        sink_event = ComponentConfigChangeEvent(
            project_id=uuid4(), component_type=ComponentType.SINK, component_id=uuid4()
        )

        mock_dispatcher.dispatch.side_effect = [Exception("boom"), None]

        svc._pending_events = [source_event, sink_event]
        svc._dispatch_pending_events()  # should not raise

        assert mock_dispatcher.dispatch.call_count == 2
        assert svc._pending_events == []
