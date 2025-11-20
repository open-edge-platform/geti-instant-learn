# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from contextlib import contextmanager

from sqlalchemy.orm import Session

from domain.dispatcher import ConfigChangeDispatcher, ConfigChangeEvent

logger = logging.getLogger(__name__)


class BaseService:
    def __init__(
        self,
        session: Session,
        config_change_dispatcher: ConfigChangeDispatcher | None = None,
    ):
        """Initialize the service"""
        self.session = session
        self._dispatcher = config_change_dispatcher
        self._pending_events: list[ConfigChangeEvent] = []

    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions with automatic event dispatching.

        Usage:
            with self.transaction():
                # perform DB operations
                self._emit_event(...)
                # commit happens automatically at the end
                # events are dispatched after successful commit
        """
        try:
            yield
            self.session.commit()
            self._dispatch_pending_events()
        except Exception:
            self.session.rollback()
            self._pending_events.clear()
            raise

    def _dispatch_pending_events(self) -> None:
        """
        Dispatch and clear queued events (call only after a successful commit).
        """
        if self._dispatcher and self._pending_events:
            for event in self._pending_events:
                try:
                    self._dispatcher.dispatch(event)
                except Exception as exc:
                    logger.error("Failed to dispatch event %s: %s", event, exc, exc_info=True)
        self._pending_events.clear()
