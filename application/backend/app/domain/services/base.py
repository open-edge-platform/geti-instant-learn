# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Generator
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
    def db_transaction(self) -> Generator[None, None, None]:
        """
        Context manager for database transactions with automatic event dispatching.
        Commit happens automatically at the end, events are dispatched after successful commit.

        Usage:
            with self.db_transaction():
                # perform DB operations
                # create relevant events and add them to self._pending_events
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
                except Exception:
                    logger.exception("Failed to dispatch event %s", event)
        self._pending_events.clear()
