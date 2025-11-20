# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from sqlalchemy.orm import Session

from domain.dispatcher import ConfigChangeDispatcher, ConfigChangeEvent


class BaseService:
    def __init__(
        self,
        session: Session,
        config_change_dispatcher: ConfigChangeDispatcher | None = None,
        pending_events: list[ConfigChangeEvent] | None = None,
    ):
        """Initialize the service"""
        self.session = session
        self._dispatcher = config_change_dispatcher
        self._pending_events = pending_events
