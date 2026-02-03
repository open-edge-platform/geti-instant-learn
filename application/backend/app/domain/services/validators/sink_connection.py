# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from uuid import UUID

from domain.services.schemas.writer import WriterConfig


class SinkConnectionValidator(ABC):
    """Interface for validating sink connectivity."""

    @abstractmethod
    def validate(self, config: WriterConfig, sink_id: UUID | None) -> None:
        """Validate connectivity for a sink configuration."""
        raise NotImplementedError


class NoOpSinkConnectionValidator(SinkConnectionValidator):
    """Default validator that skips connectivity checks."""

    def validate(self, config: WriterConfig, sink_id: UUID | None) -> None:
        pass
