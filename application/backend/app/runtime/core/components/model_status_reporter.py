#  Copyright (C) 2026 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

"""Protocol for reporting the lifecycle of an inference model from inside the runtime.

The ``Processor`` runs on its own worker thread and is the only component that
knows when the underlying ``ModelHandler`` is actually being initialised vs.
ready to predict. It calls into a ``ModelStatusReporter`` to surface those
transitions to higher layers (``PipelineManager`` SSE broadcast, logs, etc.)
without introducing a direct dependency on the application layer.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class ModelStatusReporter(Protocol):
    """Callback contract used by ``Processor`` to broadcast model lifecycle events."""

    def loading_model(self) -> None:
        """Mark the model as being loaded onto its target device."""

    def ready(self) -> None:
        """Mark the model as ready to serve predictions."""

    def idle(self) -> None:
        """Mark the model as idle (passthrough mode or stopped)."""

    def error(self, exc: BaseException) -> None:
        """Mark the model as failed with a captured exception."""
