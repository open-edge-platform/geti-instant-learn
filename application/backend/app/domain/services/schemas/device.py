# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from enum import StrEnum

from pydantic import BaseModel, Field

# Canonical user-facing device string format: `auto`, `cpu`, `xpu`, `cuda`, `xpu-<N>`, `cuda-<N>`.
# Device index: 0, or a positive integer with no leading zeros
DEVICE_STR_PATTERN = r"^(?:auto|cpu|(?:xpu|cuda)(?:-(?:0|[1-9]\d*))?)$"


class DeviceType(StrEnum):
    """Enumeration of device types."""

    AUTO = "auto"
    CPU = "cpu"
    XPU = "xpu"
    CUDA = "cuda"
    NPU = "npu"


_OPENVINO_DEVICE_TYPES = frozenset({DeviceType.CPU, DeviceType.XPU, DeviceType.NPU})


class DeviceInfo(BaseModel):
    """Information about a single runtime device."""

    type: DeviceType = Field(..., description="Device type (auto, cpu, xpu, cuda, or npu)")
    name: str = Field(..., description="Device name")
    memory: int | None = Field(None, description="Total memory available to the device, in bytes (null for CPU)")
    index: int | None = Field(None, description="Device index among those of the same type (null for CPU)")

    @property
    def as_torch(self) -> str:
        """Render as a torch-style device string (e.g. ``cuda:1``, ``cpu``).

        AUTO must be collapsed to a concrete device before calling this.
        """
        if self.type == DeviceType.AUTO:
            raise ValueError("Cannot convert AUTO device to a torch device string.")
        if self.type == DeviceType.NPU:
            raise ValueError("NPU has no torch backend support; run the torch model on the CPU instead.")
        if self.type == DeviceType.CPU:
            return "cpu"
        if self.index is None:
            return self.type.value
        return f"{self.type.value}:{self.index}"

    @property
    def as_key(self) -> str:
        """Render as a stable ``<type>[-<index>]`` selection key used in API and UI."""
        if self.type in (DeviceType.AUTO, DeviceType.CPU) or self.index is None:
            return self.type.value
        return f"{self.type.value}-{self.index}"

    @property
    def as_openvino(self) -> str:
        """Render as an OpenVINO device string (e.g. ``CPU``, ``GPU.1``, ``NPU``).

        OpenVINO addresses Intel discrete and integrated GPUs through the ``GPU`` plugin
        and indexes them as ``GPU.0``, ``GPU.1``, ... NVIDIA CUDA devices are also routed
        through the ``GPU`` plugin when the appropriate OV backend is installed.

        AUTO must be collapsed to a concrete device before calling this: letting the OV
        ``AUTO`` plugin choose would bypass :meth:`DeviceService.resolve_auto`.
        """
        if self.type == DeviceType.AUTO:
            raise ValueError("Cannot convert AUTO device to an OpenVINO device string.")
        if self.type == DeviceType.CPU:
            return "CPU"
        if self.type == DeviceType.NPU:
            return "NPU"
        if self.index is None:
            return "GPU"
        return f"GPU.{self.index}"

    @property
    def uses_openvino(self) -> bool:
        """Return ``True`` when this device is served by the OpenVINO runtime.

        Backend selection is derived from the device alone: CPU, XPU, and NPU run the
        exported OpenVINO model, everything else runs the torch model.

        AUTO must be collapsed to a concrete device before calling this.
        """
        if self.type == DeviceType.AUTO:
            raise ValueError("Cannot determine the backend of an AUTO device.")
        return self.type in _OPENVINO_DEVICE_TYPES
