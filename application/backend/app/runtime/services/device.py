# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import NAMESPACE_URL, UUID, uuid5

from domain.services.schemas.device import AvailableDeviceSchema, Device

DEVICE_NS = uuid5(NAMESPACE_URL, "device")


def _build_id(backend: Device, name: str, index: int | None = None) -> UUID:
    """Build a deterministic UUID for a discovered runtime device."""
    return uuid5(DEVICE_NS, f"{backend}/{index}/{name}")


def _list_xpu_devices() -> list[AvailableDeviceSchema]:
    """List all Intel XPU devices exposed by PyTorch."""
    try:
        import torch

        if not torch.xpu.is_available():
            return []

        devices: list[AvailableDeviceSchema] = []
        for index in range(torch.xpu.device_count()):
            name = torch.xpu.get_device_name(index)
            devices.append(
                AvailableDeviceSchema(
                    id=_build_id(Device.XPU, name, index),
                    backend=Device.XPU,
                    name=name,
                    device_id=f"{Device.XPU}:{index}",
                )
            )
        return devices
    except (ImportError, AttributeError, RuntimeError):
        return []


def _list_cuda_devices() -> list[AvailableDeviceSchema]:
    """List all CUDA devices exposed by PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        devices: list[AvailableDeviceSchema] = []
        for index in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(index)
            devices.append(
                AvailableDeviceSchema(
                    id=_build_id(Device.CUDA, name, index),
                    backend=Device.CUDA,
                    name=name,
                    device_id=f"{Device.CUDA}:{index}",
                )
            )
        return devices
    except (ImportError, AttributeError, RuntimeError):
        return []


def list_available_devices() -> list[AvailableDeviceSchema]:
    """List all currently available runtime devices.

    CPU is always available. Intel XPU and NVIDIA CUDA devices are enumerated when detected.
    """
    return [
        *_list_xpu_devices(),
        *_list_cuda_devices(),
        AvailableDeviceSchema(
            id=_build_id(Device.CPU, "CPU"),
            backend=Device.CPU,
            name="CPU",
            device_id=Device.CPU,
        ),
    ]
