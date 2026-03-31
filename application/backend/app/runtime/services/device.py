# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from domain.services.schemas.device import AvailableDeviceSchema
from domain.services.schemas.project import Device


def _list_xpu_devices() -> list[AvailableDeviceSchema]:
    """Enumerate Intel XPU devices exposed by PyTorch."""
    try:
        import torch

        if not torch.xpu.is_available():
            return []

        return [
            AvailableDeviceSchema(
                backend=Device.XPU,
                device_id=f"xpu:{index}",
                name=torch.xpu.get_device_name(index),
                index=index,
            )
            for index in range(torch.xpu.device_count())
        ]
    except (ImportError, AttributeError, RuntimeError):
        return []


def _list_cuda_devices() -> list[AvailableDeviceSchema]:
    """Enumerate CUDA devices exposed by PyTorch."""
    try:
        import torch

        if not torch.cuda.is_available():
            return []

        return [
            AvailableDeviceSchema(
                backend=Device.CUDA,
                device_id=f"cuda:{index}",
                name=torch.cuda.get_device_name(index),
                index=index,
            )
            for index in range(torch.cuda.device_count())
        ]
    except (ImportError, AttributeError, RuntimeError):
        return []


def list_available_devices() -> list[AvailableDeviceSchema]:
    """List all currently available runtime devices.

    CPU is always available. Intel XPU and NVIDIA CUDA devices are enumerated when detected.
    """
    devices = [
        *_list_xpu_devices(),
        *_list_cuda_devices(),
    ]
    devices.append(
        AvailableDeviceSchema(
            backend=Device.CPU,
            device_id="cpu",
            name="CPU",
        )
    )
    return devices
