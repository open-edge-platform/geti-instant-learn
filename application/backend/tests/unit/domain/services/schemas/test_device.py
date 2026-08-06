# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import pytest

from domain.services.schemas.device import DeviceInfo, DeviceType


def _device(type_: DeviceType, index: int | None = None) -> DeviceInfo:
    return DeviceInfo(type=type_, name=type_.value.upper(), memory=None, index=index)


class TestUsesOpenvino:
    @pytest.mark.parametrize(
        ("type_", "expected"),
        [
            (DeviceType.CPU, True),
            (DeviceType.XPU, True),
            (DeviceType.NPU, True),
            (DeviceType.CUDA, False),
        ],
    )
    def test_backend_is_derived_from_device_type(self, type_, expected):
        assert _device(type_).uses_openvino is expected

    def test_auto_must_be_collapsed_first(self):
        with pytest.raises(ValueError, match="AUTO"):
            _ = _device(DeviceType.AUTO).uses_openvino


class TestAsOpenvino:
    @pytest.mark.parametrize(
        ("type_", "index", "expected"),
        [
            (DeviceType.CPU, None, "CPU"),
            (DeviceType.NPU, None, "NPU"),
            (DeviceType.XPU, None, "GPU"),
            (DeviceType.XPU, 1, "GPU.1"),
        ],
    )
    def test_renders_openvino_device_string(self, type_, index, expected):
        assert _device(type_, index=index).as_openvino == expected

    def test_auto_raises_instead_of_delegating_to_the_ov_auto_plugin(self):
        with pytest.raises(ValueError, match="AUTO"):
            _ = _device(DeviceType.AUTO).as_openvino


class TestAsTorch:
    @pytest.mark.parametrize(
        ("type_", "index", "expected"),
        [
            (DeviceType.CPU, None, "cpu"),
            (DeviceType.XPU, 0, "xpu:0"),
            (DeviceType.CUDA, 1, "cuda:1"),
        ],
    )
    def test_renders_torch_device_string(self, type_, index, expected):
        assert _device(type_, index=index).as_torch == expected

    @pytest.mark.parametrize("type_", [DeviceType.AUTO, DeviceType.NPU])
    def test_types_without_a_torch_backend_raise(self, type_):
        with pytest.raises(ValueError):
            _ = _device(type_).as_torch
