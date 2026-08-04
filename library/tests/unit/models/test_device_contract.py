# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Tests for the public model device constructor contract."""

import inspect
from pathlib import Path
from unittest.mock import patch

import pytest

from instantlearn.device import DeviceInfo, DeviceType
from instantlearn.models.dinotxt import DinoTxtZeroShotClassification
from instantlearn.models.efficient_sam3 import EfficientSAM3
from instantlearn.models.grounded_sam import GroundedSAM
from instantlearn.models.matcher import Matcher, MatcherOpenVINO
from instantlearn.models.model_card import ModelCard, RuntimeCapability
from instantlearn.models.openvino_base import OpenVINOModel
from instantlearn.models.per_dino import PerDino, PerDinoOpenVINO
from instantlearn.models.sam3 import SAM3, SAM3OpenVINO
from instantlearn.models.soft_matcher import SoftMatcher, SoftMatcherOpenVINO
from instantlearn.models.torch_base import ExportConfig, TorchModel
from instantlearn.utils.constants import Backend

_CARD = ModelCard(
    name="Test model",
    family="test",
    description="test",
    prompt_types=frozenset(),
    shot_modes=frozenset(),
    exportable_to=frozenset(),
    supported_runtimes=frozenset(
        {
            RuntimeCapability(Backend.TORCH, frozenset({DeviceType.CPU, DeviceType.GPU})),
            RuntimeCapability(Backend.OPENVINO, frozenset({DeviceType.CPU, DeviceType.NPU})),
        },
    ),
)


class _TorchTestModel(TorchModel):
    @classmethod
    def card(cls) -> ModelCard:
        return _CARD

    def fit(self, _reference) -> None:  # noqa: ANN001
        pass

    def predict(self, _target) -> list:  # noqa: ANN001
        return []

    def to_openvino(self, export_path: Path | None = None, _config: ExportConfig | None = None) -> Path:
        return export_path or Path("model.xml")


class _OpenVINOTestModel(OpenVINOModel):
    @classmethod
    def card(cls) -> ModelCard:
        return _CARD

    def fit(self, _reference) -> None:  # noqa: ANN001
        pass

    def predict(self, _target) -> list:  # noqa: ANN001
        return []


def _device(
    device_type: DeviceType,
    name: str,
    runtime_ids: dict[Backend, str],
    index: int | None = None,
    memory: int | None = None,
) -> DeviceInfo:
    return DeviceInfo(
        type=device_type,
        name=name,
        memory=memory,
        index=index,
        runtime_ids=runtime_ids,
    )


def test_torch_base_stores_device_info_and_runtime_id() -> None:
    """Torch models retain the physical device and exact Torch identifier."""
    device = _device(DeviceType.GPU, "NVIDIA", {Backend.TORCH: "cuda:1"}, index=1)

    model = _TorchTestModel(device=device)

    assert model.device_info is device
    assert model.device == "cuda:1"


def test_torch_base_auto_selects_supported_device() -> None:
    """A missing Torch device is selected from the model card capabilities."""
    cpu = _device(DeviceType.CPU, "CPU", {Backend.TORCH: "cpu"})
    gpu = _device(DeviceType.GPU, "GPU", {Backend.TORCH: "xpu:0"}, index=0)

    with patch("instantlearn.device.resolver.enumerate_system_devices", return_value=[cpu, gpu]):
        model = _TorchTestModel()

    assert model.device_info is gpu
    assert model.device == "xpu:0"


def test_openvino_base_stores_device_info_and_runtime_id(tmp_path: Path) -> None:
    """OpenVINO models retain the physical device and exact OpenVINO identifier."""
    device = _device(DeviceType.NPU, "NPU", {Backend.OPENVINO: "NPU"}, index=0)

    with patch("instantlearn.models.openvino_base.ov.Core"):
        model = _OpenVINOTestModel(model_dir=tmp_path, device=device)

    assert model.device_info is device
    assert model.device == "NPU"


def test_openvino_base_auto_selects_supported_device(tmp_path: Path) -> None:
    """A missing OpenVINO device is selected from model card capabilities."""
    cpu = _device(DeviceType.CPU, "CPU", {Backend.OPENVINO: "CPU"})
    npu = _device(DeviceType.NPU, "NPU", {Backend.OPENVINO: "NPU"}, index=0)

    with (
        patch("instantlearn.device.resolver.enumerate_system_devices", return_value=[cpu, npu]),
        patch("instantlearn.models.openvino_base.ov.Core"),
    ):
        model = _OpenVINOTestModel(model_dir=tmp_path)

    assert model.device_info is npu
    assert model.device == "NPU"


@pytest.mark.parametrize(
    "model_class",
    [
        DinoTxtZeroShotClassification,
        EfficientSAM3,
        GroundedSAM,
        Matcher,
        PerDino,
        SAM3,
        SoftMatcher,
        MatcherOpenVINO,
        PerDinoOpenVINO,
        SAM3OpenVINO,
        SoftMatcherOpenVINO,
    ],
)
def test_public_model_device_signature_is_strict(model_class: type) -> None:
    """Every public card-backed model accepts only DeviceInfo or None."""
    parameter = inspect.signature(model_class.__init__).parameters["device"]

    assert parameter.default is None
    assert parameter.annotation in {DeviceInfo | None, "DeviceInfo | None"}
