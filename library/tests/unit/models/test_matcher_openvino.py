# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MatcherOpenVINO (mocked OpenVINO runtime, no real IR files)."""

import json
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from instantlearn.data.base.sample import Category, Sample
from instantlearn.models.matcher import MatcherOpenVINO
from instantlearn.utils.constants import Backend

INPUT_SIZE = 28
PATCH_SIZE = 14


def _make_sample(h: int = 28, w: int = 28, category: str = "cat") -> Sample:
    """Create a minimal in-memory sample with random pixels."""
    return Sample(
        image=np.random.default_rng().integers(0, 255, (h, w, 3), dtype=np.uint8),
        masks=np.ones((1, h, w), dtype=np.uint8),
        categories=[Category(id=0, label=category)],
    )


def _write_dummy_ir(tmp_path: Path, *, with_metadata: bool = True) -> None:
    """Create dummy IR/metadata files so construction path checks pass."""
    if with_metadata:
        (tmp_path / "metadata.json").write_text(
            json.dumps(
                {
                    "input_size": INPUT_SIZE,
                    "patch_size": PATCH_SIZE,
                    "categories": {"0": "cat"},
                },
            ),
        )
    (tmp_path / "matcher.xml").touch()
    (tmp_path / "matcher.bin").touch()


@pytest.fixture
def mock_model(tmp_path: Path) -> Iterator[MatcherOpenVINO]:
    """Yield a MatcherOpenVINO instance with OpenVINO fully mocked."""
    _write_dummy_ir(tmp_path)
    with patch("openvino.Core") as mock_core:
        mock_core.return_value.compile_model.return_value = MagicMock()
        yield MatcherOpenVINO(model_dir=str(tmp_path), device="CPU")


class TestMatcherOpenVINO:
    """Tests for the Matcher OpenVINO sibling."""

    def test_card_delegates_to_torch(self) -> None:
        """card() delegates to the torch sibling."""
        card = MatcherOpenVINO.card()
        assert card.name == "Matcher"
        assert card.family == "matcher"
        assert Backend.OPENVINO in card.exportable_to

    def test_backend(self, mock_model: MatcherOpenVINO) -> None:
        """Backend property reports OpenVINO."""
        assert mock_model.backend == Backend.OPENVINO

    def test_metadata_loaded(self, mock_model: MatcherOpenVINO) -> None:
        """Metadata (input/patch size + categories) is read on construction."""
        assert mock_model.input_size == INPUT_SIZE
        assert mock_model.patch_size == PATCH_SIZE
        assert mock_model._category_names == {0: "cat"}  # noqa: SLF001

    def test_fit_raises_not_implemented(self, mock_model: MatcherOpenVINO) -> None:
        """fit() is unsupported: references are baked at export time."""
        with pytest.raises(NotImplementedError):
            mock_model.fit(_make_sample())

    def test_missing_metadata_raises(self, tmp_path: Path) -> None:
        """Missing metadata.json raises FileNotFoundError."""
        _write_dummy_ir(tmp_path, with_metadata=False)
        with patch("openvino.Core"), pytest.raises(FileNotFoundError):
            MatcherOpenVINO(model_dir=str(tmp_path), device="CPU")

    def test_missing_ir_raises(self, tmp_path: Path) -> None:
        """Missing matcher.xml raises FileNotFoundError."""
        (tmp_path / "metadata.json").write_text(
            json.dumps({"input_size": INPUT_SIZE, "patch_size": PATCH_SIZE, "categories": {}}),
        )
        with patch("openvino.Core"), pytest.raises(FileNotFoundError):
            MatcherOpenVINO(model_dir=str(tmp_path), device="CPU")

