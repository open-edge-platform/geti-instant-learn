#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from runtime.core.components.scene_change import (
    SceneChangeDetector,
    dhash,
    hamming_distance,
)


def make_frame(seed: int, shape: tuple[int, int, int] = (120, 160, 3)) -> np.ndarray:
    """Create a deterministic random RGB HWC uint8 frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, shape, dtype=np.uint8)


def add_noise(frame: np.ndarray, amplitude: int, seed: int) -> np.ndarray:
    """Return a near-duplicate of frame with small per-pixel noise added."""
    rng = np.random.default_rng(seed)
    noise = rng.integers(-amplitude, amplitude + 1, frame.shape)
    return np.clip(frame.astype(int) + noise, 0, 255).astype(np.uint8)


class TestDhash:
    def test_identical_frames_have_equal_hash(self) -> None:
        frame = make_frame(seed=1)
        assert dhash(frame) == dhash(frame.copy())

    def test_default_hash_uses_64_bits(self) -> None:
        # A 8x8 grid hash must fit in 64 bits.
        value = dhash(make_frame(seed=2))
        assert 0 <= value < (1 << 64)

    def test_hash_size_controls_bit_width(self) -> None:
        # hash_size=4 -> 16-bit hash.
        value = dhash(make_frame(seed=3), hash_size=4)
        assert 0 <= value < (1 << 16)

    def test_distinct_frames_have_different_hash(self) -> None:
        assert dhash(make_frame(seed=4)) != dhash(make_frame(seed=5))

    def test_grayscale_input_supported(self) -> None:
        gray = make_frame(seed=6)[:, :, 0]  # 2D array
        assert isinstance(dhash(gray), int)

    def test_constant_frame_hash_is_zero(self) -> None:
        # No adjacent-pixel differences -> all bits zero.
        flat = np.full((120, 160, 3), 128, dtype=np.uint8)
        assert dhash(flat) == 0

    @pytest.mark.parametrize("hash_size", [0, -1])
    def test_invalid_hash_size_raises(self, hash_size: int) -> None:
        with pytest.raises(ValueError):
            dhash(make_frame(seed=7), hash_size=hash_size)

    def test_unsupported_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            dhash(np.zeros((4, 4, 4, 3), dtype=np.uint8))


class TestHammingDistance:
    def test_equal_hashes_distance_zero(self) -> None:
        assert hamming_distance(0b1010, 0b1010) == 0

    def test_counts_differing_bits(self) -> None:
        assert hamming_distance(0b0000, 0b1011) == 3

    def test_symmetric(self) -> None:
        assert hamming_distance(123, 456) == hamming_distance(456, 123)


class TestSceneChangeDetector:
    def test_first_frame_is_always_new_scene(self) -> None:
        detector = SceneChangeDetector(threshold=0.1)
        assert detector.is_new_scene(make_frame(seed=1)) is True

    def test_identical_frame_is_same_scene(self) -> None:
        detector = SceneChangeDetector(threshold=0.1)
        frame = make_frame(seed=1)
        detector.is_new_scene(frame)
        assert detector.is_new_scene(frame.copy()) is False

    def test_near_duplicate_is_same_scene(self) -> None:
        detector = SceneChangeDetector(threshold=0.1)
        frame = make_frame(seed=2)
        detector.is_new_scene(frame)
        assert detector.is_new_scene(add_noise(frame, amplitude=5, seed=99)) is False

    def test_distinct_frame_is_new_scene(self) -> None:
        detector = SceneChangeDetector(threshold=0.1)
        detector.is_new_scene(make_frame(seed=3))
        assert detector.is_new_scene(make_frame(seed=4)) is True

    def test_new_scene_updates_stored_hash(self) -> None:
        detector = SceneChangeDetector(threshold=0.1)
        first = make_frame(seed=5)
        second = make_frame(seed=6)
        detector.is_new_scene(first)
        detector.is_new_scene(second)
        assert detector.last_hash == dhash(second)

    def test_same_scene_does_not_update_stored_hash(self) -> None:
        detector = SceneChangeDetector(threshold=0.1)
        frame = make_frame(seed=7)
        detector.is_new_scene(frame)
        committed = detector.last_hash
        detector.is_new_scene(add_noise(frame, amplitude=5, seed=11))
        assert detector.last_hash == committed

    def test_reset_makes_next_frame_new_scene(self) -> None:
        detector = SceneChangeDetector(threshold=0.1)
        frame = make_frame(seed=8)
        detector.is_new_scene(frame)
        detector.reset()
        assert detector.last_hash is None
        assert detector.is_new_scene(frame.copy()) is True

    def test_zero_threshold_treats_any_difference_as_new_scene(self) -> None:
        detector = SceneChangeDetector(threshold=0.0)
        frame = make_frame(seed=9)
        detector.is_new_scene(frame)
        # A single-bit difference exceeds threshold 0.0.
        assert detector.is_new_scene(add_noise(frame, amplitude=40, seed=12)) is True

    def test_threshold_one_treats_everything_as_same_scene(self) -> None:
        detector = SceneChangeDetector(threshold=1.0)
        detector.is_new_scene(make_frame(seed=10))
        # Even a completely different frame cannot exceed the maximum normalized distance.
        assert detector.is_new_scene(make_frame(seed=11)) is False

    def test_threshold_property_exposed(self) -> None:
        assert SceneChangeDetector(threshold=0.25).threshold == 0.25

    @pytest.mark.parametrize("threshold", [-0.1, 1.1])
    def test_invalid_threshold_raises(self, threshold: float) -> None:
        with pytest.raises(ValueError):
            SceneChangeDetector(threshold=threshold)

    @pytest.mark.parametrize("hash_size", [0, -1])
    def test_invalid_hash_size_raises(self, hash_size: int) -> None:
        with pytest.raises(ValueError):
            SceneChangeDetector(hash_size=hash_size)
