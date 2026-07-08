#  Copyright (C) 2025 Intel Corporation
#  SPDX-License-Identifier: Apache-2.0

"""Content-aware scene change detection using difference hash (dHash).

dHash overview:
    1. Convert the frame to grayscale and downsample to a (hash_size, hash_size + 1)
       thumbnail (default 8x9 -> 64-bit hash).
    2. For each row, compare each adjacent pair of pixels -> 64 boolean bits.
    3. Pack the bits into a single integer (the hash).
    4. Two frames are "the same scene" when the normalized Hamming distance between
       their hashes is <= a configurable threshold.
"""
import logging

import numpy as np

logger = logging.getLogger(__name__)

# Luminance weights for RGB -> grayscale (ITU-R BT.601).
_RGB_TO_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Convert an RGB HWC frame to a 2D grayscale array. Pass 2D input through unchanged."""
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.ndim == 3 and frame.shape[2] >= 3:
        return frame[:, :, :3].astype(np.float32) @ _RGB_TO_GRAY
    raise ValueError(f"Expected an HWC RGB frame or a 2D grayscale array, got shape {frame.shape}")


def _resize_nearest(gray: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor downsample of a 2D array to (out_h, out_w). Dependency-free."""
    h, w = gray.shape
    if h == 0 or w == 0:
        raise ValueError(f"Cannot resize an empty frame with shape {gray.shape}")
    row_idx = (np.arange(out_h) * h) // out_h
    col_idx = (np.arange(out_w) * w) // out_w
    return gray[row_idx][:, col_idx]


def dhash(frame: np.ndarray, hash_size: int = 8) -> int:
    """Compute the difference hash of a frame.

    Args:
        frame: RGB HWC uint8 frame (H, W, 3) or a 2D grayscale array.
        hash_size: Number of rows/columns in the comparison grid. Produces a
            hash_size * hash_size bit hash (default 8 -> 64 bits).

    Returns:
        The dHash as a non-negative integer.

    Raises:
        ValueError: If hash_size < 1 or the frame shape is unsupported.
    """
    if hash_size < 1:
        raise ValueError(f"hash_size must be >= 1, got {hash_size}")

    gray = _to_grayscale(frame)
    # Width is one larger than height so adjacent-column diffs yield hash_size columns.
    small = _resize_nearest(gray, hash_size, hash_size + 1)
    diff = small[:, 1:] > small[:, :-1]  # (hash_size, hash_size) boolean grid

    # Pack the row-major bit grid into a single integer.
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def hamming_distance(hash_a: int, hash_b: int) -> int:
    """Number of differing bits between two hashes."""
    return int(bin(hash_a ^ hash_b).count("1"))


class SceneChangeDetector:
    """Tracks the last processed frame's dHash and decides whether a new frame is a new scene.

    A frame is considered a *new scene* (and therefore worth processing) when the
    normalized Hamming distance to the last processed frame exceeds ``threshold``.
    The first frame seen is always a new scene.

    Args:
        threshold: Normalized Hamming distance (0..1) at or below which two frames are
            treated as the same scene. Default 0.5 (<= ~32 bits for a 64-bit hash).
        hash_size: dHash grid size; ``hash_size * hash_size`` bits total.

    Raises:
        ValueError: If threshold is outside [0, 1] or hash_size < 1.
    """

    def __init__(self, threshold: float = 0.5, hash_size: int = 8) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {threshold}")
        if hash_size < 1:
            raise ValueError(f"hash_size must be >= 1, got {hash_size}")
        self._threshold = threshold
        self._hash_size = hash_size
        self._bits = hash_size * hash_size
        self._last_hash: int | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def last_hash(self) -> int | None:
        return self._last_hash

    def is_new_scene(self, frame: np.ndarray) -> bool:
        """Return True if the frame differs enough from the last processed frame to be a new scene.

        Updates the stored hash only when a new scene is detected, so slow drift across
        many near-duplicate frames is always measured against the last *committed* scene.
        """
        current = dhash(frame, hash_size=self._hash_size)
        if self._last_hash is None:
            self._last_hash = current
            return True

        normalized = hamming_distance(current, self._last_hash) / self._bits
        logger.debug("Normalized Hamming Distance: %.4f, Threshold: %.4f", normalized, self._threshold)
        if normalized > self._threshold:
            self._last_hash = current
            return True
        return False

    def reset(self) -> None:
        """Forget the last processed frame so the next frame is treated as a new scene."""
        self._last_hash = None
