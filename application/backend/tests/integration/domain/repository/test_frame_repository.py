# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from uuid import uuid4

import cv2
import numpy as np
import pytest

from domain.repositories.frame import FrameRepository


@pytest.fixture
def temp_frame_dir(tmp_path):
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    return frame_dir


@pytest.fixture
def frame_repository(temp_frame_dir):
    return FrameRepository(base_dir=temp_frame_dir)


@pytest.fixture
def sample_frame():
    # create a simple 100x100 BGR image with a blue background
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, :] = [0, 0, 255]
    return frame


def test_save_frame_creates_directory_structure(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id = uuid4()

    saved_path = frame_repository.save_frame(project_id, frame_id, sample_frame)

    assert saved_path.exists()
    assert saved_path.parent.name == "frames"
    assert saved_path.parent.parent.name == str(project_id)


def test_save_frame_stores_as_jpeg(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id = uuid4()

    saved_path = frame_repository.save_frame(project_id, frame_id, sample_frame)

    assert saved_path.suffix == ".jpg"
    assert saved_path.name == f"{frame_id}.jpg"


def test_save_frame_content_is_valid_image(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id = uuid4()

    saved_path = frame_repository.save_frame(project_id, frame_id, sample_frame)

    loaded_frame = cv2.imread(str(saved_path))
    assert loaded_frame is not None
    assert loaded_frame.shape == sample_frame.shape


def test_save_frame_encoding_failure_raises_error(frame_repository):
    project_id = uuid4()
    frame_id = uuid4()
    invalid_frame = np.array([])  # empty array that will fail encoding

    with pytest.raises(RuntimeError, match=f"Failed to encode frame {frame_id}"):
        frame_repository.save_frame(project_id, frame_id, invalid_frame)


def test_get_frame_path_returns_path_when_exists(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id = uuid4()

    frame_repository.save_frame(project_id, frame_id, sample_frame)
    retrieved_path = frame_repository.get_frame_path(project_id, frame_id)

    assert retrieved_path is not None
    assert retrieved_path.exists()
    assert retrieved_path.name == f"{frame_id}.jpg"


def test_get_frame_path_returns_none_when_not_exists(frame_repository):
    project_id = uuid4()
    frame_id = uuid4()

    retrieved_path = frame_repository.get_frame_path(project_id, frame_id)

    assert retrieved_path is None


def test_delete_frame_removes_file(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id = uuid4()

    saved_path = frame_repository.save_frame(project_id, frame_id, sample_frame)
    assert saved_path.exists()

    deletion_result = frame_repository.delete_frame(project_id, frame_id)

    assert deletion_result is True
    assert not saved_path.exists()


def test_delete_frame_returns_false_when_not_exists(frame_repository):
    project_id = uuid4()
    frame_id = uuid4()

    deletion_result = frame_repository.delete_frame(project_id, frame_id)

    assert deletion_result is False


def test_save_multiple_frames_in_same_project(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id_1 = uuid4()
    frame_id_2 = uuid4()

    path_1 = frame_repository.save_frame(project_id, frame_id_1, sample_frame)
    path_2 = frame_repository.save_frame(project_id, frame_id_2, sample_frame)

    assert path_1.exists()
    assert path_2.exists()
    assert path_1.parent == path_2.parent
    assert path_1 != path_2


def test_save_frames_in_different_projects(frame_repository, sample_frame):
    project_id_1 = uuid4()
    project_id_2 = uuid4()
    frame_id = uuid4()

    path_1 = frame_repository.save_frame(project_id_1, frame_id, sample_frame)
    path_2 = frame_repository.save_frame(project_id_2, frame_id, sample_frame)

    assert path_1.exists()
    assert path_2.exists()
    assert path_1.parent.parent != path_2.parent.parent


def test_get_frame_returns_frame_when_exists(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id = uuid4()

    frame_repository.save_frame(project_id, frame_id, sample_frame)
    retrieved_frame = frame_repository.get_frame(project_id, frame_id)

    assert retrieved_frame is not None
    assert retrieved_frame.shape == sample_frame.shape
    assert retrieved_frame.dtype == np.uint8


def test_get_frame_converts_bgr_to_rgb(frame_repository):
    project_id = uuid4()
    frame_id = uuid4()
    height = 640
    width = 480
    channels = 3

    # Create a frame with specific colors in RGB format
    # Red=10, Green=128, Blue=245
    rgb_frame_input = np.zeros((height, width, channels), dtype=np.uint8)
    rgb_frame_input[:, :] = [10, 128, 245]

    frame_repository.save_frame(project_id, frame_id, rgb_frame_input)
    rgb_frame_output = frame_repository.get_frame(project_id, frame_id)

    assert rgb_frame_output is not None
    assert rgb_frame_output.shape == (height, width, channels)

    # Check pixel values with tolerance for JPEG compression artifacts (±10)
    pixel = rgb_frame_output[50, 50]
    assert abs(int(pixel[0]) - 10) <= 10, f"Red channel mismatch: expected ~10, got {pixel[0]}"
    assert abs(int(pixel[1]) - 128) <= 10, f"Green channel mismatch: expected ~128, got {pixel[1]}"
    assert abs(int(pixel[2]) - 245) <= 10, f"Blue channel mismatch: expected ~245, got {pixel[2]}"


def test_get_frame_returns_none_when_not_exists(frame_repository):
    project_id = uuid4()
    frame_id = uuid4()

    retrieved_frame = frame_repository.get_frame(project_id, frame_id)

    assert retrieved_frame is None


def test_get_frame_returns_none_on_corrupted_file(frame_repository, sample_frame):
    project_id = uuid4()
    frame_id = uuid4()

    # Save a valid frame first
    frame_path = frame_repository.save_frame(project_id, frame_id, sample_frame)

    # Corrupt the file by overwriting with garbage data
    frame_path.write_bytes(b"corrupted data not an image")

    # Should return None instead of raising an exception
    retrieved_frame = frame_repository.get_frame(project_id, frame_id)
    assert retrieved_frame is None


def test_get_frame_preserves_dimensions(frame_repository):
    """Test that get_frame preserves frame dimensions."""
    project_id = uuid4()
    frame_id = uuid4()

    # Create frames with different dimensions
    for height, width in [(100, 100), (480, 640), (1080, 1920)]:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = [100, 150, 200]

        frame_repository.save_frame(project_id, frame_id, frame)
        retrieved_frame = frame_repository.get_frame(project_id, frame_id)

        assert retrieved_frame is not None
        assert retrieved_frame.shape == (height, width, 3)
