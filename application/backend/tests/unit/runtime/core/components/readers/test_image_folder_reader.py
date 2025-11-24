from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from domain.services.schemas.reader import FrameListResponse, FrameMetadata, ReaderConfig
from runtime.core.components.readers.image_folder_reader import ImageFolderReader
from settings import get_settings

settings = get_settings()


@pytest.fixture
def temp_image_folder(tmp_path):
    """Create a temporary folder with test images."""
    # Create test images
    for i in range(5):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = [i * 50, i * 50, i * 50]  # Different colors
        cv2.imwrite(str(tmp_path / f"image_{i}.jpg"), img)

    # Add images with different extensions
    cv2.imwrite(str(tmp_path / "test.png"), np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.imwrite(str(tmp_path / "test.bmp"), np.zeros((100, 100, 3), dtype=np.uint8))

    # Add a non-image file
    (tmp_path / "readme.txt").write_text("test")

    return tmp_path


@pytest.fixture
def reader_config(temp_image_folder):
    """Create a ReaderConfig for testing."""
    config = MagicMock(spec=ReaderConfig)
    config.images_folder_path = str(temp_image_folder)
    return config


@pytest.fixture
def reader(reader_config):
    return ImageFolderReader(reader_config, settings.supported_extensions)


class TestImageFolderReaderInitialization:
    def test_initialization(self, reader_config):
        reader = ImageFolderReader(reader_config, settings.supported_extensions)
        assert reader._config == reader_config
        assert reader._image_paths == []
        assert reader._current_index == 0
        assert reader._thumbnail_cache == {}


class TestImageFolderReaderConnect:
    def test_connect_success(self, reader, temp_image_folder):
        """Test successful connection and image scanning."""
        reader.connect()

        assert len(reader._image_paths) == 7  # 5 jpg + 1 png + 1 bmp
        assert reader._current_index == 0
        assert all(p.suffix.lower() in settings.supported_extensions for p in reader._image_paths)

    def test_connect_pregenerate_thumbnails(self, reader):
        """Test that thumbnails are pre-generated for first 30 images."""
        reader.connect()

        # Should pre-generate thumbnails for first 7 images (less than 30)
        assert len(reader._thumbnail_cache) == 7
        for idx in range(7):
            assert idx in reader._thumbnail_cache
            assert isinstance(reader._thumbnail_cache[idx], str)  # base64 string

    def test_connect_invalid_path(self):
        """Test connect with invalid folder path."""
        config = MagicMock(spec=ReaderConfig)
        config.images_folder_path = "/invalid/path"
        reader = ImageFolderReader(config, settings.supported_extensions)

        with pytest.raises(ValueError, match="Invalid folder path"):
            reader.connect()

    def test_connect_file_instead_of_folder(self, tmp_path):
        """Test connect when path points to a file instead of folder."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        config = MagicMock(spec=ReaderConfig)
        config.images_folder_path = str(file_path)
        reader = ImageFolderReader(config, settings.supported_extensions)

        with pytest.raises(ValueError, match="Invalid folder path"):
            reader.connect()


def test_natural_sort_key():
    """Test natural sorting of filenames."""
    paths = [Path(f"img_{i}.jpg") for i in [1, 10, 2, 20, 3]]
    sorted_paths = sorted(paths, key=ImageFolderReader._natural_sort_key)
    expected = ["img_1.jpg", "img_2.jpg", "img_3.jpg", "img_10.jpg", "img_20.jpg"]
    assert [path.name for path in sorted_paths] == expected


class TestImageFolderReaderSeek:
    def test_seek_valid_index(self, reader):
        """Test seeking to a valid index."""
        reader.connect()
        reader.seek(3)
        assert reader._current_index == 3

    def test_seek_without_connect(self, reader):
        """Test seeking before calling connect."""
        with pytest.raises(ValueError, match="No images loaded"):
            reader.seek(0)

    def test_seek_negative_index(self, reader):
        """Test seeking to a negative index."""
        reader.connect()
        with pytest.raises(IndexError, match="out of range"):
            reader.seek(-1)

    def test_seek_index_too_large(self, reader):
        """Test seeking beyond available images."""
        reader.connect()
        with pytest.raises(IndexError, match="out of range"):
            reader.seek(100)

    def test_seek_clears_cache(self, reader):
        reader.connect()
        reader.read()
        assert reader._last_image is not None

        reader.seek(3)
        assert reader._last_image is None
        assert reader._last_image_path is None


class TestImageFolderReaderIndex:
    def test_index_initial(self, reader):
        """Test index returns 0 initially."""
        assert reader.index() == 0

    def test_index_after_seek(self, reader):
        """Test index after seeking."""
        reader.connect()
        reader.seek(5)
        assert reader.index() == 5

    def test_index_after_read(self, reader):
        reader.connect()
        initial_index = reader.index()
        reader.read()
        assert reader.index() == initial_index


class TestImageFolderReaderLength:
    def test_len_after_connect(self, reader):
        reader.connect()
        assert len(reader) == 7

    def test_len_before_connect(self, reader):
        assert len(reader) == 0


class TestImageFolderReaderRead:
    def test_read_success(self, reader):
        """Test successful image reading."""
        reader.connect()
        data = reader.read()

        assert data is not None
        assert isinstance(data.frame, np.ndarray)
        assert isinstance(data.timestamp, int)
        assert "path" in data.context
        assert "index" in data.context
        assert data.context["index"] == 0

    def test_read_does_not_increment_index(self, reader):
        reader.connect()
        initial_index = reader.index()
        reader.read()
        assert reader.index() == initial_index

    def test_read_specific_images_with_seek(self, reader):
        reader.connect()
        total = len(reader)

        for i in range(total):
            reader.seek(i)
            data = reader.read()
            assert data is not None
            assert data.context["index"] == i

    def test_read_without_connect(self, reader):
        assert reader.read() is None

    def test_read_corrupted_image(self, reader, temp_image_folder):
        (temp_image_folder / "corrupted.jpg").write_bytes(b"not an image")

        reader.connect()

        with patch("cv2.imread", side_effect=[None]):
            data = reader.read()
            assert data is None

    def test_read_timestamp_format(self, reader):
        reader.connect()
        data = reader.read()

        assert data.timestamp > 1000000000000
        assert data.timestamp < 9999999999999

    def test_read_caches_image(self, reader):
        reader.connect()
        path = reader._image_paths[0]

        reader.read()
        assert reader._last_image is not None
        assert reader._last_image_path == path

        with patch("cv2.imread") as mock_imread:
            reader.read()
            mock_imread.assert_not_called()


class TestImageFolderReaderListFrames:
    def test_list_frames_first_page(self, reader):
        """Test listing first page of frames."""
        reader.connect()
        result = reader.list_frames(offset=0, limit=3)

        assert isinstance(result, FrameListResponse)
        assert result.pagination.total == 7
        assert result.pagination.offset == 0
        assert result.pagination.limit == 3
        assert result.pagination.count == 3
        assert len(result.frames) == 3

    def test_list_frames_second_page(self, reader):
        """Test listing second page of frames."""
        reader.connect()
        result = reader.list_frames(offset=3, limit=3)

        assert result.pagination.total == 7
        assert result.pagination.offset == 3
        assert result.pagination.count == 3
        assert len(result.frames) == 3

    def test_list_frames_last_page_partial(self, reader):
        """Test listing last page with fewer items."""
        reader.connect()
        result = reader.list_frames(offset=6, limit=3)

        assert result.pagination.count == 1  # 7 total, offset 6, only 1 remaining
        assert len(result.frames) == 1

    def test_list_frames_beyond_available(self, reader):
        """Test listing page beyond available frames."""
        reader.connect()
        result = reader.list_frames(offset=300, limit=30)

        assert len(result.frames) == 0
        assert result.pagination.count == 0

    def test_list_frames_metadata_structure(self, reader):
        """Test that frame metadata has correct structure."""
        reader.connect()
        result = reader.list_frames(offset=0, limit=1)

        assert len(result.frames) == 1
        frame = result.frames[0]
        assert isinstance(frame, FrameMetadata)
        assert isinstance(frame.index, int)
        assert isinstance(frame.thumbnail, str)
        assert frame.thumbnail.startswith("data:image/jpeg;base64,")
        assert frame.index == 0

    def test_list_frames_uses_cache(self, reader):
        """Test that thumbnails are retrieved from cache."""
        reader.connect()

        # First call should use pre-generated cache
        with patch.object(ImageFolderReader, "_generate_thumbnail") as mock_gen:
            result = reader.list_frames(offset=0, limit=3)
            assert len(result.frames) == 3
            # Should not call _generate_thumbnail since thumbnails are cached
            mock_gen.assert_not_called()

    def test_list_frames_generates_uncached(self, tmp_path):
        """Test that uncached thumbnails are generated on demand."""
        # Create 35 images (more than initial cache of 30)
        for i in range(35):
            cv2.imwrite(str(tmp_path / f"img_{i}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

        config = Mock(spec=ReaderConfig)
        config.images_folder_path = str(tmp_path)
        reader = ImageFolderReader(config, settings.supported_extensions)
        reader.connect()

        # Request frames 30-34 - should generate thumbnails
        result = reader.list_frames(offset=30, limit=30)

        # All frames should have thumbnails (generated on demand)
        assert len(result.frames) == 5
        for frame in result.frames:
            assert frame.thumbnail is not None
            assert frame.thumbnail.startswith("data:image/jpeg;base64,")

        # Cache should now include indexes 30-34
        for idx in range(30, 35):
            assert idx in reader._thumbnail_cache


class TestImageFolderReaderClose:
    def test_close_clears_state(self, reader):
        reader.connect()
        reader.seek(3)

        reader.close()

        assert reader._image_paths == []
        assert reader._current_index == 0
        assert reader._last_image is None
        assert reader._last_image_path is None


class TestImageFolderReaderContextManager:
    def test_context_manager(self, reader_config):
        with ImageFolderReader(reader_config, settings.supported_extensions) as reader:
            reader.connect()
            assert len(reader._image_paths) > 0

        # After exiting context, close should have been called
        assert reader._image_paths == []


class TestImageFolderReaderThumbnailGeneration:
    def test_generate_thumbnail_valid_image(self, tmp_path):
        """Test thumbnail generation for valid image."""
        img_path = tmp_path / "test.jpg"
        cv2.imwrite(str(img_path), np.zeros((200, 200, 3), dtype=np.uint8))

        thumbnail = ImageFolderReader._generate_thumbnail(img_path)

        assert thumbnail is not None
        assert isinstance(thumbnail, str)
        assert len(thumbnail) > 0  # base64 encoded string

    def test_generate_thumbnail_invalid_image(self, tmp_path):
        """Test thumbnail generation for invalid image."""
        img_path = tmp_path / "invalid.jpg"
        img_path.write_bytes(b"not an image")

        thumbnail = ImageFolderReader._generate_thumbnail(img_path)

        assert thumbnail is None

    def test_generate_thumbnail_nonexistent_file(self):
        """Test thumbnail generation for non-existent file."""
        thumbnail = ImageFolderReader._generate_thumbnail(Path("/nonexistent/file.jpg"))

        assert thumbnail is None
