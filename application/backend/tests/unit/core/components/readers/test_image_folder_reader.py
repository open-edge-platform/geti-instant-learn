from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from core.components.readers.image_folder_reader import ImageFolderReader
from core.components.schemas.reader import ReaderConfig


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
    """Create an ImageFolderReader instance."""
    return ImageFolderReader(reader_config)


class TestImageFolderReaderInitialization:
    def test_initialization(self, reader_config):
        """Test reader initialization."""
        reader = ImageFolderReader(reader_config)
        assert reader._config == reader_config
        assert reader._image_paths == []
        assert reader._current_index == 0


class TestImageFolderReaderConnect:
    def test_connect_success(self, reader, temp_image_folder):
        """Test successful connection and image scanning."""
        reader.connect()

        assert len(reader._image_paths) == 7  # 5 jpg + 1 png + 1 bmp
        assert reader._current_index == 0
        assert all(p.suffix.lower() in ImageFolderReader.SUPPORTED_EXTENSIONS for p in reader._image_paths)

    def test_connect_invalid_path(self):
        """Test connect with invalid folder path."""
        config = MagicMock(spec=ReaderConfig)
        config.images_folder_path = "/invalid/path"
        reader = ImageFolderReader(config)

        with pytest.raises(ValueError, match="Invalid folder path"):
            reader.connect()

    def test_connect_file_instead_of_folder(self, tmp_path):
        """Test connect when path points to a file instead of folder."""
        file_path = tmp_path / "file.txt"
        file_path.write_text("test")

        config = MagicMock(spec=ReaderConfig)
        config.images_folder_path = str(file_path)
        reader = ImageFolderReader(config)

        with pytest.raises(ValueError, match="Invalid folder path"):
            reader.connect()

    def test_connect_sorting(self, tmp_path):
        """Test that images are sorted correctly."""
        # Create images with numeric names
        for i in [1, 10, 2, 20, 3]:
            cv2.imwrite(str(tmp_path / f"img_{i}.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))

        config = Mock(spec=ReaderConfig)
        config.images_folder_path = str(tmp_path)
        reader = ImageFolderReader(config)
        reader.connect()

        names = [p.stem for p in reader._image_paths]
        assert names == ["img_1", "img_2", "img_3", "img_10", "img_20"]


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
        """Test index increments after read."""
        reader.connect()
        reader.read()
        assert reader.index() == 1


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

    def test_read_increments_index(self, reader):
        """Test that read increments the current index."""
        reader.connect()
        initial_index = reader.index()
        reader.read()
        assert reader.index() == initial_index + 1

    def test_read_all_images(self, reader):
        """Test reading all images sequentially."""
        reader.connect()
        total = reader.input_data()

        for i in range(total):
            data = reader.read()
            assert data is not None
            assert data.context["index"] == i

        # Next read should return None
        assert reader.read() is None

    def test_read_without_connect(self, reader):
        """Test reading without calling connect first."""
        assert reader.read() is None

    def test_read_corrupted_image(self, reader, temp_image_folder):
        """Test handling of corrupted images."""
        # Create a corrupted image file
        (temp_image_folder / "corrupted.jpg").write_bytes(b"not an image")

        reader.connect()

        # Should skip corrupted image and continue
        with patch("cv2.imread", side_effect=[None, np.zeros((10, 10, 3))]):
            data = reader.read()
            assert data is not None
            assert reader.index() == 2  # Skipped corrupted, read next

    def test_read_timestamp_format(self, reader):
        """Test that timestamp is in milliseconds."""
        reader.connect()
        data = reader.read()

        # Timestamp should be reasonable (current time in ms)
        assert data.timestamp > 1000000000000  # After year 2001
        assert data.timestamp < 9999999999999  # Before year 2286


class TestImageFolderReaderListFrames:
    def test_list_frames_first_page(self, reader):
        """Test listing first page of frames."""
        reader.connect()
        result = reader.list_frames(page=1, page_size=3)

        assert result["total"] == 7
        assert result["page"] == 1
        assert result["page_size"] == 3
        assert len(result["frames"]) == 3

    def test_list_frames_second_page(self, reader):
        """Test listing second page of frames."""
        reader.connect()
        result = reader.list_frames(page=2, page_size=3)

        assert result["total"] == 7
        assert result["page"] == 2
        assert len(result["frames"]) == 3

    def test_list_frames_last_page_partial(self, reader):
        """Test listing last page with fewer items."""
        reader.connect()
        result = reader.list_frames(page=3, page_size=3)

        assert len(result["frames"]) == 1  # 7 total, 3 per page, last page has 1

    def test_list_frames_beyond_available(self, reader):
        """Test listing page beyond available frames."""
        reader.connect()
        result = reader.list_frames(page=10, page_size=30)

        assert len(result["frames"]) == 0


class TestImageFolderReaderInputData:
    def test_input_data_returns_count(self, reader):
        """Test input_data returns correct count."""
        reader.connect()
        assert reader.input_data() == 7


class TestImageFolderReaderClose:
    def test_close_clears_state(self, reader):
        """Test that close clears internal state."""
        reader.connect()
        reader.seek(3)

        reader.close()

        assert reader._image_paths == []
        assert reader._current_index == 0


class TestImageFolderReaderContextManager:
    def test_context_manager(self, reader_config):
        """Test using reader as context manager."""
        with ImageFolderReader(reader_config) as reader:
            reader.connect()
            assert len(reader._image_paths) > 0

        # After exiting context, close should have been called
        assert reader._image_paths == []
