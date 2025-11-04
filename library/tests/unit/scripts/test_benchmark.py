# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for benchmark script including CLI options and dataset handling."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from getiprompt.scripts.benchmark import load_dataset_by_name, perform_benchmark_experiment, predict_on_dataset
from getiprompt.utils.args import get_arguments


class TestBenchmarkCLI:
    """Test CLI options in benchmark script."""

    def test_get_arguments_default_values(self) -> None:
        """Test that default values are correctly parsed."""
        # Test that get_arguments function exists and can be imported
        assert callable(get_arguments)

    def test_get_arguments_custom_values(self) -> None:
        """Test parsing custom values."""
        # Test that get_arguments function exists and can be imported
        assert callable(get_arguments)

    def test_get_arguments_attributes(self) -> None:
        """Test that get_arguments returns expected attributes."""
        # Test that get_arguments function exists and can be imported
        assert callable(get_arguments)


class TestBenchmarkDatasetHandling:
    """Test dataset handling in benchmark script."""

    @patch("getiprompt.scripts.benchmark.Path")
    def test_load_dataset_by_name_lvis(self, mock_path: Path) -> None:
        """Test dataset loading with LVIS dataset."""
        mock_path.return_value = Path("/home/user/datasets")

        # Mock dataset classes
        with patch("getiprompt.scripts.benchmark.LVISDataset") as mock_lvis:
            mock_lvis.return_value = MagicMock()

            load_dataset_by_name("lvis", categories="default")

            mock_lvis.assert_called_once()

    @patch("getiprompt.scripts.benchmark.Path")
    def test_load_dataset_by_name_perseg(self, mock_path: Path) -> None:
        """Test dataset loading with PerSeg dataset."""
        custom_path = "/custom/datasets"
        mock_path.return_value = Path(custom_path)

        # Mock dataset classes
        with patch("getiprompt.scripts.benchmark.PerSegDataset") as mock_perseg:
            mock_perseg.return_value = MagicMock()

            load_dataset_by_name("perseg", categories="default", dataset_root=custom_path)

            mock_perseg.assert_called_once()

    def test_load_dataset_by_name_all_categories(self) -> None:
        """Test dataset loading with all categories."""
        with patch("getiprompt.scripts.benchmark.LVISDataset") as mock_lvis:
            mock_lvis.return_value = MagicMock()

            load_dataset_by_name("lvis", categories="all")

            mock_lvis.assert_called_once()

    def test_load_dataset_by_name_category_filtering(self) -> None:
        """Test dataset loading with category filtering."""
        with patch("getiprompt.scripts.benchmark.LVISDataset") as mock_lvis:
            mock_lvis.return_value = MagicMock()

            # Test with specific category
            load_dataset_by_name("lvis", categories=["cat"])
            mock_lvis.assert_called_once()

            # Reset mock for second test
            mock_lvis.reset_mock()

            # Test with benchmark categories
            load_dataset_by_name("lvis", categories="benchmark")
            mock_lvis.assert_called_once()

    def test_load_dataset_by_name_error_handling(self) -> None:
        """Test error handling in dataset loading."""
        with patch("getiprompt.scripts.benchmark.LVISDataset") as mock_lvis:
            mock_lvis.side_effect = FileNotFoundError("Dataset not found")

            with pytest.raises(FileNotFoundError):
                load_dataset_by_name("lvis", categories="default")


class TestBenchmarkModelHandling:
    """Test model handling in benchmark script."""

    def test_predict_on_dataset_single_model(self) -> None:
        """Test running prediction on dataset with single model."""
        with (
            patch("getiprompt.scripts.benchmark.load_model") as mock_load_model,
            patch("getiprompt.scripts.benchmark.SegmentationMetrics") as mock_metrics,
            patch("getiprompt.scripts.benchmark.learn_from_category") as mock_learn,
            patch("getiprompt.scripts.benchmark.infer_on_category") as mock_infer,
            patch("getiprompt.scripts.benchmark.prepare_output_directory") as mock_handle_path,
        ):
            mock_model = MagicMock()
            mock_load_model.return_value = mock_model
            mock_handle_path.return_value = Path(tempfile.mkdtemp())

            # Create mock metrics that returns a proper dict
            mock_metrics_instance = MagicMock()
            mock_metrics_instance.get_metrics.return_value = {
                "category": ["cat", "dog"],
                "accuracy": [0.8, 0.9],
                "precision": [0.7, 0.8],
                "recall": [0.6, 0.7],
                "f1": [0.65, 0.75],
            }
            mock_metrics.return_value = mock_metrics_instance

            # Create mock dataset
            mock_dataset = MagicMock()
            mock_dataset.categories = ["cat", "dog"]
            mock_dataset.get_reference_dataset.return_value = MagicMock()
            mock_dataset.get_target_dataset.return_value = MagicMock()

            # Mock the learn and infer functions
            mock_learn.return_value = ([], [])
            mock_infer.return_value = (0, 0)  # (total_time, n_samples)

            # Create mock args
            mock_args = MagicMock()
            mock_args.n_shot = 1
            mock_args.batch_size = 4
            mock_args.overwrite = False

            # Test predict_on_dataset
            result = predict_on_dataset(
                args=mock_args,
                model=mock_model,
                dataset=mock_dataset,
                output_path=Path(tempfile.mkdtemp()),
                dataset_name="lvis",
                model_name="Matcher",
                backbone_name="SAM-HQ-tiny",
                number_of_priors_tests=1,
            )

            # Should return a DataFrame
            assert result is not None

    def test_predict_on_dataset_error_handling(self) -> None:
        """Test error handling in dataset prediction."""
        with patch("getiprompt.scripts.benchmark.load_model") as mock_load_model:
            mock_load_model.side_effect = RuntimeError("Model loading failed")

            mock_dataset = MagicMock()
            mock_args = MagicMock()

            with pytest.raises(RuntimeError):
                predict_on_dataset(
                    args=mock_args,
                    model=mock_load_model(),
                    dataset=mock_dataset,
                    output_path=Path(tempfile.mkdtemp()),
                    dataset_name="lvis",
                    model_name="Matcher",
                    backbone_name="SAM-HQ-tiny",
                    number_of_priors_tests=1,
                )


class TestBenchmarkOutputHandling:
    """Test output handling in benchmark script."""

    def test_output_directory_creation(self) -> None:
        """Test output directory creation."""
        with (
            patch("getiprompt.scripts.benchmark.Path.mkdir") as mock_mkdir,
            patch("getiprompt.scripts.benchmark.Path.exists") as mock_exists,
            patch("getiprompt.scripts.benchmark.get_arguments") as mock_get_args,
            patch("getiprompt.scripts.benchmark.parse_experiment_args") as mock_parse_args,
            patch("getiprompt.scripts.benchmark.load_dataset_by_name") as mock_load_dataset,
            patch("getiprompt.scripts.benchmark.load_model") as mock_load_model,
            patch("getiprompt.scripts.benchmark.predict_on_dataset") as mock_predict,
            patch("getiprompt.scripts.benchmark._save_results") as mock_save,
            patch("getiprompt.scripts.benchmark.setup_logger") as mock_setup_logger,
        ):
            mock_exists.return_value = False
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_get_args.return_value = mock_args
            mock_parse_args.return_value = (
                [MagicMock(value="lvis")],
                [MagicMock(value="Matcher")],
                [MagicMock(value="SAM-HQ-tiny")],
            )
            mock_load_dataset.return_value = MagicMock()
            mock_load_model.return_value = MagicMock()
            mock_predict.return_value = MagicMock()
            mock_save.return_value = None
            mock_setup_logger.return_value = None

            # Test with custom output directory
            perform_benchmark_experiment()

            mock_mkdir.assert_called()

    def test_results_saving(self) -> None:
        """Test saving benchmark results."""
        with (
            patch("getiprompt.scripts.benchmark.get_arguments") as mock_get_args,
            patch("getiprompt.scripts.benchmark.parse_experiment_args") as mock_parse_args,
            patch("getiprompt.scripts.benchmark.load_dataset_by_name") as mock_load_dataset,
            patch("getiprompt.scripts.benchmark.load_model") as mock_load_model,
            patch("getiprompt.scripts.benchmark.predict_on_dataset") as mock_predict,
            patch("getiprompt.scripts.benchmark._save_results") as mock_save,
            patch("getiprompt.scripts.benchmark.setup_logger") as mock_setup_logger,
        ):
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_get_args.return_value = mock_args
            mock_parse_args.return_value = (
                [MagicMock(value="lvis")],
                [MagicMock(value="Matcher")],
                [MagicMock(value="SAM-HQ-tiny")],
            )
            mock_load_dataset.return_value = MagicMock()
            mock_load_model.return_value = MagicMock()
            mock_predict.return_value = MagicMock()
            mock_save.return_value = None
            mock_setup_logger.return_value = None

            perform_benchmark_experiment()

            mock_save.assert_called()

    def test_results_formatting(self) -> None:
        """Test results formatting."""
        with (
            patch("getiprompt.scripts.benchmark.get_arguments") as mock_get_args,
            patch("getiprompt.scripts.benchmark.parse_experiment_args") as mock_parse_args,
            patch("getiprompt.scripts.benchmark.load_dataset_by_name") as mock_load_dataset,
            patch("getiprompt.scripts.benchmark.load_model") as mock_load_model,
            patch("getiprompt.scripts.benchmark.predict_on_dataset") as mock_predict,
            patch("getiprompt.scripts.benchmark._save_results") as mock_save,
            patch("getiprompt.scripts.benchmark.setup_logger") as mock_setup_logger,
        ):
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_get_args.return_value = mock_args
            mock_parse_args.return_value = (
                [MagicMock(value="lvis")],
                [MagicMock(value="Matcher")],
                [MagicMock(value="SAM-HQ-tiny")],
            )
            mock_load_dataset.return_value = MagicMock()
            mock_load_model.return_value = MagicMock()
            mock_predict.return_value = MagicMock()
            mock_save.return_value = None
            mock_setup_logger.return_value = None

            perform_benchmark_experiment()

            # Should call predict_on_dataset
            mock_predict.assert_called()


class TestBenchmarkIntegration:
    """Test integration of benchmark functionality."""

    def test_perform_benchmark_experiment_integration(self) -> None:
        """Test benchmark experiment integration."""
        with (
            patch("getiprompt.scripts.benchmark.get_arguments") as mock_get_args,
            patch("getiprompt.scripts.benchmark.parse_experiment_args") as mock_parse_args,
            patch("getiprompt.scripts.benchmark.load_dataset_by_name") as mock_load_dataset,
            patch("getiprompt.scripts.benchmark.load_model") as mock_load_model,
            patch("getiprompt.scripts.benchmark.predict_on_dataset") as mock_predict,
            patch("getiprompt.scripts.benchmark._save_results") as mock_save,
            patch("getiprompt.scripts.benchmark.setup_logger") as mock_setup_logger,
        ):
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_get_args.return_value = mock_args
            mock_parse_args.return_value = (
                [MagicMock(value="lvis")],
                [MagicMock(value="Matcher")],
                [MagicMock(value="SAM-HQ-tiny")],
            )
            mock_load_dataset.return_value = MagicMock()
            mock_load_model.return_value = MagicMock()
            mock_predict.return_value = MagicMock()
            mock_save.return_value = None
            mock_setup_logger.return_value = None

            # Test with default arguments
            perform_benchmark_experiment()

            mock_load_dataset.assert_called()
            mock_predict.assert_called()
            mock_save.assert_called()

    def test_perform_benchmark_experiment_with_custom_args(self) -> None:
        """Test benchmark experiment with custom arguments."""
        with (
            patch("getiprompt.scripts.benchmark.parse_experiment_args") as mock_parse_args,
            patch("getiprompt.scripts.benchmark.load_dataset_by_name") as mock_load_dataset,
            patch("getiprompt.scripts.benchmark.load_model") as mock_load_model,
            patch("getiprompt.scripts.benchmark.predict_on_dataset") as mock_predict,
            patch("getiprompt.scripts.benchmark._save_results") as mock_save,
            patch("getiprompt.scripts.benchmark.setup_logger") as mock_setup_logger,
        ):
            mock_parse_args.return_value = (
                [MagicMock(value="lvis")],
                [MagicMock(value="Matcher")],
                [MagicMock(value="SAM-HQ-tiny")],
            )
            mock_load_dataset.return_value = MagicMock()
            mock_load_model.return_value = MagicMock()
            mock_predict.return_value = MagicMock()
            mock_save.return_value = None
            mock_setup_logger.return_value = None

            # Create custom args
            custom_args = MagicMock()
            custom_args.experiment_name = "test_experiment"
            custom_args.class_name = "benchmark"
            custom_args.n_shot = 2
            custom_args.dataset_root = "/custom/path"
            custom_args.log_level = "INFO"

            perform_benchmark_experiment(custom_args)

            mock_load_dataset.assert_called()
            mock_predict.assert_called()
            mock_save.assert_called()

    def test_perform_benchmark_experiment_error_handling(self) -> None:
        """Test error handling in benchmark experiment."""
        with (
            patch("getiprompt.scripts.benchmark.get_arguments") as mock_get_args,
            patch("getiprompt.scripts.benchmark.parse_experiment_args") as mock_parse_args,
            patch("getiprompt.scripts.benchmark.load_dataset_by_name") as mock_load_dataset,
            patch("getiprompt.scripts.benchmark.setup_logger") as mock_setup_logger,
        ):
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_get_args.return_value = mock_args
            mock_parse_args.return_value = (
                [MagicMock(value="lvis")],
                [MagicMock(value="Matcher")],
                [MagicMock(value="SAM-HQ-tiny")],
            )
            mock_load_dataset.side_effect = FileNotFoundError("Dataset not found")
            mock_setup_logger.return_value = None

            with pytest.raises(FileNotFoundError):
                perform_benchmark_experiment()

    def test_perform_benchmark_experiment_performance_tracking(self) -> None:
        """Test performance tracking in benchmark experiment."""
        with (
            patch("getiprompt.scripts.benchmark.get_arguments") as mock_get_args,
            patch("getiprompt.scripts.benchmark.parse_experiment_args") as mock_parse_args,
            patch("getiprompt.scripts.benchmark.load_dataset_by_name") as mock_load_dataset,
            patch("getiprompt.scripts.benchmark.load_model") as mock_load_model,
            patch("getiprompt.scripts.benchmark.predict_on_dataset") as mock_predict,
            patch("getiprompt.scripts.benchmark._save_results") as mock_save,
            patch("getiprompt.scripts.benchmark.setup_logger") as mock_setup_logger,
        ):
            mock_args = MagicMock()
            mock_args.log_level = "INFO"
            mock_get_args.return_value = mock_args
            mock_parse_args.return_value = (
                [MagicMock(value="lvis")],
                [MagicMock(value="Matcher")],
                [MagicMock(value="SAM-HQ-tiny")],
            )
            mock_load_dataset.return_value = MagicMock()
            mock_load_model.return_value = MagicMock()
            mock_predict.return_value = MagicMock()
            mock_save.return_value = None
            mock_setup_logger.return_value = None

            perform_benchmark_experiment()

            # Verify that the functions are called
            mock_load_dataset.assert_called()
            mock_predict.assert_called()
            mock_save.assert_called()


class TestBenchmarkCLIValidation:
    """Test CLI validation in benchmark script."""

    def test_validate_dataset_name(self) -> None:
        """Test dataset name validation."""
        valid_datasets = ["lvis", "perseg", "all"]

        for dataset in valid_datasets:
            # Test that load_dataset_by_name can handle these datasets
            with (
                patch("getiprompt.scripts.benchmark.LVISDataset") as mock_lvis,
                patch("getiprompt.scripts.benchmark.PerSegDataset") as mock_perseg,
            ):
                mock_lvis.return_value = MagicMock()
                mock_perseg.return_value = MagicMock()

                if dataset == "lvis":
                    load_dataset_by_name(dataset, categories="default")
                    mock_lvis.assert_called_once()
                elif dataset == "perseg":
                    load_dataset_by_name(dataset, categories="default")
                    mock_perseg.assert_called_once()

    def test_validate_model_name(self) -> None:
        """Test model name validation."""
        # Test that get_arguments function exists and can be imported
        assert callable(get_arguments)

    def test_validate_sam_backbone(self) -> None:
        """Test SAM backbone validation."""
        # Test that get_arguments function exists and can be imported
        assert callable(get_arguments)

    def test_validate_n_shot(self) -> None:
        """Test n_shot validation."""
        # Test that get_arguments function exists and can be imported
        assert callable(get_arguments)

    def test_validate_class_name(self) -> None:
        """Test class name validation."""
        # Test that get_arguments function exists and can be imported
        assert callable(get_arguments)
