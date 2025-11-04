# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Benchmark utilities."""

import shutil
import time
import warnings
from collections.abc import Callable
from functools import wraps
from logging import getLogger
from pathlib import Path
from typing import Any

import polars as pl

from getiprompt.types.results import Results
from getiprompt.utils.constants import DatasetName, ModelName, SAMModelName

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


logger = getLogger("Geti Prompt")


def prepare_output_directory(output_path: str, overwrite: bool) -> Path:
    """Handle output path to avoid overwriting existing data.

    Args:
        output_path: The path to the output data
        overwrite: Whether to overwrite existing data

    Raises:
        ValueError: If the output path already exists and overwrite is False

    Returns:
        The path to the output data
    """
    output_path_obj = Path(output_path)
    if output_path_obj.exists():
        if overwrite:
            shutil.rmtree(output_path_obj)
        else:
            msg = (
                f"Output path {output_path_obj} already exists. "
                "Set overwrite=True to overwrite it or change the output path."
            )
            raise ValueError(msg)

    output_path_obj.mkdir(parents=True, exist_ok=True)
    return output_path_obj


def _get_output_path_for_experiment(
    output_path: Path,
    experiment_name: str | None,
    dataset: DatasetName,
    model: ModelName,
    backbone: SAMModelName,
) -> Path:
    """Construct a unique output path for an experiment.

    Args:
        output_path: The path to save the results
        experiment_name: The name of the experiment
        dataset: The dataset to run
        model: The model to run
        backbone: The backbone to run

    Returns:
        The path to save the results
    """
    combo_str = f"{dataset.value}_{backbone.value}_{model.value}"

    if experiment_name:
        return output_path / experiment_name / combo_str

    return output_path / combo_str


def _save_results(all_results: list[pl.DataFrame], output_path: Path) -> None:
    """Concatenate and save all experiment results.

    Args:
        all_results: The results to save
        output_path: The path to save the results
    """
    if not all_results:
        logger.warning("No experiments were run. Check your arguments.")
        return

    all_result_dataframe = pl.concat(all_results)
    all_results_dataframe_filename = output_path / "all_results.csv"
    all_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    all_result_dataframe.write_csv(str(all_results_dataframe_filename))
    msg = f"Saved all results to: {all_results_dataframe_filename}"
    logger.info(msg)

    avg_results_dataframe_filename = output_path / "avg_results.csv"
    avg_results_dataframe_filename.parent.mkdir(parents=True, exist_ok=True)
    avg_result_dataframe = all_result_dataframe.group_by(
        ["dataset_name", "model_name", "backbone_name"],
    ).mean()
    avg_result_dataframe.write_csv(str(avg_results_dataframe_filename))
    msg = f"Saved average results to: {avg_results_dataframe_filename}"
    logger.info(msg)
    msg = f"\n\n Final Average Results:\n {avg_result_dataframe}"
    logger.info(msg)


def track_duration(func: Callable[..., Results]) -> Callable[..., Results]:
    """Decorator to track the duration of a method."""

    @wraps(func)
    def wrapper(self, *args, **kwargs) -> Any:  # noqa: ANN001,ANN401, ANN003, ANN002
        start_time = time.perf_counter()
        result = func(self, *args, **kwargs)
        if isinstance(result, Results):
            result.duration = time.perf_counter() - start_time
        return result

    return wrapper
