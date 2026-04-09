# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID, uuid5

from domain.errors import DatasetNotFoundError
from domain.services.schemas.base import Pagination
from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema
from runtime.services.image_thumbnail import generate_image_thumbnail
from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Fixed namespace for dataset ID generation.
DATASET_NS = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def _get_first_image(dataset_dir: Path) -> Path | None:
    for entry in sorted(dataset_dir.iterdir()):
        if entry.is_file() and entry.suffix.lower() in settings.supported_extensions:
            return entry
    return None


def resolve_dataset_path_from_cache(dataset_id: UUID | None, dataset_paths: Mapping[UUID, Path]) -> Path:
    """Resolve a dataset path from the startup cache.

    Args:
        dataset_id: Stable dataset UUID, or None to use the lexicographically first cached dataset directory.
        dataset_paths: Startup-cached dataset id-to-path mapping.

    Returns:
        Resolved dataset path.

    Raises:
        DatasetNotFoundError: If the dataset id is missing or no cached datasets exist.
    """
    if dataset_id is not None:
        try:
            return dataset_paths[dataset_id]
        except KeyError as exc:
            logger.warning("Sample dataset id '%s' could not be resolved from startup cache.", dataset_id)
            raise DatasetNotFoundError(f"Sample dataset id '{dataset_id}' was not found.") from exc

    if not dataset_paths:
        logger.warning("No sample datasets available in startup cache.")
        raise DatasetNotFoundError("No sample datasets available.")

    return min(dataset_paths.values(), key=lambda dataset_path: (dataset_path.name, str(dataset_path)))


def scan_datasets(datasets_root: Path) -> tuple[DatasetsListSchema, dict[UUID, Path]]:
    """Scan dataset root and return API schema plus internal id-to-path mapping."""
    datasets: list[DatasetSchema] = []
    dataset_paths: dict[UUID, Path] = {}

    if not datasets_root.exists():
        logger.error("Template dataset directory '%s' does not exist", datasets_root)
        raise DatasetNotFoundError(f"Template dataset directory '{datasets_root}' does not exist.")

    if not datasets_root.is_dir():
        logger.warning(
            "Template dataset path '%s' exists but is not a directory",
            datasets_root,
        )
        raise DatasetNotFoundError(f"Template dataset path '{datasets_root}' is not a directory.")

    for entry in sorted(datasets_root.iterdir()):
        if not entry.is_dir():
            logger.debug("Skipping non-directory entry '%s'", entry.name)
            continue

        dataset_id = uuid5(DATASET_NS, entry.name)
        dataset_paths[dataset_id] = entry
        name = entry.name.replace("-", " ").replace("_", " ").title()
        first_image = _get_first_image(entry)
        datasets.append(
            DatasetSchema(
                id=dataset_id,
                name=name,
                description=f"This is sample dataset of {name.lower()}.",
                thumbnail=generate_image_thumbnail(first_image) if first_image is not None else None,
            )
        )

    logger.info("Discovered %d dataset(s) under '%s'", len(dataset_paths), datasets_root)
    count = len(datasets)
    return (
        DatasetsListSchema(
            datasets=datasets,
            pagination=Pagination(count=count, total=count, offset=0, limit=count),
        ),
        dataset_paths,
    )
