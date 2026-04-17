# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
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


class DatasetResolver:
    """Service for resolving dataset paths from a cached dataset mapping.

    This class owns the dataset cache and provides methods to resolve dataset paths.
    The cache is built during initialization and remains static for the lifetime of the instance.
    """

    def __init__(self, datasets_root: Path) -> None:
        """Initialize the resolver by scanning the dataset root directory.

        Args:
            datasets_root: Root directory containing dataset subdirectories.

        Raises:
            DatasetNotFoundError: If the datasets root does not exist or is not a directory.
        """
        logger.info("Initializing DatasetResolver")
        self._datasets_schema, self._dataset_paths = scan_datasets(datasets_root)
        if self._dataset_paths:
            dataset_lines = "\n".join(
                f"- {dataset_id}: {path}" for dataset_id, path in sorted(self._dataset_paths.items())
            )
            logger.debug("Cached dataset paths:\n%s", dataset_lines)
        logger.info("Cached %d dataset(s)", len(self._dataset_paths))

    def get_datasets(self) -> DatasetsListSchema:
        """Get the list of available datasets.

        Returns:
            Schema containing all discovered datasets with metadata.
        """
        return self._datasets_schema

    def get_dataset_path(self, dataset_id: UUID | None = None) -> Path:
        """Resolve a dataset path from the cache.

        Args:
            dataset_id: Dataset UUID, or None to use the lexicographically first cached dataset.

        Returns:
            Resolved dataset path.

        Raises:
            DatasetNotFoundError: If the dataset id is not found in the cache or no cached datasets exist.
        """
        if dataset_id is not None:
            try:
                return self._dataset_paths[dataset_id]
            except KeyError as exc:
                logger.warning("Sample dataset id '%s' could not be resolved from startup cache.", dataset_id)
                raise DatasetNotFoundError(f"Sample dataset id '{dataset_id}' was not found.") from exc

        if not self._dataset_paths:
            logger.warning("No sample datasets available in startup cache.")
            raise DatasetNotFoundError("No sample datasets available.")

        return min(self._dataset_paths.values(), key=lambda dataset_path: (dataset_path.name, str(dataset_path)))
