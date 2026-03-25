# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from uuid import UUID, uuid5

from domain.services.schemas.base import Pagination
from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema
from runtime.errors import DatasetNotFoundError
from runtime.services.image_thumbnail import generate_image_thumbnail
from settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Fixed namespace for dataset ID generation.
DATASET_NS = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def resolve_dataset_path(dataset_id: UUID, template_dataset_dir: Path) -> Path | None:
    """Resolve stable dataset ID to a directory path under template datasets."""
    if not template_dataset_dir.exists() or not template_dataset_dir.is_dir():
        logger.warning("Template dataset directory '%s' is not available.", template_dataset_dir)
        return None

    for entry in sorted(template_dataset_dir.iterdir()):
        if entry.is_dir() and uuid5(DATASET_NS, entry.name) == dataset_id:
            logger.debug("Resolved sample dataset id '%s' to '%s'.", dataset_id, entry)
            return entry

    return None


def get_first_dataset_path(template_dataset_dir: Path) -> Path | None:
    """Return the first available dataset directory under template datasets."""
    if not template_dataset_dir.exists() or not template_dataset_dir.is_dir():
        logger.warning("Template dataset directory '%s' is not available.", template_dataset_dir)
        return None
    return next((entry for entry in sorted(template_dataset_dir.iterdir()) if entry.is_dir()), None)


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
