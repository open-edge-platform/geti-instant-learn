# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from uuid import UUID, uuid5

from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema

logger = logging.getLogger(__name__)

# Fixed namespace for dataset ID generation.
DATASET_NS = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


def scan_datasets(datasets_root: Path) -> tuple[DatasetsListSchema, dict[UUID, Path]]:
    """Scan dataset root and return API schema plus internal id-to-path mapping."""
    datasets: list[DatasetSchema] = []
    dataset_paths: dict[UUID, Path] = {}

    if not datasets_root.exists():
        logger.warning("Template dataset directory '%s' does not exist, returning empty list", datasets_root)
        return DatasetsListSchema(datasets=[]), dataset_paths

    if not datasets_root.is_dir():
        logger.warning(
            "Template dataset path '%s' exists but is not a directory, returning empty list",
            datasets_root,
        )
        return DatasetsListSchema(datasets=[]), dataset_paths

    for entry in sorted(datasets_root.iterdir()):
        if not entry.is_dir():
            logger.debug("Skipping non-directory entry '%s'", entry.name)
            continue

        dataset_id = uuid5(DATASET_NS, entry.name)
        dataset_paths[dataset_id] = entry
        name = entry.name.replace("-", " ").replace("_", " ").title()
        datasets.append(
            DatasetSchema(
                id=dataset_id,
                name=name,
                description=f"This is sample dataset of {name.lower()}.",
            )
        )

    logger.info("Discovered %d dataset(s) under '%s'", len(dataset_paths), datasets_root)
    return DatasetsListSchema(datasets=datasets), dataset_paths
