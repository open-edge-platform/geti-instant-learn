# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from pathlib import Path
from uuid import UUID, uuid5

from domain.services.schemas.dataset import DatasetSchema, DatasetsListSchema
from settings import get_settings

logger = logging.getLogger(__name__)

# Fixed namespace for dataset ID generation. A project-specific UUID is used
# instead of the stdlib NAMESPACE_URL to avoid accidental collisions with other
# uuid5 calls elsewhere that happen to share the same name string.
_DATASET_NS = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class DatasetRegistryService:
    """Provides dataset metadata discovered from the filesystem."""

    def __init__(self) -> None:
        datasets_root = get_settings().template_dataset_dir
        logger.debug("Loading datasets from '%s'", datasets_root)
        self._datasets, self._paths = self._load_from_filesystem(datasets_root)
        logger.info("Loaded %d dataset(s) from '%s'", len(self._datasets), datasets_root)

    def list_datasets(self) -> DatasetsListSchema:
        """Return dataset metadata discovered from the template dataset directory."""
        return DatasetsListSchema(datasets=self._datasets)

    def get_dataset_path(self, dataset_id: UUID) -> Path:
        """Resolve dataset directory path by dataset id."""
        path = self._paths[dataset_id]
        logger.debug("Resolved dataset id '%s' to '%s'", dataset_id, path)
        return path

    @staticmethod
    def _load_from_filesystem(datasets_root: Path) -> tuple[list[DatasetSchema], dict[UUID, Path]]:
        datasets: list[DatasetSchema] = []
        paths: dict[UUID, Path] = {}
        if not datasets_root.exists():
            logger.warning("Template dataset directory '%s' does not exist, returning empty list", datasets_root)
            return datasets, paths
        for entry in sorted(datasets_root.iterdir()):
            if not entry.is_dir():
                logger.debug("Skipping non-directory entry '%s'", entry.name)
                continue
            name = entry.name.replace("-", " ").replace("_", " ").title()
            dataset_id = uuid5(_DATASET_NS, entry.name)
            logger.debug("Discovered dataset '%s' (id=%s) at '%s'", name, dataset_id, entry)
            datasets.append(
                DatasetSchema(
                    id=dataset_id,
                    name=name,
                    description=f"This is sample dataset of {name.lower()}.",
                )
            )
            paths[dataset_id] = entry
        return datasets, paths
