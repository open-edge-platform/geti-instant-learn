# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Protocol-aware model loading.

A model directory can live on the local filesystem or on a remote model hub
(Hugging Face, S3, ...). :func:`resolve_model_dir` inspects the URI scheme and
dispatches to the matching :class:`ModelLoader`, returning a local directory
that concrete models can read ``.xml`` / ``.bin`` files from.

Currently shipped: local filesystem and Hugging Face Hub. S3 is stubbed behind
the same interface so it can be added without touching call sites.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse


class ModelLoader(ABC):
    """Strategy that fetches a model directory for a given URI scheme."""

    @staticmethod
    @abstractmethod
    def can_handle(scheme: str) -> bool:
        """Return ``True`` if this loader handles URIs with *scheme*.

        Args:
            scheme: URL scheme (``""`` for bare local paths).
        """

    @staticmethod
    @abstractmethod
    def fetch(uri: str) -> Path:
        """Resolve *uri* to a local directory and return its path."""


class LocalModelLoader(ModelLoader):
    """Loads models already present on the local filesystem."""

    @staticmethod
    def can_handle(scheme: str) -> bool:
        """Handle bare paths and ``file://`` URIs."""
        return scheme in {"", "file"}

    @staticmethod
    def fetch(uri: str) -> Path:
        """Return the local path, stripping a ``file://`` prefix if present.

        Raises:
            FileNotFoundError: If the resolved path does not exist.
        """
        path = Path(uri.removeprefix("file://"))
        if not path.exists():
            msg = f"Model directory not found: {path}"
            raise FileNotFoundError(msg)
        return path


class HuggingFaceModelLoader(ModelLoader):
    """Downloads a model snapshot from the Hugging Face Hub.

    URI form: ``hf://<repo_id>`` (optionally ``hf://<repo_id>@<revision>``).
    """

    @staticmethod
    def can_handle(scheme: str) -> bool:
        """Handle ``hf://`` URIs."""
        return scheme == "hf"

    @staticmethod
    def fetch(uri: str) -> Path:
        """Download the snapshot and return the local cache directory."""
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        repo = uri[len("hf://") :]
        repo_id, _, revision = repo.partition("@")
        local_dir = snapshot_download(repo_id=repo_id, revision=revision or None)
        return Path(local_dir)


class S3ModelLoader(ModelLoader):
    """Stub for loading models from S3 (not yet implemented)."""

    @staticmethod
    def can_handle(scheme: str) -> bool:
        """Handle ``s3://`` URIs."""
        return scheme == "s3"

    @staticmethod
    def fetch(uri: str) -> Path:
        """Not implemented yet.

        Raises:
            NotImplementedError: Always — S3 loading is not wired up.
        """
        msg = "S3 model loading is not implemented yet."
        raise NotImplementedError(msg)


#: Registered loaders, checked in order.
_LOADERS: tuple[ModelLoader, ...] = (
    LocalModelLoader(),
    HuggingFaceModelLoader(),
    S3ModelLoader(),
)


def resolve_model_dir(uri: str | Path) -> Path:
    """Resolve *uri* to a local model directory using the right loader.

    Args:
        uri: A local path, ``file://``, ``hf://``, or ``s3://`` URI.

    Returns:
        Local directory containing the model files.

    Raises:
        ValueError: If no loader handles the URI scheme.
    """
    uri_str = str(uri)
    scheme = urlparse(uri_str).scheme
    # On Windows a drive letter (e.g. ``C:``) parses as a scheme — treat single
    # letter schemes as local paths.
    if len(scheme) == 1:
        scheme = ""

    for loader in _LOADERS:
        if loader.can_handle(scheme):
            return loader.fetch(uri_str)

    msg = f"No model loader registered for URI scheme '{scheme}': {uri_str}"
    raise ValueError(msg)
