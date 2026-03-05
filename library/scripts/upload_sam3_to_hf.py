# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Upload SAM3 OpenVINO models to HuggingFace Hub.

Uploads OpenVINO IR files (.xml/.bin) and tokenizer files to a HuggingFace
repository for distribution.

Usage:
    # Upload all models from the default directory
    python upload_sam3_to_hf.py --model-dir ./sam3-openvino/openvino-fp16

    # Upload to a specific repo
    python upload_sam3_to_hf.py --model-dir ./sam3-openvino/openvino-fp16 --repo-id rajeshgangireddy/exported_sam3

    # Dry run (list files without uploading)
    python upload_sam3_to_hf.py --model-dir ./sam3-openvino/openvino-fp16 --dry-run
"""

import argparse
import logging
import sys
from pathlib import Path

from huggingface_hub import HfApi

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "rajeshgangireddy/exported_sam3"

# Files to upload (OpenVINO IR + ONNX + tokenizer)
UPLOAD_PATTERNS = [
    "*.xml",
    "*.bin",
    "*.onnx",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def collect_files(model_dir: Path) -> list[Path]:
    """Collect all files to upload from the model directory.

    Args:
        model_dir: Directory containing OpenVINO IR and tokenizer files.

    Returns:
        List of file paths to upload.
    """
    files = []
    for pattern in UPLOAD_PATTERNS:
        files.extend(model_dir.glob(pattern))
    return sorted(set(files))


def upload_to_hf(
    model_dir: Path,
    repo_id: str = DEFAULT_REPO_ID,
    commit_message: str = "Upload SAM3 OpenVINO IR models (FP16)",
    dry_run: bool = False,
    path_in_repo: str = "",
) -> None:
    """Upload OpenVINO models and tokenizer files to HuggingFace Hub.

    Args:
        model_dir: Directory containing files to upload.
        repo_id: HuggingFace repository ID (e.g., "user/repo").
        commit_message: Commit message for the upload.
        dry_run: If True, list files without uploading.
        path_in_repo: Subdirectory in the repo to upload files to.
    """
    files = collect_files(model_dir)

    if not files:
        logger.error("No files found in %s matching upload patterns.", model_dir)
        return

    logger.info("Files to upload to %s:", repo_id)
    total_size = 0
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024)
        total_size += size_mb
        logger.info("  %s (%.1f MB)", f.name, size_mb)
    logger.info("Total: %d files, %.1f MB", len(files), total_size)

    if dry_run:
        logger.info("Dry run — no files uploaded.")
        return

    api = HfApi()

    # Ensure the repo exists
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload all files in a single commit
    operations = []
    for f in files:
        operations.append(f)
        logger.info("Uploading %s...", f.name)

    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=commit_message,
        allow_patterns=list(UPLOAD_PATTERNS),
        path_in_repo=path_in_repo or None,
    )

    logger.info("Upload complete! View at: https://huggingface.co/%s", repo_id)


def main() -> None:
    """CLI entry point for uploading SAM3 OpenVINO models to HuggingFace."""
    parser = argparse.ArgumentParser(
        description="Upload SAM3 OpenVINO models to HuggingFace Hub.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory containing OpenVINO IR (.xml/.bin) and tokenizer files.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"HuggingFace repository ID. Default: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload SAM3 OpenVINO IR models (FP16)",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without uploading.",
    )
    parser.add_argument(
        "--path-in-repo",
        type=str,
        default="",
        help="Subdirectory in the HF repo to upload files into (e.g., 'nncf-int8').",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(name)s: %(message)s",
        stream=sys.stdout,
    )

    if not args.model_dir.exists():
        logger.error("Model directory does not exist: %s", args.model_dir)
        sys.exit(1)

    upload_to_hf(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        commit_message=args.commit_message,
        dry_run=args.dry_run,
        path_in_repo=args.path_in_repo,
    )


if __name__ == "__main__":
    main()
