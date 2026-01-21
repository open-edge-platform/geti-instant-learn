"""Checkpoint loading strategies for EfficientSAM3 models.

This module provides different strategies for loading EfficientSAM3 model checkpoints:
- Unified loading: Single checkpoint containing both image and text encoder
- Separate loading: Separate checkpoints for image encoder, text encoder, and optional tracker

Follows Strategy Pattern for extensibility while maintaining backward compatibility.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

    from getiprompt.models.foundation.efficientsam3.efficientsam3_image import EfficientSAM3Image

logger = logging.getLogger(__name__)


@dataclass
class EfficientSAM3CheckpointPaths:
    """Value object for EfficientSAM3 checkpoint paths.

    Args:
        unified: Path to unified checkpoint containing both encoders
        image_encoder: Path to separate image encoder checkpoint
        text_encoder: Path to separate text encoder checkpoint
        tracker: Path to separate tracker/SAM heads checkpoint
    """

    unified: Path | None = None
    image_encoder: Path | None = None
    text_encoder: Path | None = None
    tracker: Path | None = None

    def is_unified(self) -> bool:
        """Check if using unified checkpoint."""
        return self.unified is not None

    def is_separate(self) -> bool:
        """Check if using separate checkpoints."""
        return self.image_encoder is not None or self.text_encoder is not None


class EfficientSAM3CheckpointLoader(ABC):
    """Abstract base class for EfficientSAM3 checkpoint loading strategies."""

    @abstractmethod
    def load(
        self,
        model: EfficientSAM3Image,
        paths: EfficientSAM3CheckpointPaths,
        enable_inst_interactivity: bool = False,
    ) -> None:
        """Load checkpoint weights into model.

        Args:
            model: EfficientSAM3 model to load weights into
            paths: Checkpoint paths to load from
            enable_inst_interactivity: Whether to load tracker/SAM heads weights
        """


class EfficientSAM3UnifiedCheckpointLoader(EfficientSAM3CheckpointLoader):
    """Loads EfficientSAM3 from a single unified checkpoint.

    This is the default loading strategy, compatible with the official
    EfficientSAM3 checkpoints that contain both image and text encoder weights.
    """

    def load(
        self,
        model: EfficientSAM3Image,
        paths: EfficientSAM3CheckpointPaths,
        enable_inst_interactivity: bool = False,
    ) -> None:
        """Load from unified checkpoint containing both encoders.

        Args:
            model: EfficientSAM3 model to load weights into
            paths: Must have unified path set
            enable_inst_interactivity: Whether to load tracker/SAM heads weights

        Raises:
            ValueError: If unified checkpoint path is not provided
        """
        if not paths.is_unified():
            raise ValueError("EfficientSAM3UnifiedCheckpointLoader requires unified checkpoint path")

        self._load_unified_checkpoint(model, paths.unified, enable_inst_interactivity)

    def _load_unified_checkpoint(
        self,
        model: EfficientSAM3Image,
        checkpoint_path: Path,
        enable_inst_interactivity: bool,
    ) -> None:
        """Load unified checkpoint into model.

        Args:
            model: Model to load weights into
            checkpoint_path: Path to unified checkpoint file
            enable_inst_interactivity: Whether to load tracker weights
        """
        # Import here to avoid circular dependency
        from iopath.common.file_io import g_pathmgr

        logger.info(f"Loading EfficientSAM3 unified checkpoint from: {checkpoint_path}")

        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)

        # Unwrap nested "model" key if present
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        efficientsam3_state_dict = {}

        # Load detector (image encoder + text encoder) weights
        for k, v in ckpt.items():
            if "detector" in k:
                # Remove "detector." prefix
                new_key = k.replace("detector.", "")
                # Fix extra ".model" layer in backbone path
                # e.g., "trunk.model.backbone.stages" -> "trunk.backbone.stages"
                new_key = new_key.replace("trunk.model.", "trunk.")
                efficientsam3_state_dict[new_key] = v

        # Load tracker (SAM heads) weights if enabled
        if enable_inst_interactivity:
            for k, v in ckpt.items():
                if "tracker" in k:
                    new_key = k.replace("tracker.", "")
                    # Only load SAM-related weights
                    if new_key.startswith("sam_") or new_key == "no_mem_embed":
                        efficientsam3_state_dict[new_key] = v

        # Load state dict with non-strict matching
        missing_keys, unexpected_keys = model.load_state_dict(
            efficientsam3_state_dict,
            strict=False,
        )

        logger.debug(f"Missing keys: {missing_keys}")
        logger.debug(f"Unexpected keys: {unexpected_keys}")
        logger.info(f"Successfully loaded EfficientSAM3 checkpoint from {checkpoint_path}")


class EfficientSAM3SeparateCheckpointLoader(EfficientSAM3CheckpointLoader):
    """Loads EfficientSAM3 from separate image and text encoder checkpoints.

    This strategy allows loading image encoder and text encoder from different
    checkpoint files, enabling mix-and-match of encoders and more flexible
    model composition.
    """

    def load(
        self,
        model: EfficientSAM3Image,
        paths: EfficientSAM3CheckpointPaths,
        enable_inst_interactivity: bool = False,
    ) -> None:
        """Load from separate checkpoints for each component.

        Args:
            model: EfficientSAM3 model to load weights into
            paths: Must have image_encoder and/or text_encoder paths set
            enable_inst_interactivity: Whether to load tracker weights

        Raises:
            ValueError: If no separate checkpoint paths are provided
        """
        if not paths.is_separate():
            raise ValueError(
                "EfficientSAM3SeparateCheckpointLoader requires separate checkpoint paths "
                "(image_encoder and/or text_encoder)",
            )

        # Load image encoder if provided
        if paths.image_encoder:
            self._load_efficientsam3_image_encoder_checkpoint(
                model.backbone.visual,
                paths.image_encoder,
            )

        # Load text encoder if provided
        if paths.text_encoder:
            self._load_efficientsam3_text_encoder_checkpoint(
                model.backbone.text,
                paths.text_encoder,
            )

        # Load tracker if provided and enabled
        if paths.tracker and enable_inst_interactivity:
            self._load_efficientsam3_tracker_checkpoint(model, paths.tracker)

    def _load_efficientsam3_image_encoder_checkpoint(
        self,
        vision_encoder: nn.Module,
        checkpoint_path: Path,
    ) -> None:
        """Load image encoder weights from checkpoint.

        Args:
            vision_encoder: Vision encoder module (Sam3DualViTDetNeck)
            checkpoint_path: Path to image encoder checkpoint
        """
        from iopath.common.file_io import g_pathmgr

        logger.info(f"Loading EfficientSAM3 image encoder from: {checkpoint_path}")

        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        # Filter and transform keys for vision encoder
        vision_state_dict = {}
        for k, v in ckpt.items():
            # Handle trunk (backbone) keys
            if k.startswith("trunk."):
                new_key = k.replace("trunk.model.", "trunk.")
                vision_state_dict[new_key] = v
            # Handle neck keys
            elif k.startswith("neck."):
                vision_state_dict[k] = v
            # Handle direct image encoder keys
            elif "image_encoder" in k:
                new_key = k.replace("image_encoder.", "trunk.")
                new_key = new_key.replace("trunk.model.", "trunk.")
                vision_state_dict[new_key] = v

        missing_keys, _ = vision_encoder.load_state_dict(vision_state_dict, strict=False)
        logger.debug(f"Missing keys in image encoder: {missing_keys}")
        logger.info("Successfully loaded EfficientSAM3 image encoder")

    def _load_efficientsam3_text_encoder_checkpoint(
        self,
        text_encoder: nn.Module,
        checkpoint_path: Path,
    ) -> None:
        """Load text encoder weights from checkpoint.

        Args:
            text_encoder: Text encoder module (VETextEncoder or TextStudentEncoder)
            checkpoint_path: Path to text encoder checkpoint
        """
        from iopath.common.file_io import g_pathmgr

        logger.info(f"Loading EfficientSAM3 text encoder from: {checkpoint_path}")

        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        # Filter keys for text encoder
        text_state_dict = {}
        for k, v in ckpt.items():
            if k.startswith("text_encoder."):
                text_state_dict[k.replace("text_encoder.", "")] = v
            elif k.startswith("text."):
                text_state_dict[k.replace("text.", "")] = v
            # Handle direct text encoder keys (already properly prefixed)
            elif not k.startswith("image") and not k.startswith("trunk"):
                text_state_dict[k] = v

        missing_keys, _ = text_encoder.load_state_dict(text_state_dict, strict=False)
        logger.debug(f"Missing keys in text encoder: {missing_keys}")
        logger.info("Successfully loaded EfficientSAM3 text encoder")

    def _load_efficientsam3_tracker_checkpoint(
        self,
        model: EfficientSAM3Image,
        checkpoint_path: Path,
    ) -> None:
        """Load SAM tracker/interactive predictor weights from checkpoint.

        Args:
            model: Full EfficientSAM3 model
            checkpoint_path: Path to tracker checkpoint
        """
        from iopath.common.file_io import g_pathmgr

        logger.info(f"Loading EfficientSAM3 tracker from: {checkpoint_path}")

        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        # Filter tracker keys
        tracker_state_dict = {}
        for k, v in ckpt.items():
            if "tracker" in k:
                new_key = k.replace("tracker.", "")
                # Only load SAM-related weights
                if new_key.startswith("sam_") or new_key == "no_mem_embed":
                    tracker_state_dict[new_key] = v

        missing_keys, _ = model.load_state_dict(tracker_state_dict, strict=False)
        logger.debug(f"Missing keys in tracker: {missing_keys}")
        logger.info("Successfully loaded EfficientSAM3 tracker")


def get_efficientsam3_checkpoint_loader(
    strategy: str = "unified",
) -> EfficientSAM3CheckpointLoader:
    """Factory function to get appropriate checkpoint loader.

    Args:
        strategy: Loading strategy - "unified" or "separate"

    Returns:
        Checkpoint loader instance

    Raises:
        ValueError: If strategy is not recognized

    Examples:
        >>> loader = get_efficientsam3_checkpoint_loader("unified")
        >>> paths = EfficientSAM3CheckpointPaths(unified=Path("model.pth"))
        >>> loader.load(model, paths, enable_inst_interactivity=False)
    """
    if strategy == "unified":
        return EfficientSAM3UnifiedCheckpointLoader()
    if strategy == "separate":
        return EfficientSAM3SeparateCheckpointLoader()
    raise ValueError(
        f"Unknown checkpoint loading strategy: {strategy}. Must be 'unified' or 'separate'.",
    )
