# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

    current_dir: Path = Path(__file__).parent.resolve()

    # Application
    app_name: str = "Geti Prompt"
    version: str = "0.1.0"
    summary: str = "Geti Prompt server"
    description: str = (
        "Geti Prompt is a modular framework for few-shot visual segmentation using visual prompting techniques. "
        "Enables easy experimentation with different algorithms, backbones (SAM, MobileSAM, EfficientViT-SAM, DinoV2), "
        "and project components for finding and segmenting objects from just a few examples."
    )
    openapi_url: str = "/api/openapi.json"
    debug: bool = Field(default=False, alias="DEBUG")
    log_format: str = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"
    environment: Literal["dev", "prod"] = "dev"

    static_files_dir: str | None = Field(default=None, alias="STATIC_FILES_DIR")

    # Runtime
    device: Literal["cpu", "cuda", "xpu"] = Field(default="cpu", alias="DEVICE")

    # Server
    host: str = Field(default="localhost", alias="HOST")
    port: int = Field(default=9100, alias="PORT")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000, http://localhost:9100",
        alias="CORS_ORIGINS",
    )

    # Database
    db_data_dir: Path = Field(default=current_dir.parent / ".data", alias="DB_DATA_DIR")
    db_filename: str = "geti_prompt.db"

    # Template datasets
    template_dataset_path: str = Field(default="templates/datasets/coffee-berries", alias="TEMPLATE_DATASET_PATH")

    @property
    def template_dataset_dir(self) -> Path:
        """Full path to the template dataset directory"""
        return self.db_data_dir / self.template_dataset_path

    @property
    def database_url(self) -> str:
        """Database connection URL"""
        return f"sqlite:///{self.db_data_dir / self.db_filename}"

    @property
    def cors_allowed_origins(self) -> list[str]:
        """Parsed list of allowed CORS origins."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    # Logs
    logs_dir: Path = Field(default=current_dir.parent / ".logs", alias="LOGS_DIR")

    @property
    def log_file(self) -> str:
        """Log file location"""
        return str(self.logs_dir / "geti-prompt-backend.log")

    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    alembic_config_path: str = str(current_dir / "alembic.ini")
    alembic_script_location: str = str(current_dir / "domain" / "alembic")

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")

    # Supported file formats
    supported_extensions: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    # Thumbnail generation
    thumbnail_max_dimension: int = 300
    thumbnail_line_thickness_ratio: float = 0.005  # 0.5% of smaller image dimension
    thumbnail_min_line_thickness: int = 2
    thumbnail_fill_opacity: float = 0.5  # 50% opacity for annotation fill
    thumbnail_jpeg_quality: int = 85

    # WebRTC
    webrtc_advertise_ip: str | None = Field(default=None, alias="WEBRTC_ADVERTISE_IP")

    # Simplified WebRTC config
    coturn_host: str | None = Field(default=None, alias="COTURN_HOST")
    coturn_port: int = Field(default=3478, alias="COTURN_PORT")
    coturn_username: str = Field(default="user", alias="COTURN_USERNAME")
    coturn_password: str = Field(default="password", alias="COTURN_PASSWORD")
    stun_server: str | None = Field(default=None, alias="STUN_SERVER")

    @property
    def ice_servers(self) -> list[dict]:
        """Compute ICE servers from coturn and STUN configuration."""
        servers = []
        if self.coturn_host:
            servers.append(
                {
                    "urls": f"turn:{self.coturn_host}:{self.coturn_port}?transport=tcp",
                    "username": self.coturn_username,
                    "credential": self.coturn_password,
                }
            )

        if self.stun_server:
            servers.append({"urls": self.stun_server})

        return servers

    @field_validator("static_files_dir", "alembic_config_path", "alembic_script_location", mode="after")
    def prefix_paths(cls, v: str | None) -> str | None:
        if v and getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            # If application is running in pyinstaller bundle, adjust the path accordingly.
            return os.path.join(getattr(sys, "_MEIPASS", ""), v)
        return v


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
