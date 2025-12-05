# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Application configuration management"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore")

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
    environment: Literal["dev", "prod"] = "dev"
    static_files_dir: str | None = Field(default=None, alias="STATIC_FILES_DIR")

    # Server
    host: str = Field(default="localhost", alias="HOST")
    port: int = Field(default=9100, alias="PORT")

    # CORS
    cors_origins: str = Field(
        default="http://localhost:3000, http://localhost:9100",
        alias="CORS_ORIGINS",
    )

    # Database
    current_dir: Path = Path(__file__).parent.resolve()
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


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
