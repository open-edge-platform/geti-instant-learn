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

    # Server
    host: str = Field(default="0.0.0.0", alias="HOST")  # noqa: S104
    port: int = Field(default=9100, alias="PORT")

    # Database
    database_url: str = Field(
        default="sqlite:///./geti_prompt.db", alias="DATABASE_URL", description="Database connection URL"
    )
    db_echo: bool = Field(default=False, alias="DB_ECHO")

    # Alembic
    current_dir: Path = Path(__file__).parent.resolve()
    alembic_config_path: str = str(current_dir / "alembic.ini")
    alembic_script_location: str =str(current_dir / "alembic")

    # Proxy settings
    no_proxy: str = Field(default="localhost,127.0.0.1,::1", alias="no_proxy")


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()
