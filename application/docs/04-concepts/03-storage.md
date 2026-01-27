# Storage

This document describes how Geti Prompt stores and manages persistent data, including the file system layout, database configuration, and volume mounting for Docker deployments.

## Overview

Geti Prompt uses a SQLite database for configuration persistence and a file-based structure for media assets (datasets, prompts, thumbnails).

## File System Layout

```
data/                              # DB_DATA_DIR
├── geti_prompt.db                 # SQLite database
└── templates/
    └── datasets/
        └── coffee-berries/        # Sample dataset

logs/                              # LOGS_DIR
└── geti-prompt-backend.log        # Application log
```

## Database

### SQLite Configuration

Geti Prompt uses **SQLite** as its embedded database for storing:

- Project configurations
- Source settings (cameras, video files, image folders)
- Sink configurations (MQTT, custom outputs)
- Model selections and parameters
- Prompt definitions (visual and text prompts)
- Label schemas

### Schema Management

Database migrations are managed with [Alembic](https://alembic.sqlalchemy.org/). Migration scripts are located in `application/backend/app/domain/alembic/versions/`.

## Docker Volume Mounting

### Container Mount Points

| Container Path         | Purpose                           |
| ---------------------- | --------------------------------- |
| `/geti_prompt/data`    | Database and user data            |
| `/geti_prompt/logs`    | Application logs                  |

> **Note:** The container runs as non-root user (UID 10001). For bind mounts, ensure host directories have correct permissions.

## Configuration Reference

### Environment Variables

| Variable        | Description                    |
| --------------- | ------------------------------ |
| `DB_DATA_DIR`   | Database and data directory    |
| `LOGS_DIR`      | Log files directory            |
