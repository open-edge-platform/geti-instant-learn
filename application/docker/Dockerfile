FROM python:3.13-slim-bookworm@sha256:e66df2153a7cc47b4438848efb65e2d9442db4330db9befaee5107fc75464959 AS base
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images.
ENV UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:0.7.13@sha256:6c1e19020ec221986a210027040044a5df8de762eb36d5240e382bc41d7a9043 /uv /bin/uv

FROM base AS backend_base

# Set working directory
ARG WORKDIR_PATH=""
WORKDIR ${WORKDIR_PATH}

ENV PYTHONPATH=/app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable

COPY --link backend/app ./app

#############
# geti_ui_packages
#############
FROM node:24-alpine3.22@sha256:f36fed0b2129a8492535e2853c64fbdbd2d29dc1219ee3217023ca48aebd3787 AS geti_ui_packages

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# web_ui deps
#############
FROM node:24-alpine3.22@sha256:f36fed0b2129a8492535e2853c64fbdbd2d29dc1219ee3217023ca48aebd3787 AS ui_base

WORKDIR /home/app/web_ui/

COPY --link ui/package.json ./
COPY --link ui/package-lock.json ./
COPY --from=geti_ui_packages /home/app/web_ui/packages/ /home/app/web_ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link ui/tsconfig.json ./
COPY --link ui/rsbuild.config.ts ./
COPY --link ui/src/ src/

ARG PUBLIC_API_URL=""
ENV PUBLIC_API_URL=${PUBLIC_API_URL}

#############
# web_ui build
#############
FROM ui_base AS web_ui

ARG ASSET_PREFIX=""
ENV ASSET_PREFIX=${ASSET_PREFIX}
# build with --no-cache in case of sub-dependency changes (from geti)
RUN npm run build

#############
# Server
#############
FROM backend_base AS geti-prompt

ARG STATIC_FILES_DIR=""
ENV STATIC_FILES_DIR=${STATIC_FILES_DIR}
ARG DB_DATA_DIR=""
ENV DB_DATA_DIR=${DB_DATA_DIR}
ARG ASSET_PREFIX=""
ENV ASSET_PREFIX=${ASSET_PREFIX}
ARG MOCKED_FILE=""
ENV MOCKED_FILE=${MOCKED_FILE}

VOLUME ${DB_DATA_DIR}

COPY --from=web_ui /home/app/web_ui/dist/ ${STATIC_FILES_DIR}
# temporary file for testing
COPY --from=web_ui /home/app/web_ui/src/assets/test.webp ${MOCKED_FILE}

CMD ["uv", "run", "app/main.py"]
