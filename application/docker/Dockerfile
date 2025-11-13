FROM python:3.12-slim-bookworm@sha256:3291ae895c4a3af495196e48292a19fad4ea7ce82528b91cca64961a4d04f538 AS backend-base
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
# across both images.
ENV UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /bin/uv
WORKDIR /geti_prompt/

# Install git for git dependencies (sam-2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git=1:2.39.5-0+deb12u2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy library and create proper directory structure
COPY --link --from=libs ./src /geti_prompt/library/src
COPY --link --from=libs ./pyproject.toml /geti_prompt/library/pyproject.toml

# Copy backend source code
COPY --link backend/app ./app

# Change to a subdirectory so ../../library resolves correctly
RUN mkdir -p /geti_prompt/app/backend
WORKDIR /geti_prompt/app/backend

FROM backend-base AS cpu-base

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra cpu

#############
# geti_ui_packages
#############
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS geti_ui_packages

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# web_ui deps
#############
FROM node:24-alpine3.22@sha256:3e843c608bb5232f39ecb2b25e41214b958b0795914707374c8acc28487dea17 AS ui_base

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

######################
# Server (CPU version)
######################
FROM python:3.12-slim-bookworm@sha256:3291ae895c4a3af495196e48292a19fad4ea7ce82528b91cca64961a4d04f538 AS geti-prompt-cpu

ARG STATIC_FILES_DIR=""
ENV STATIC_FILES_DIR=${STATIC_FILES_DIR}
ARG DB_DATA_DIR=""
ENV DB_DATA_DIR=${DB_DATA_DIR}
ARG MOCKED_FILE=""
ENV MOCKED_FILE=${MOCKED_FILE}
ENV PATH="/geti_prompt/.venv/bin:$PATH"
ENV PYTHONPATH=/geti_prompt/app

# Create directory and set ownership before VOLUME instruction
RUN mkdir -p ${DB_DATA_DIR} && chown 10001:10001 ${DB_DATA_DIR}
VOLUME ${DB_DATA_DIR}

# Install OpenGL libraries for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1=1.6.0-1 \
    libglib2.0-0=2.74.6-2+deb12u7 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd -l -u 10001 non-root && \
    usermod -G video non-root && \
    mkdir -p /home/non-root/.cache/uv && \
    chown -R non-root:non-root /home/non-root && \
    pip3 uninstall -y setuptools pip wheel && \
    rm -rf /root/.cache/pip

USER non-root

COPY --from=cpu-base --chown=10001 /geti_prompt/app/backend/.venv /geti_prompt/.venv
COPY --from=backend-base --chown=10001 /geti_prompt/app /geti_prompt/app
COPY --from=backend-base --chown=10001 /geti_prompt/library /geti_prompt/library
COPY --from=backend-base --chown=10001 /bin/uv /bin/uv
COPY --from=web_ui --chown=10001 /home/app/web_ui/dist/ ${STATIC_FILES_DIR}
# temporary file for testing
COPY --from=web_ui --chown=10001 /home/app/web_ui/src/assets/test.webp ${MOCKED_FILE}

WORKDIR /geti_prompt

CMD ["uv", "run", "app/main.py"]
