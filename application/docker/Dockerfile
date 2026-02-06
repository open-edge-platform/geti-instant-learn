################
# Libraries Base
################
FROM python:3.12-slim-bookworm@sha256:ae30458df6c5167adfff1131d78f8c306805a35bb20d8333c20ddfb70715f5e2 AS backend-base
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Disable Python downloads, because we want to use the system interpreter
ENV UV_PYTHON_DOWNLOADS=0

COPY --from=ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /bin/uv
WORKDIR /instant_learn/

# Install git for git dependencies (sam-2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git=1:2.39.5-0+deb12u2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy library and create proper directory structure
COPY --link --from=libs ./src /instant_learn/library/src
COPY --link --from=libs ./pyproject.toml /instant_learn/library/pyproject.toml

# Change to a subdirectory so ../../library resolves correctly
RUN mkdir -p /instant_learn/app/backend
WORKDIR /instant_learn/app/backend

ARG DEVICE=""
ENV DEVICE=${DEVICE}

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra ${DEVICE} --extra mqtt

##################
# Geti UI packages
##################
FROM node:24-alpine3.22@sha256:bb089be859f2741e5ede9d85f47dc7daca754015b50e9642a7a45c5252807d2c AS geti-ui-packages

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:bb089be859f2741e5ede9d85f47dc7daca754015b50e9642a7a45c5252807d2c AS web-ui-base

WORKDIR /home/app/web_ui/

COPY --link ui/package.json ./
COPY --link ui/package-lock.json ./
COPY --from=geti-ui-packages /home/app/web_ui/packages/ /home/app/web_ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link ui/tsconfig.json ./
COPY --link ui/rsbuild.config.ts ./
COPY --link ui/src/ src/

########
# Web UI
########
FROM web-ui-base AS web-ui

ENV ASSET_PREFIX="/html"

# build with --no-cache in case of sub-dependency changes (from geti)
RUN npm run build

##########################################
# Base for CPU, GPU and XPU runtime images
##########################################
FROM python:3.12-slim-bookworm@sha256:ae30458df6c5167adfff1131d78f8c306805a35bb20d8333c20ddfb70715f5e2 AS runtime-base

ENV STATIC_FILES_DIR="/instant_learn/html"
ENV DB_DATA_DIR="/instant_learn/data"
ENV LOGS_DIR="/instant_learn/logs"
ENV PATH="/instant_learn/.venv/bin:$PATH"
ENV PYTHONPATH=/instant_learn/app
ARG UID=10001
ENV UID=${UID}
ARG USER_NAME="non-root"
ENV USER_NAME=${USER_NAME}
ARG TARGET_DATASET_DIR=""
ARG SOURCE_DATASET_DIR=""
ARG DEVICE=""
ENV DEVICE=${DEVICE}

# Create directory and set ownership so mounting a volume works correctly
RUN mkdir -p ${DB_DATA_DIR} && chown ${UID}:${UID} ${DB_DATA_DIR}
RUN mkdir -p ${LOGS_DIR} && chown ${UID}:${UID} ${LOGS_DIR}

# Install OpenGL libraries for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1=1.6.0-1 \
      libglib2.0-0=2.74.6-2+deb12u8 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Remove setuptools, pip, and wheel because uv is used as the package manager.
# This reduces image size and attack surface, and avoids conflicts with uv.
RUN useradd -l -u ${UID} ${USER_NAME} \
    && usermod -G video ${USER_NAME} \
    && mkdir -p /home/${USER_NAME}/.cache/uv \
    && chown -R ${UID}:${UID} /home/${USER_NAME} \
    && pip3 uninstall -y setuptools pip wheel \
    && rm -rf /root/.cache/pip

# Add sample dataset with coffee berries
COPY --link --chown=${UID} ${SOURCE_DATASET_DIR} ${TARGET_DATASET_DIR}

# Copy backend source code
COPY --link --chown=${UID} backend/app /instant_learn/app
COPY --from=backend-base --chown=${UID} /instant_learn/library/src /instant_learn/library/src
COPY --from=backend-base --chown=${UID} /bin/uv /bin/uv
COPY --from=web-ui --chown=${UID} /home/app/web_ui/dist/ ${STATIC_FILES_DIR}

WORKDIR /instant_learn

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://localhost:9100/health')" || exit 1

USER ${USER_NAME}

CMD ["uv", "run", "app/main.py"]

######################
# Server (CPU version)
######################
FROM runtime-base AS cpu

COPY --from=backend-base --chown=${UID} /instant_learn/app/backend/.venv /instant_learn/.venv

######################
# Server (CUDA version)
######################
FROM runtime-base AS cuda

USER root

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.21.3-1+deb12u1 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      nvidia-open=580.105.08-1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# needed to define it here to allow installation of packages
USER ${USER_NAME}

COPY --from=backend-base --chown=${UID} /instant_learn/app/backend/.venv /instant_learn/.venv

######################
# Server (XPU version)
######################
FROM runtime-base AS xpu

USER root

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.21.3-1+deb12u1 gpg=2.2.40-1.1+deb12u2 \
    && wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
       gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg \
    && echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy unified" | \
      tee /etc/apt/sources.list.d/intel-gpu-jammy.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    libze-intel-gpu1  \
    libze1  \
    intel-opencl-icd  \
    clinfo \
    libze-dev \
    intel-ocloc \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get remove -y wget gpg \
    && apt-get clean

# needed to define it here to allow installation of packages
USER ${USER_NAME}

COPY --from=backend-base --chown=${UID} /instant_learn/app/backend/.venv /instant_learn/.venv
