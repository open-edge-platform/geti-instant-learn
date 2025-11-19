##################
# Geti UI packages
##################
FROM node:24-alpine3.22@sha256:2867d550cf9d8bb50059a0fff528741f11a84d985c732e60e19e8e75c7239c43 AS geti-ui-packages

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:2867d550cf9d8bb50059a0fff528741f11a84d985c732e60e19e8e75c7239c43 AS web-ui-base

WORKDIR /home/app/web_ui/

COPY --link ui/package.json ./
COPY --link ui/package-lock.json ./
COPY --from=geti-ui-packages /home/app/web_ui/packages/ /home/app/web_ui/packages/
RUN npm ci --audit=false --ignore-scripts

COPY --link ui/tsconfig.json ./
COPY --link ui/rsbuild.config.ts ./
COPY --link ui/src/ src/

ARG PUBLIC_API_URL=""
ENV PUBLIC_API_URL=${PUBLIC_API_URL}

########
# Web UI
########
FROM web-ui-base AS web-ui

ARG ASSET_PREFIX=""
ENV ASSET_PREFIX=${ASSET_PREFIX}

# build with --no-cache in case of sub-dependency changes (from geti)
RUN npm run build

##########################################
# Base for CPU, GPU and XPU runtime images
##########################################
FROM python:3.12-slim-bookworm@sha256:874450e1da63e36bf5c1ca8b0a13cb9eb0bbb1c4015151e376978f9aa6ae3bc2 AS runtime-base
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Disable Python downloads, because we want to use the system interpreter
ENV UV_PYTHON_DOWNLOADS=0

ARG STATIC_FILES_DIR=""
ENV STATIC_FILES_DIR=${STATIC_FILES_DIR}
ARG DB_DATA_DIR=""
ENV DB_DATA_DIR=${DB_DATA_DIR}
ARG MOCKED_FILE=""
ENV MOCKED_FILE=${MOCKED_FILE}
ENV PYTHONPATH=${WORKDIR_PATH}/app
ENV PATH="${PYTHONPATH}/backend/.venv/bin:$PATH"
ENV LIBRARY_PATH="${WORKDIR_PATH}/library"
ARG UID=10001
ENV UID=${UID}
ARG USER_NAME="non-root"
ENV USER_NAME=${USER_NAME}
ARG PUBLIC_API_URL=""
ENV PUBLIC_API_URL=${PUBLIC_API_URL}

# Create directory and set ownership before VOLUME instruction
RUN mkdir -p ${DB_DATA_DIR} && chown ${UID}:${UID} ${DB_DATA_DIR}
VOLUME ${DB_DATA_DIR}

# Install OpenGL libraries for opencv-python
RUN apt-get update && apt-get install -y --no-install-recommends \
      git=1:2.39.5-0+deb12u2 \
      libgl1=1.6.0-1 \
      libglib2.0-0=2.74.6-2+deb12u7 \
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

# Change to a subdirectory so ../../library resolves correctly
RUN mkdir -p ${PYTHONPATH}/backend
WORKDIR ${PYTHONPATH}/backend

# Copy library and create proper directory structure
COPY --link --from=libs --chown=${UID} ./src ${LIBRARY_PATH}/src
COPY --link --from=libs --chown=${UID} ./pyproject.toml ${LIBRARY_PATH}/pyproject.toml
# Copy backend source code
COPY --link --chown=${UID} backend/app ${PYTHONPATH}
COPY --chown=${UID} --from=ghcr.io/astral-sh/uv:0.9.7@sha256:ba4857bf2a068e9bc0e64eed8563b065908a4cd6bfb66b531a9c424c8e25e142 /uv /bin/uv
COPY --from=web-ui --chown=${UID} /home/app/web_ui/dist/ ${STATIC_FILES_DIR}
# temporary file for testing
COPY --from=web-ui --chown=${UID} /home/app/web_ui/src/assets/test.webp ${MOCKED_FILE}

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('${PUBLIC_API_URL}/health')" || exit 1

USER ${USER_NAME}

CMD ["uv", "run", "app/main.py"]

######################
# Server (CPU version)
######################
FROM runtime-base AS geti-prompt-cpu

USER root

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra cpu

RUN apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* ${LIBRARY_PATH}/pyproject.toml

USER ${USER_NAME}

WORKDIR ${WORKDIR_PATH}

#######################
# Intermediary GPU Base
#######################
FROM runtime-base AS gpu-base

USER root

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.21.3-1+deb12u1 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      nvidia-open=580.105.08-1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra cu126

# Remove Specific Non-Distributable Components
RUN apt-get update && apt-get remove -y \
    cuda-compiler-* \
    cuda-cuobjdump-* \
    cuda-gdb-* \
    cuda-memcheck-* \
    cuda-nvcc-* \
    cuda-nvdisasm-* \
    cuda-nvprof-* \
    cuda-sanitizer-* \
    cuda-documentation-* \
    cuda-samples-* \
    nsight-* \
    2>/dev/null || true \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Remove all non-redistributable CUDA components per EULA Section 2.7
RUN set -e; \
    echo "Removing non-redistributable CUDA components per NVIDIA EULA..."; \
    \
    # Remove development tools (non-redistributable)
    find /usr /opt ${WORKDIR_PATH} -type f \( \
        -name "nvcc" -o \
        -name "cuda-gdb" -o \
        -name "nvprof" -o \
        -name "nvdisasm" -o \
        -name "cuobjdump" -o \
        -name "cuda-memcheck" -o \
        -name "nsight*" -o \
        -name "compute-sanitizer" \
    \) -delete 2>/dev/null || true; \
    \
    # Remove static libraries (typically non-redistributable)
    find /usr /opt ${WORKDIR_PATH} -name "*.a" -path "*cuda*" -delete 2>/dev/null || true; \
    \
    # Remove headers and include directories (non-redistributable)
    find /usr /opt ${WORKDIR_PATH} -type d \( \
        -name "include" -o \
        -name "headers" \
    \) -path "*cuda*" -exec rm -rf {} + 2>/dev/null || true; \
    \
    # Remove documentation and samples (non-redistributable)
    find /usr /opt ${WORKDIR_PATH} -type d \( \
        -name "doc" -o \
        -name "docs" -o \
        -name "documentation" -o \
        -name "samples" -o \
        -name "sample" \
    \) -path "*cuda*" -exec rm -rf {} + 2>/dev/null || true; \
    \
    # Remove compiler-related directories
    find /usr /opt ${WORKDIR_PATH} -type d -name "*nvcc*" -exec rm -rf {} + 2>/dev/null || true; \
    \
    # Clean up package manager traces
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* ${LIBRARY_PATH}/pyproject.toml

######################
# Server (GPU version)
######################
FROM gpu-base AS geti-prompt-gpu

USER ${USER_NAME}

WORKDIR ${WORKDIR_PATH}

######################
# Server (XPU version)
######################
FROM runtime-base AS geti-prompt-xpu

USER root

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.21.3-1+deb12u1 gpg=2.2.40-1.1+deb12u1 \
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
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get remove -y wget gpg \
    && apt-get clean

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra xpu

RUN apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* ${LIBRARY_PATH}/pyproject.toml

# needed to define it here to allow installation of packages
USER ${USER_NAME}

WORKDIR ${WORKDIR_PATH}
