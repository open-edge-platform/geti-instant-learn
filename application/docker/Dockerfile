##################
# Geti UI packages
##################
FROM node:24-alpine3.22@sha256:d28696cabe6a72c5addbb608b344818e5a158d849174abd4b1ae85ab48536280 AS geti-ui-packages

WORKDIR /home/app/web_ui/
RUN apk add --no-cache git=2.49.1-r0
COPY --link ./ui/package.json ./
RUN npm run clone-geti-ui-packages

#############
# Web UI deps
#############
FROM node:24-alpine3.22@sha256:d28696cabe6a72c5addbb608b344818e5a158d849174abd4b1ae85ab48536280 AS web-ui-base

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
FROM python:3.13-slim@sha256:3de9a8d7aedbb7984dc18f2dff178a7850f16c1ae7c34ba9d7ecc23d0755e35f AS runtime-base
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
# Disable Python downloads, because we want to use the system interpreter
ENV UV_PYTHON_DOWNLOADS=0

ENV STATIC_FILES_DIR="/instant_learn/html"
ENV DB_DATA_DIR="/instant_learn/data"
ENV LOGS_DIR="/instant_learn/logs"
ENV PATH="/instant_learn/app/backend/.venv/bin:$PATH"
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
    git=1:2.47.3-0+deb13u1 \
    libgl1=1.7.0-1+b2 \
    libglib2.0-0=2.84.4-3~deb13u2 \
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
RUN mkdir -p /instant_learn/app/backend
WORKDIR /instant_learn/app/backend

# Add sample dataset with coffee berries
COPY --link --chown=${UID} ${SOURCE_DATASET_DIR} ${TARGET_DATASET_DIR}

# Copy library and create proper directory structure
COPY --link --from=libs --chown=${UID} ./src /instant_learn/library/src
COPY --link --from=libs --chown=${UID} ./pyproject.toml /instant_learn/library/pyproject.toml

# Copy backend source code
COPY --link --chown=${UID} backend/app /instant_learn/app
COPY --chown=${UID} --from=ghcr.io/astral-sh/uv:0.10.0@sha256:78a7ff97cd27b7124a5f3c2aefe146170793c56a1e03321dd31a289f6d82a04f /uv /bin/uv
COPY --from=web-ui --chown=${UID} /home/app/web_ui/dist/ ${STATIC_FILES_DIR}

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://localhost:9100/health')" || exit 1

USER ${USER_NAME}

CMD ["uv", "run", "app/main.py"]

######################
# Server (CPU version)
######################
FROM runtime-base AS cpu

USER root

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra cpu --extra mqtt

RUN apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /instant_learn/library/pyproject.toml

USER ${USER_NAME}

WORKDIR "/instant_learn"

#######################
# Intermediary GPU Base
#######################
FROM runtime-base AS gpu-base

USER root

RUN apt-get update  \
    && apt-get install -y --no-install-recommends wget=1.25.0-2 \
    && wget -q https://developer.download.nvidia.com/compute/cuda/repos/debian13/x86_64/cuda-keyring_1.1-1_all.deb \
    && dpkg -i cuda-keyring_1.1-1_all.deb \
    && rm -rf cuda-keyring_1.1-1_all.deb \
    && apt-get remove -y wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    nvidia-open=590.48.01-1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra cuda --extra mqtt

RUN apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /instant_learn/library/pyproject.toml

FROM gpu-base AS cleanup-gpu

# CUDA EULA Compliance Cleanup
RUN echo "Removing non-redistributable CUDA components per NVIDIA EULA..." && \
    # Define search paths including venv
    SEARCH_PATHS="/usr /opt ${PYTHONPATH}" && \
    # Add any detected venvs
    for venv in $(find ${PYTHONPATH} -name "site-packages" -type d 2>/dev/null | sed 's|/lib/python[0-9.]*/site-packages||' | sort -u); do \
    if [ -d "$venv" ]; then \
    SEARCH_PATHS="$SEARCH_PATHS $venv"; \
    fi; \
    done && \
    echo "Cleaning CUDA components in: $SEARCH_PATHS" && \
    # Remove development tools (non-redistributable)
    find $SEARCH_PATHS -type f \( \
    -name "nvcc" -o \
    -name "cuda-gdb" -o \
    -name "nvprof" -o \
    -name "nvdisasm" -o \
    -name "cuobjdump" -o \
    -name "cuda-memcheck" -o \
    -name "nsight*" -o \
    -name "compute-sanitizer" -o \
    -name "ncu" -o \
    -name "nv-nsight-cu*" \
    \) -delete 2>/dev/null || true && \
    # Remove headers and include directories (non-redistributable)
    find $SEARCH_PATHS -type d \( \
    -name "include" -o \
    -name "headers" \
    \) -path "*cuda*" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove NVIDIA package includes specifically
    find $SEARCH_PATHS -type d -path "*/nvidia/*/include" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove static libraries (typically non-redistributable)
    find $SEARCH_PATHS -name "*.a" -path "*cuda*" -delete 2>/dev/null || true && \
    find $SEARCH_PATHS -name "*.a" -path "*nvidia*" -delete 2>/dev/null || true && \
    # Remove documentation and samples (non-redistributable)
    find $SEARCH_PATHS -type d \( \
    -name "doc" -o \
    -name "docs" -o \
    -name "documentation" -o \
    -name "samples" -o \
    -name "sample" \
    \) -path "*cuda*" -exec rm -rf {} + 2>/dev/null || true && \
    # Remove specific non-redistributable CUPTI components
    find $SEARCH_PATHS -type f \( \
    -name "libcheckpoint.so*" -o \
    -name "libnvperf_host.so*" -o \
    -name "libnvperf_target.so*" -o \
    -name "libpcsamplingutil.so*" \
    \) -path "*cupti*" -delete 2>/dev/null || true && \
    # Remove cuSolver MG libraries (enterprise feature)
    find $SEARCH_PATHS -name "libcusolverMg.so*" -delete 2>/dev/null || true && \
    # Remove Triton NVIDIA backends (development components)
    find $SEARCH_PATHS -type d -path "*/triton/backends/nvidia" -exec rm -rf {} + 2>/dev/null || true && \
    # System package cleanup
    apt-get remove -y \
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
    2>/dev/null || true && \
    apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*

######################
# Server (GPU version)
######################
FROM runtime-base AS cuda

COPY --link --from=cleanup-gpu --chown=${UID} /usr/local /usr/local
COPY --link --from=cleanup-gpu --chown=${UID} ${PYTHONPATH} ${PYTHONPATH}

USER ${USER_NAME}

WORKDIR "/instant_learn"

######################
# Server (XPU version)
######################
FROM runtime-base AS xpu

USER root

# based on https://dgpu-docs.intel.com/driver/client/overview.html#ubuntu-latest
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN echo "deb [arch=amd64 trusted=yes] https://ppa.launchpadcontent.net/kobuk-team/intel-graphics/ubuntu noble main" \
    | tee /etc/apt/sources.list.d/kobuk-intel.list

# hadolint ignore=DL3008
RUN apt-get update  \
    && apt-get install -y --no-install-recommends \
    libze-intel-gpu1=26.01.36711.4-1~24.04~ppa1 \
    libze1=1.26.2-1~24.04~ppa1 \
    libze-dev=1.26.2-1~24.04~ppa1 \
    intel-metrics-discovery=1.14.183-1~24.04~ppa1 \
    intel-opencl-icd=26.01.36711.4-1~24.04~ppa1 \
    clinfo=3.0.25.02.14-1 \
    intel-gsc=0.9.5-1~24.04~ppa2 \
    intel-ocloc=26.01.36711.4-1~24.04~ppa1 \
    intel-media-va-driver-non-free=25.4.6-1~24.04~ppa1 \
    libmfx-gen1=25.4.0-0ubuntu1~24.04~ppa1 \
    libvpl2=1:2.16.0-1~24.04~ppa1 \
    libvpl-tools=1.5.0-1~24.04~ppa1 \
    libva-glx2=2.22.0-3 \
    va-driver-all=2.22.0-3 \
    vainfo=2.22.0+ds1-2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=backend/uv.lock,target=uv.lock \
    --mount=type=bind,source=backend/pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-editable --extra xpu --extra mqtt

RUN apt-get remove -y git \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /instant_learn/library/pyproject.toml

# needed to define it here to allow installation of packages
USER ${USER_NAME}

WORKDIR "/instant_learn"
