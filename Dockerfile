# Linux CUDA 12.4 runtime for QLoRA / DPO / inference.
# Default base is runtime-sized. For flash-attn source builds, pass:
#   --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
#   --build-arg INSTALL_FLASH_ATTN=true
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG INSTALL_FLASH_ATTN=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/hf-cache \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_NO_TORCHVISION=1 \
    TRANSFORMERS_IMAGE_TRANSFORMS_DISABLED=1 \
    PATH="/opt/venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.10 \
        python3.10-venv \
        python3-pip \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m venv /opt/venv \
    && pip install --no-cache-dir --upgrade pip

WORKDIR /workspace

RUN pip install --no-cache-dir \
        torch==2.5.1 \
        torchvision==0.20.1 \
        torchaudio==2.5.1 \
        --index-url https://download.pytorch.org/whl/cu124

COPY requirements.docker.txt /tmp/requirements.docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements.docker.txt \
    && rm /tmp/requirements.docker.txt

# Optional; requires the devel CUDA image and a working compiler toolchain.
RUN if [ "${INSTALL_FLASH_ATTN}" = "true" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends \
            python3.10-dev \
            build-essential \
            ninja-build \
        && pip install --no-cache-dir packaging wheel ninja \
        && pip install --no-cache-dir flash-attn --no-build-isolation \
        && rm -rf /var/lib/apt/lists/*; \
    fi

COPY src /workspace/src
COPY tests /workspace/tests
COPY pyproject.toml /workspace/pyproject.toml

RUN mkdir -p /hf-cache /workspace/src/training_output \
    && chmod -R a+rwX /hf-cache /workspace/src/training_output

CMD ["python", "-m", "src.main", "--config-path", "src/config.yaml"]
