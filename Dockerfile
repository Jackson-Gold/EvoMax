FROM python:3.11-slim AS smoke

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt

COPY evomax_runner.py /app/evomax_runner.py
COPY tests /app/tests

RUN python -m py_compile /app/evomax_runner.py /app/tests/validate_smoke.py && \
    mkdir -p /data /results

ENTRYPOINT ["python", "/app/evomax_runner.py"]
CMD ["--help"]

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements-gpu.txt /app/requirements-gpu.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install -r /app/requirements.txt -r /app/requirements-gpu.txt
RUN python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
RUN python -m pip install \
    pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.6.0+cu124.html && \
    python -m pip install torch-geometric==2.7.0

COPY evomax_runner.py /app/evomax_runner.py
COPY tests /app/tests

RUN python -m py_compile /app/evomax_runner.py /app/tests/validate_smoke.py && \
    mkdir -p /data /results /models/huggingface

ENTRYPOINT ["python", "/app/evomax_runner.py"]
CMD ["--help"]

FROM smoke AS final
