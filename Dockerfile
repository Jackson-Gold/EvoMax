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

RUN python -m py_compile /app/evomax_runner.py && \
    python -m py_compile /app/tests/validate_smoke.py && \
    python /app/tests/test_stage1_selection.py && \
    mkdir -p /data /results

RUN python /app/evomax_runner.py \
        --config /app/tests/fixtures/smoke_config.json && \
    python /app/tests/validate_smoke.py \
        --results-dir /results \
        --config /app/tests/fixtures/smoke_config.json \
        --expected /app/tests/fixtures/smoke_expectations.json && \
    rm -rf /results/*

ENTRYPOINT ["python", "/app/evomax_runner.py"]
CMD ["--help"]

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    PATH=/opt/conda/bin:$PATH

WORKDIR /app

COPY requirements.txt /app/requirements.txt
COPY requirements-gpu.txt /app/requirements-gpu.txt

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        --no-index \
        --find-links https://data.pyg.org/whl/torch-2.6.0+cu124.html \
        torch-scatter==2.1.2 && \
    python -m pip install -r /app/requirements-gpu.txt

RUN python -c "import sys, torch, sklearn, biotite, torch_geometric, torch_scatter; from transformers import EsmForMaskedLM, EsmTokenizer; from esm.inverse_folding.util import CoordBatchConverter; from esm.pretrained import esm_if1_gvp4_t16_142M_UR50; assert sys.version_info[:2] == (3, 11); assert torch.__version__.startswith('2.6.0'); assert torch.version.cuda == '12.4'; print(f'Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA {torch.version.cuda}')"

COPY evomax_runner.py /app/evomax_runner.py
COPY tests /app/tests

RUN python -m py_compile /app/evomax_runner.py && \
    python -m py_compile /app/tests/validate_smoke.py && \
    python /app/tests/test_stage1_selection.py && \
    mkdir -p /data /results /models/huggingface /models/torch

ENTRYPOINT ["python", "/app/evomax_runner.py"]
CMD ["--help"]

FROM smoke AS final
