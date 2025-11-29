FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 1. Instalar dependencias del sistema, PPA y Python 3.12
# Nota: Se agregó 'curl' para instalar pip manualmente después
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. Instalar PIP para Python 3.12 manualmente
# (Esto es necesario porque el paquete python3-pip de Ubuntu suele ser para Python 3.10)
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

COPY requirements.txt .

# 3. Instalar dependencias usando pip de Python 3.12
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 4. Ejecutar con Python 3.12
CMD exec python3.12 -m gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sistema_vigilancia.src.cloud_run_main:app
