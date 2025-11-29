FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 1. EVITAR PREGUNTAS (Solución al bloqueo)
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 2. Instalar dependencias, PPA, curl y Python 3.12
# Se añade 'curl' y se elimina 'python3.12-distutils'
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tzdata \
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

# 3. Instalar PIP para Python 3.12 manualmente
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

WORKDIR /app

COPY requirements.txt .

# 4. Instalar dependencias usando pip de Python 3.12
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 5. Ejecutar con Python 3.12
CMD exec python3.12 -m gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sistema_vigilancia.src.cloud_run_main:app
