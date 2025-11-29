FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Instalar Python 3.12, Build Tools y dependencias
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    # Paquetes necesarios para la compilación (Corrige error como 'contourpy')
    build-essential \
    # Paquetes necesarios para añadir el PPA
    software-properties-common \
    # Añadir el PPA de deadsnakes para Python 3.12
    && add-apt-repository ppa:deadsnakes/ppa -y \
    # Actualizar e instalar Python 3.12
    && apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    python3.12-distutils \
    python3.12-venv \
    # Otras dependencias del sistema
    libgl1-mesa-glx \
    libglib2.0-0 \
    # Limpiar
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# CORRECCIÓN: Llamar a pip usando el ejecutable python3.12
RUN python3.12 -m pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# CORRECCIÓN: Ejecutar gunicorn usando el ejecutable python3.12
CMD exec python3.12 -m gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sistema_vigilancia.src.cloud_run_main:app
