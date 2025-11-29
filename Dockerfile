FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# 1. Configuración de zona horaria y sistema
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 2. Instalar dependencias del sistema y Python 3.12
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

# 3. CREAR ENTORNO VIRTUAL (VENV)
# Esto aísla tu código de las librerías viejas del sistema (soluciona lo de blinker)
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4. Instalar pip y dependencias DENTRO del venv
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 5. Ejecutar (Al estar el PATH configurado, 'gunicorn' usa el del venv automáticamente)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sistema_vigilancia.src.cloud_run_main:app
