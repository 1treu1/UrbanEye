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
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 4. PREPARAR EL DIRECTORIO DE TRABAJO
WORKDIR /app

# --- AQUÍ ESTÁ LA CORRECCIÓN ---
# Primero copiamos SOLO el requirements.txt
COPY requirements.txt .

# Ahora sí podemos instalar, porque el archivo ya existe dentro de la imagen
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install tf-keras

# Finalmente copiamos el resto del código
COPY . .

ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# 5. Ejecutar
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 sistema_vigilancia.src.cloud_run_main:app
