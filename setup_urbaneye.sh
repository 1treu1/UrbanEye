#!/bin/bash

# setup_urbaneye.sh
# Script de instalación para UrbanEye en Raspberry Pi Zero 2 W
# Rama: gcp

set -e  # Salir si ocurre algún error

echo "--- Iniciando instalación de UrbanEye ---"

# 1. Actualizar sistema e instalar dependencias del sistema
echo "[1/6] Actualizando lista de paquetes e instalando dependencias del sistema..."
sudo apt-get update
sudo apt-get install -y git ffmpeg python3-pip python3-venv

# 2. Clonar el repositorio
REPO_URL="https://github.com/1treu1/UrbanEye.git"
REPO_DIR="UrbanEye"

if [ -d "$REPO_DIR" ]; then
    echo "[2/6] El directorio $REPO_DIR ya existe. Eliminando para una instalación limpia..."
    rm -rf "$REPO_DIR"
fi

echo "[2/6] Clonando repositorio desde $REPO_URL..."
git clone "$REPO_URL"

# 3. Cambiar al directorio y a la rama 'gcp'
echo "[3/6] Configurando rama 'gcp'..."
cd "$REPO_DIR"
git checkout gcp
echo "Rama actual:"
git branch --show-current

# 4. Crear entorno virtual
echo "[4/6] Creando entorno virtual (.venv)..."
python3 -m venv .venv
source .venv/bin/activate

# 5. Instalar dependencias de Python
echo "[5/6] Instalando dependencias desde requirements.txt..."
# Actualizar pip primero
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "ADVERTENCIA: No se encontró requirements.txt"
fi

# 6. Verificación final
echo "[6/6] Verificando instalación..."

echo "Verificando FFmpeg:"
ffmpeg -version | head -n 1

echo "Verificando Python en venv:"
which python

echo "--- Instalación Completada Exitosamente ---"
echo "Para ejecutar el script:"
echo "1. cd $REPO_DIR"
echo "2. source .venv/bin/activate"
echo "3. python save_stream2_segments.py [ARGUMENTOS]"
