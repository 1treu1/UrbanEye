#!/usr/bin/env python3
"""
Script de instalación automática para el Sistema de Vigilancia Avanzado
======================================================================

Este script automatiza la instalación de todas las dependencias necesarias
y configura el entorno para el sistema de vigilancia.
"""

import subprocess
import sys
import platform
import os
from pathlib import Path


def run_command(command, description=""):
    """Ejecutar comando y manejar errores"""
    print(f"🔄 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, 
                               capture_output=True, text=True)
        print(f"✅ {description} - Completado")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}: {e}")
        print(f"   Comando: {command}")
        print(f"   Error: {e.stderr}")
        return False


def check_python_version():
    """Verificar versión de Python"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Se requiere Python 3.8 o superior")
        print(f"   Versión actual: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
    return True


def check_gpu_support():
    """Verificar soporte de GPU"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU detectada: {gpu_name} ({gpu_memory:.1f}GB)")
            return True
        else:
            print("⚠️  GPU no disponible - usando CPU")
            return False
    except ImportError:
        print("⚠️  PyTorch no instalado - verificando después de instalación")
        return None


def install_basic_dependencies():
    """Instalar dependencias básicas"""
    basic_deps = [
        "opencv-python>=4.8.0",
        "numpy>=1.21.0",
        "Pillow>=9.0.0",
        "scipy>=1.9.0",
        "tqdm>=4.64.0",
        "matplotlib>=3.5.0"
    ]
    
    for dep in basic_deps:
        if not run_command(f"pip install {dep}", f"Instalando {dep.split('>=')[0]}"):
            return False
    return True


def install_pytorch(gpu_support=True):
    """Instalar PyTorch con soporte GPU"""
    if gpu_support:
        # Detectar versión de CUDA
        try:
            result = subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                # CUDA 11.8 por defecto
                cuda_version = "cu118"
                print(f"🔄 Instalando PyTorch con CUDA {cuda_version}")
                command = f"pip install torch torchvision --index-url https://download.pytorch.org/whl/{cuda_version}"
            else:
                print("⚠️  nvidia-smi no encontrado, usando CUDA 11.8 por defecto")
                command = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        except:
            command = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    else:
        print("🔄 Instalando PyTorch para CPU")
        command = "pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
    
    return run_command(command, "Instalando PyTorch")


def install_ultralytics():
    """Instalar Ultralytics YOLO"""
    return run_command("pip install ultralytics>=8.0.0", "Instalando Ultralytics")


def test_installation():
    """Probar instalación"""
    print("\n🧪 Probando instalación...")
    
    test_imports = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("torch", "PyTorch"),
        ("ultralytics", "Ultralytics"),
        ("PIL", "Pillow")
    ]
    
    all_good = True
    for module, name in test_imports:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - Error: {e}")
            all_good = False
    
    # Probar GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA - {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA no disponible")
    except:
        print("❌ Error verificando CUDA")
        all_good = False
    
    return all_good


def create_venv():
    """Crear entorno virtual si no existe"""
    venv_path = Path(".venv")
    if venv_path.exists():
        print("✅ Entorno virtual ya existe")
        return True
    
    print("🔄 Creando entorno virtual...")
    if run_command("python -m venv .venv", "Creando entorno virtual"):
        print("✅ Entorno virtual creado")
        print("📝 Para activar el entorno virtual:")
        if platform.system() == "Windows":
            print("   .venv\\Scripts\\activate")
        else:
            print("   source .venv/bin/activate")
        return True
    return False


def main():
    """Función principal de instalación"""
    print("🚀 Instalador del Sistema de Vigilancia Avanzado")
    print("=" * 50)
    
    # Verificar Python
    if not check_python_version():
        return False
    
    # Crear entorno virtual
    create_venv()
    
    # Verificar GPU
    gpu_available = check_gpu_support()
    
    print("\n📦 Instalando dependencias...")
    
    # Instalar dependencias básicas
    if not install_basic_dependencies():
        print("❌ Error instalando dependencias básicas")
        return False
    
    # Instalar PyTorch
    if not install_pytorch(gpu_available):
        print("❌ Error instalando PyTorch")
        return False
    
    # Instalar Ultralytics
    if not install_ultralytics():
        print("❌ Error instalando Ultralytics")
        return False
    
    # Probar instalación
    if not test_installation():
        print("❌ Error en pruebas de instalación")
        return False
    
    print("\n🎉 ¡Instalación completada exitosamente!")
    print("\n📋 Próximos pasos:")
    print("1. Activar entorno virtual:")
    if platform.system() == "Windows":
        print("   .venv\\Scripts\\activate")
    else:
        print("   source .venv/bin/activate")
    print("\n2. Ejecutar sistema básico:")
    print("   python -m src.main_simple_gender --source 0 --gpu")
    print("\n3. Ejecutar sistema avanzado:")
    print("   python -m src.advanced_surveillance --source 0 --gpu")
    print("\n4. Ver README.md para más opciones")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

