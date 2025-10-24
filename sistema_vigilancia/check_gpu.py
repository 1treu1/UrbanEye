#!/usr/bin/env python3
"""
Script para verificar el estado de GPU y dependencias
"""

import torch
import cv2
import sys

def check_gpu_status():
    print("=== VERIFICACIÓN DE GPU ===")
    print(f"PyTorch versión: {torch.__version__}")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA versión: {torch.version.cuda}")
        print(f"Número de GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memoria total: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    else:
        print("❌ CUDA no está disponible")
    
    print("\n=== VERIFICACIÓN DE DEPENDENCIAS ===")
    
    # Verificar OpenCV
    print(f"OpenCV versión: {cv2.__version__}")
    
    # Verificar si TensorFlow está disponible para DeepFace
    try:
        import tensorflow as tf
        print(f"TensorFlow versión: {tf.__version__}")
        print(f"TensorFlow GPU disponible: {len(tf.config.list_physical_devices('GPU')) > 0}")
    except ImportError:
        print("❌ TensorFlow no está instalado (necesario para DeepFace con GPU)")
    
    # Verificar ultralytics
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics (YOLO) disponible")
    except ImportError:
        print("❌ Ultralytics no está disponible")
    
    # Verificar deepface
    try:
        from deepface import DeepFace
        print("✅ DeepFace disponible")
    except ImportError:
        print("❌ DeepFace no está disponible")
    
    print("\n=== RECOMENDACIONES ===")
    if not torch.cuda.is_available():
        print("• Instala PyTorch con soporte CUDA: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    try:
        import tensorflow as tf
        if not tf.config.list_physical_devices('GPU'):
            print("• Configura TensorFlow para usar GPU")
    except ImportError:
        print("• Instala TensorFlow: pip install tensorflow[and-cuda]")

if __name__ == "__main__":
    check_gpu_status()


