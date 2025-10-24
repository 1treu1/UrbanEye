#!/usr/bin/env python3
"""
Script de prueba rápida para el Sistema de Vigilancia
====================================================

Este script verifica que todos los componentes estén funcionando correctamente.
"""

import sys
import os
import time
import cv2
import numpy as np
from pathlib import Path

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_imports():
    """Probar imports básicos"""
    print("🔄 Probando imports básicos...")
    
    try:
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
    except ImportError as e:
        print(f"❌ OpenCV: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
    except ImportError as e:
        print(f"❌ NumPy: {e}")
        return False
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("   GPU: No disponible")
    except ImportError as e:
        print(f"❌ PyTorch: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO")
    except ImportError as e:
        print(f"❌ Ultralytics: {e}")
        return False
    
    return True


def test_camera():
    """Probar cámara"""
    print("\n🔄 Probando cámara...")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return False
    
    ret, frame = cap.read()
    if not ret:
        print("❌ No se pudo leer frame de la cámara")
        cap.release()
        return False
    
    print(f"✅ Cámara funcionando - Resolución: {frame.shape[1]}x{frame.shape[0]}")
    cap.release()
    return True


def test_yolo():
    """Probar YOLO"""
    print("\n🔄 Probando YOLO...")
    
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        print("✅ YOLO cargado correctamente")
        
        # Crear frame de prueba
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        results = model(test_frame, verbose=False)
        print("✅ YOLO inferencia funcionando")
        return True
    except Exception as e:
        print(f"❌ Error en YOLO: {e}")
        return False


def test_midas():
    """Probar MiDaS"""
    print("\n🔄 Probando MiDaS...")
    
    try:
        import torch
        model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
        print("✅ MiDaS cargado correctamente")
        
        # Probar inferencia
        test_tensor = torch.randn(1, 3, 480, 640)
        if torch.cuda.is_available():
            test_tensor = test_tensor.cuda()
            model = model.cuda()
        
        with torch.no_grad():
            depth = model(test_tensor)
            print("✅ MiDaS inferencia funcionando")
        return True
    except Exception as e:
        print(f"❌ Error en MiDaS: {e}")
        return False


def test_advanced_system():
    """Probar sistema avanzado"""
    print("\n🔄 Probando sistema avanzado...")
    
    try:
        from advanced_surveillance import AdvancedSurveillanceSystem
        
        # Crear sistema
        system = AdvancedSurveillanceSystem(use_gpu=True, use_midas=False)
        print("✅ Sistema avanzado creado")
        
        # Probar con frame de prueba
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        processed_frame = system.process_frame(test_frame)
        print("✅ Procesamiento de frame funcionando")
        return True
    except Exception as e:
        print(f"❌ Error en sistema avanzado: {e}")
        return False


def test_basic_system():
    """Probar sistema básico"""
    print("\n🔄 Probando sistema básico...")
    
    try:
        from main_simple_gender import AdvancedPersonAnalyzer
        
        # Crear analizador
        analyzer = AdvancedPersonAnalyzer()
        print("✅ Analizador básico creado")
        
        # Probar análisis
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 200, 300)
        gender, category, age = analyzer.analyze_person(test_frame, test_bbox)
        print(f"✅ Análisis funcionando - Resultado: {gender}, {category}, {age}")
        return True
    except Exception as e:
        print(f"❌ Error en sistema básico: {e}")
        return False


def run_performance_test():
    """Ejecutar prueba de rendimiento"""
    print("\n🔄 Ejecutando prueba de rendimiento...")
    
    try:
        from advanced_surveillance import AdvancedSurveillanceSystem
        
        system = AdvancedSurveillanceSystem(use_gpu=True, use_midas=False)
        
        # Crear frame de prueba
        test_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Medir tiempo de procesamiento
        start_time = time.time()
        for i in range(10):
            processed_frame = system.process_frame(test_frame)
        end_time = time.time()
        
        fps = 10 / (end_time - start_time)
        print(f"✅ Rendimiento: {fps:.1f} FPS")
        
        if fps > 20:
            print("✅ Rendimiento excelente")
        elif fps > 10:
            print("✅ Rendimiento bueno")
        else:
            print("⚠️  Rendimiento bajo - considerar optimizaciones")
        
        return True
    except Exception as e:
        print(f"❌ Error en prueba de rendimiento: {e}")
        return False


def main():
    """Función principal de pruebas"""
    print("Pruebas del Sistema de Vigilancia")
    print("=" * 40)
    
    tests = [
        ("Imports básicos", test_imports),
        ("Cámara", test_camera),
        ("YOLO", test_yolo),
        ("MiDaS", test_midas),
        ("Sistema básico", test_basic_system),
        ("Sistema avanzado", test_advanced_system),
        ("Rendimiento", run_performance_test)
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Error inesperado en {name}: {e}")
            results.append((name, False))
    
    # Resumen
    print(f"\n{'='*50}")
    print("📊 RESUMEN DE PRUEBAS")
    print(f"{'='*50}")
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {name}")
        if result:
            passed += 1
    
    print(f"\nResultado: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print("¡Todas las pruebas pasaron! El sistema está listo.")
        print("\nPara ejecutar el sistema:")
        print("   python -m src.main_simple_gender --source 0 --gpu")
        print("   python -m src.advanced_surveillance --source 0 --gpu")
    else:
        print("Algunas pruebas fallaron. Revisar errores arriba.")
        print("Sugerencias:")
        print("   - Verificar instalación de dependencias")
        print("   - Ejecutar: python install.py")
        print("   - Verificar drivers de GPU")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
