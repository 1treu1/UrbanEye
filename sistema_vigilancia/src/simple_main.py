#!/usr/bin/env python3
"""
Versión simplificada del sistema de vigilancia sin DeepFace
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sistema de Vigilancia Simplificado")
    parser.add_argument("--source", type=str, default="0", help="Fuente de video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="Modelo YOLO")
    args = parser.parse_args()
    
    # Configurar fuente
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Cargar modelo YOLO
    print("Cargando modelo YOLO...")
    model = YOLO(args.model)
    
    # Configurar cámara
    print("Iniciando cámara...")
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara")
        return
    
    # Configurar resolución para mejor rendimiento
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("Sistema iniciado. Presiona 'q' para salir.")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar personas
            results = model(frame, verbose=False)
            
            # Dibujar detecciones
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Solo personas (clase 0 en COCO)
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            if conf > 0.5:  # Umbral de confianza
                                # Dibujar rectángulo
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                # Dibujar etiqueta
                                label = f"Persona: {conf:.2f}"
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow('Sistema de Vigilancia', frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nSistema detenido por el usuario")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Sistema cerrado correctamente")

if __name__ == "__main__":
    main()


