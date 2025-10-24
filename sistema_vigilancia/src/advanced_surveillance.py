"""
Sistema de Vigilancia Avanzado con Auto-calibración y MiDaS
==========================================================

Características:
- Auto-calibración basada en alturas observadas
- Suavizado adaptativo
- MiDaS para estimación de profundidad (GPU)
- Clasificador de sexo mejorado con fusión altura + probabilidad
- Estructura modular
"""

from __future__ import annotations

import argparse
import cv2
import numpy as np
import math
import torch
from typing import Optional, List, Tuple, Dict
from collections import deque
from dataclasses import dataclass
from ultralytics import YOLO

# Importar módulos del sistema
from .config import AppConfig, build_default_polygon
from .application.event_logger import EventLogger


@dataclass
class PersonMeasurement:
    """Medición de una persona con información completa"""
    bbox: Tuple[int, int, int, int]
    height_pixels: float
    real_height: float
    distance: float
    confidence: float
    timestamp: float
    position: Tuple[int, int]  # (center_x, center_y)


@dataclass
class CalibrationData:
    """Datos de calibración del sistema"""
    camera_height: float
    focal_length: float
    sensor_height: float
    angle_degrees: float
    confidence: float
    sample_count: int


class AutoCalibrator:
    """Auto-calibrador basado en alturas observadas"""
    
    def __init__(self, min_samples: int = 10, max_samples: int = 100):
        self.min_samples = min_samples
        self.max_samples = max_samples
        self.measurements: deque = deque(maxlen=max_samples)
        self.calibration_data: Optional[CalibrationData] = None
        self.height_statistics = {
            'adult_heights': [],
            'child_heights': [],
            'all_heights': []
        }
    
    def add_measurement(self, measurement: PersonMeasurement):
        """Agregar nueva medición para calibración"""
        self.measurements.append(measurement)
        self.height_statistics['all_heights'].append(measurement.real_height)
        
        # Clasificar por altura
        if measurement.real_height >= 1.5:
            self.height_statistics['adult_heights'].append(measurement.real_height)
        else:
            self.height_statistics['child_heights'].append(measurement.real_height)
    
    def calibrate(self) -> Optional[CalibrationData]:
        """Calibrar sistema basado en mediciones"""
        if len(self.measurements) < self.min_samples:
            return None
        
        # Análisis estadístico de alturas
        adult_heights = np.array(self.height_statistics['adult_heights'])
        child_heights = np.array(self.height_statistics['child_heights'])
        
        if len(adult_heights) < 5:
            return None
        
        # Calcular altura promedio de adultos (referencia antropométrica)
        adult_height_mean = np.mean(adult_heights)
        adult_height_std = np.std(adult_heights)
        
        # Filtrar outliers (alturas muy anómalas)
        valid_adult_heights = adult_heights[
            (adult_heights >= adult_height_mean - 2*adult_height_std) &
            (adult_heights <= adult_height_mean + 2*adult_height_std)
        ]
        
        if len(valid_adult_heights) < 3:
            return None
        
        # Calcular parámetros de cámara basados en estadísticas
        target_adult_height = 1.70  # Altura promedio mundial
        height_ratio = target_adult_height / np.mean(valid_adult_heights)
        
        # Estimar altura de cámara
        estimated_camera_height = 7.0 * height_ratio
        
        # Estimar ángulo de cámara basado en variación de alturas
        height_variance = np.var(valid_adult_heights)
        if height_variance < 0.1:  # Poca variación = cámara overhead
            angle_degrees = 90
        elif height_variance < 0.3:  # Variación media = cámara alta
            angle_degrees = 75
        else:  # Mucha variación = cámara normal
            angle_degrees = 60
        
        # Calcular confianza basada en consistencia
        consistency = 1.0 - min(height_variance, 1.0)
        sample_confidence = min(len(valid_adult_heights) / 20, 1.0)
        confidence = (consistency + sample_confidence) / 2
        
        self.calibration_data = CalibrationData(
            camera_height=estimated_camera_height,
            focal_length=50.0,  # Valor por defecto
            sensor_height=24.0,  # Valor por defecto
            angle_degrees=angle_degrees,
            confidence=confidence,
            sample_count=len(valid_adult_heights)
        )
        
        return self.calibration_data
    
    def get_calibration_status(self) -> str:
        """Obtener estado de calibración"""
        if not self.calibration_data:
            return f"Calibrando... ({len(self.measurements)}/{self.min_samples} muestras)"
        
        return (f"Calibrado: {self.calibration_data.camera_height:.1f}m, "
                f"ángulo {self.calibration_data.angle_degrees}°, "
                f"confianza {self.calibration_data.confidence:.2f}")


class AdaptiveSmoother:
    """Suavizado adaptativo para mediciones"""
    
    def __init__(self, window_size: int = 5, alpha: float = 0.3):
        self.window_size = window_size
        self.alpha = alpha
        self.height_history: deque = deque(maxlen=window_size)
        self.distance_history: deque = deque(maxlen=window_size)
        self.smoothed_height = 0.0
        self.smoothed_distance = 0.0
    
    def update(self, height: float, distance: float) -> Tuple[float, float]:
        """Actualizar con suavizado adaptativo"""
        self.height_history.append(height)
        self.distance_history.append(distance)
        
        if len(self.height_history) < 2:
            return height, distance
        
        # Suavizado exponencial adaptativo
        height_trend = np.mean(np.diff(list(self.height_history)))
        distance_trend = np.mean(np.diff(list(self.distance_history)))
        
        # Ajustar alpha basado en variabilidad
        height_variance = np.var(list(self.height_history))
        distance_variance = np.var(list(self.distance_history))
        
        adaptive_alpha = self.alpha * (1 + min(height_variance + distance_variance, 1.0))
        adaptive_alpha = min(adaptive_alpha, 0.8)  # Límite superior
        
        # Aplicar suavizado
        self.smoothed_height = (adaptive_alpha * height + 
                               (1 - adaptive_alpha) * self.smoothed_height)
        self.smoothed_distance = (adaptive_alpha * distance + 
                                 (1 - adaptive_alpha) * self.smoothed_distance)
        
        return self.smoothed_height, self.smoothed_distance


class MiDaSDepthEstimator:
    """Estimador de profundidad usando MiDaS (GPU)"""
    
    def __init__(self, model_name: str = "DPT_Large", use_gpu: bool = True):
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._load_model()
    
    def _load_model(self):
        """Cargar modelo MiDaS"""
        try:
            print(f"🔄 Cargando modelo MiDaS {self.model_name}...")
            self.model = torch.hub.load("intel-isl/MiDaS", self.model_name)
            self.model.to(self.device).eval()
            print(f"✅ Modelo MiDaS cargado en {self.device}")
        except Exception as e:
            print(f"❌ Error cargando MiDaS: {e}")
            self.model = None
    
    def estimate_depth_map(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Estimar mapa de profundidad"""
        if self.model is None:
            return None
        
        try:
            # Preprocesar frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).float().to(self.device)
            frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
            
            # Inferencia
            with torch.no_grad():
                depth_map = self.model(frame_tensor)
                depth_map = depth_map.squeeze().cpu().numpy()
            
            return depth_map
        except Exception as e:
            print(f"❌ Error en MiDaS: {e}")
            return None
    
    def get_depth_at_point(self, depth_map: np.ndarray, x: int, y: int) -> float:
        """Obtener profundidad en un punto específico"""
        if depth_map is None:
            return 0.0
        
        h, w = depth_map.shape
        x = max(0, min(x, w-1))
        y = max(0, min(y, h-1))
        
        return float(depth_map[y, x])


class AdvancedGenderClassifier:
    """Clasificador de sexo mejorado con fusión altura + probabilidad"""
    
    def __init__(self):
        self.height_thresholds = {
            'child_max': 1.4,
            'adult_min': 1.5,
            'adult_typical': 1.7
        }
        self.gender_features = {
            'aspect_ratio_weight': 0.3,
            'color_analysis_weight': 0.4,
            'height_weight': 0.3
        }
    
    def classify_gender(self, bbox: Tuple[int, int, int, int], 
                       real_height: float, frame: np.ndarray) -> Tuple[str, float]:
        """Clasificar género con fusión de características"""
        x1, y1, x2, y2 = bbox
        
        # Extraer región de la persona
        person_region = frame[y1:y2, x1:x2]
        
        # 1. Análisis de altura
        height_score = self._analyze_height(real_height)
        
        # 2. Análisis de proporciones
        aspect_ratio = (x2 - x1) / (y2 - y1)
        proportion_score = self._analyze_proportions(aspect_ratio, real_height)
        
        # 3. Análisis de colores
        color_score = self._analyze_colors(person_region)
        
        # 4. Fusión de características
        total_score = (
            height_score * self.gender_features['height_weight'] +
            proportion_score * self.gender_features['aspect_ratio_weight'] +
            color_score * self.gender_features['color_analysis_weight']
        )
        
        # Determinar género y categoría
        if real_height < self.height_thresholds['child_max']:
            if total_score > 0.5:
                return "Niño", abs(total_score)
            else:
                return "Niña", abs(total_score)
        else:
            if total_score > 0.3:
                return "Hombre", abs(total_score)
            else:
                return "Mujer", abs(total_score)
    
    def _analyze_height(self, height: float) -> float:
        """Analizar altura para determinar género"""
        if height < self.height_thresholds['child_max']:
            return 0.0  # Neutral para niños
        
        # Adultos: hombres típicamente más altos
        if height > self.height_thresholds['adult_typical']:
            return 0.7  # Probablemente hombre
        elif height < 1.6:
            return -0.5  # Probablemente mujer
        else:
            return 0.0  # Neutral
    
    def _analyze_proportions(self, aspect_ratio: float, height: float) -> float:
        """Analizar proporciones corporales"""
        if height < self.height_thresholds['child_max']:
            return 0.0  # Neutral para niños
        
        # Adultos: hombres típicamente más anchos
        if aspect_ratio > 0.5:  # Más ancho
            return 0.6
        elif aspect_ratio < 0.3:  # Más delgado
            return -0.4
        else:
            return 0.0
    
    def _analyze_colors(self, person_region: np.ndarray) -> float:
        """Analizar colores de ropa"""
        if person_region.size == 0:
            return 0.0
        
        # Convertir a HSV para mejor análisis de colores
        hsv = cv2.cvtColor(person_region, cv2.COLOR_BGR2HSV)
        
        # Analizar colores típicos por género
        # Azules, grises, marrones (más masculinos)
        blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
        gray_mask = cv2.inRange(hsv, (0, 0, 50), (180, 30, 200))
        
        # Rosas, rojos, púrpuras (más femeninos)
        pink_mask = cv2.inRange(hsv, (140, 50, 50), (180, 255, 255))
        red_mask = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
        
        # Calcular proporciones
        total_pixels = person_region.shape[0] * person_region.shape[1]
        masculine_colors = (np.sum(blue_mask > 0) + np.sum(gray_mask > 0)) / total_pixels
        feminine_colors = (np.sum(pink_mask > 0) + np.sum(red_mask > 0)) / total_pixels
        
        # Retornar score basado en diferencia
        return (masculine_colors - feminine_colors) * 2


class AdvancedSurveillanceSystem:
    """Sistema de vigilancia avanzado con todas las características"""
    
    def __init__(self, use_gpu: bool = True, use_midas: bool = True):
        self.use_gpu = use_gpu
        self.use_midas = use_midas
        
        # Inicializar componentes
        self.detector = YOLO("yolov8n.pt")
        if use_gpu and torch.cuda.is_available():
            self.detector.to("cuda")
        
        self.auto_calibrator = AutoCalibrator()
        self.adaptive_smoother = AdaptiveSmoother()
        self.gender_classifier = AdvancedGenderClassifier()
        
        if use_midas:
            self.midas_estimator = MiDaSDepthEstimator(use_gpu=use_gpu)
        else:
            self.midas_estimator = None
        
        # Parámetros de cámara (serán calibrados automáticamente)
        self.camera_height = 7.0
        self.camera_angle = 90
        self.focal_length = 50.0
        self.sensor_height = 24.0
        
        print("🚀 Sistema de Vigilancia Avanzado inicializado")
        print(f"   GPU: {'✅' if use_gpu and torch.cuda.is_available() else '❌'}")
        print(f"   MiDaS: {'✅' if use_midas else '❌'}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Procesar frame completo"""
        frame_height, frame_width = frame.shape[:2]
        
        # Detectar personas
        results = self.detector(frame, conf=0.5)
        detections = results[0].boxes if len(results) > 0 else None
        
        if detections is None or len(detections) == 0:
            return frame
        
        # Obtener mapa de profundidad si está disponible
        depth_map = None
        if self.midas_estimator:
            depth_map = self.midas_estimator.estimate_depth_map(frame)
        
        # Procesar cada detección
        for i, detection in enumerate(detections):
            if detection.cls.item() != 0:  # Solo personas (clase 0)
                continue
            
            # Extraer bbox
            x1, y1, x2, y2 = map(int, detection.xyxy[0].cpu().numpy())
            conf = detection.conf.item()
            
            # Calcular altura real
            real_height, distance = self._calculate_real_height(
                (x1, y1, x2, y2), frame_height, depth_map
            )
            
            # Aplicar suavizado adaptativo
            smoothed_height, smoothed_distance = self.adaptive_smoother.update(
                real_height, distance
            )
            
            # Clasificar género
            gender, gender_confidence = self.gender_classifier.classify_gender(
                (x1, y1, x2, y2), smoothed_height, frame
            )
            
            # Agregar medición para calibración
            measurement = PersonMeasurement(
                bbox=(x1, y1, x2, y2),
                height_pixels=y2 - y1,
                real_height=smoothed_height,
                distance=smoothed_distance,
                confidence=conf,
                timestamp=0.0,  # Se puede agregar timestamp real
                position=((x1 + x2) // 2, (y1 + y2) // 2)
            )
            self.auto_calibrator.add_measurement(measurement)
            
            # Intentar calibrar si tenemos suficientes muestras
            if len(self.auto_calibrator.measurements) >= self.auto_calibrator.min_samples:
                calibration = self.auto_calibrator.calibrate()
                if calibration:
                    self.camera_height = calibration.camera_height
                    self.camera_angle = calibration.angle_degrees
                    print(f"🎯 Auto-calibración: {self.auto_calibrator.get_calibration_status()}")
            
            # Dibujar información
            self._draw_person_info(frame, (x1, y1, x2, y2), gender, 
                                 smoothed_height, smoothed_distance, 
                                 gender_confidence, conf)
        
        # Dibujar estado de calibración
        self._draw_calibration_status(frame)
        
        return frame
    
    def _calculate_real_height(self, bbox: Tuple[int, int, int, int], 
                              frame_height: int, depth_map: Optional[np.ndarray]) -> Tuple[float, float]:
        """Calcular altura real usando física óptica y/o MiDaS"""
        x1, y1, x2, y2 = bbox
        person_height_pixels = y2 - y1
        
        # Calcular distancia
        if depth_map is not None and self.midas_estimator:
            # Usar MiDaS para profundidad más precisa
            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
            distance = self.midas_estimator.get_depth_at_point(depth_map, center_x, center_y)
            if distance > 0:
                # Convertir a metros (ajustar según escala de MiDaS)
                distance = distance * 10  # Factor de escala aproximado
            else:
                distance = self._estimate_distance_physics(y2, frame_height)
        else:
            distance = self._estimate_distance_physics(y2, frame_height)
        
        # Calcular altura real
        focal_length_pixels = (self.focal_length * frame_height) / self.sensor_height
        real_height = (person_height_pixels * distance) / focal_length_pixels
        
        # Ajuste para cámara alta
        real_height *= 1.2  # Factor de corrección
        
        return max(0.5, min(2.5, real_height)), distance
    
    def _estimate_distance_physics(self, person_bottom_y: int, frame_height: int) -> float:
        """Estimar distancia usando física óptica"""
        normalized_y = person_bottom_y / frame_height
        fov_vertical = 2 * np.arctan(self.sensor_height / (2 * self.focal_length))
        angle_from_center = (normalized_y - 0.5) * fov_vertical
        camera_angle_rad = math.radians(self.camera_angle)
        total_angle = angle_from_center + camera_angle_rad
        
        if abs(total_angle) > 0.01:
            horizontal_distance = self.camera_height / np.tan(total_angle)
            distance = np.sqrt(horizontal_distance**2 + self.camera_height**2)
        else:
            distance = self.camera_height
        
        return max(2.0, min(25.0, distance))
    
    def _draw_person_info(self, frame: np.ndarray, bbox: Tuple[int, int, int, int],
                         gender: str, height: float, distance: float, 
                         gender_conf: float, detection_conf: float):
        """Dibujar información de la persona"""
        x1, y1, x2, y2 = bbox
        
        # Color según género
        colors = {
            "Hombre": (255, 0, 0),      # Azul
            "Mujer": (0, 255, 0),       # Verde
            "Niño": (0, 255, 255),      # Cian
            "Niña": (255, 0, 255)       # Magenta
        }
        color = colors.get(gender, (255, 255, 0))
        
        # Dibujar rectángulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Información de texto
        info_lines = [
            f"{gender} ({gender_conf:.2f})",
            f"Altura: {height:.2f}m",
            f"Distancia: {distance:.1f}m",
            f"Conf: {detection_conf:.2f}"
        ]
        
        # Dibujar líneas de información
        for i, line in enumerate(info_lines):
            y_pos = y1 - 10 - (i * 20)
            cv2.putText(frame, line, (x1, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Línea de profundidad
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        depth_line_length = int(distance * 3)
        end_x = center_x + depth_line_length
        cv2.line(frame, (center_x, center_y), (end_x, center_y), color, 2)
        cv2.circle(frame, (center_x, center_y), 3, color, -1)
    
    def _draw_calibration_status(self, frame: np.ndarray):
        """Dibujar estado de calibración"""
        status = self.auto_calibrator.get_calibration_status()
        cv2.putText(frame, f"Calibración: {status}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Sistema de Vigilancia Avanzado")
    parser.add_argument("--source", type=int, default=0, help="Fuente de video (0=cámara)")
    parser.add_argument("--gpu", action="store_true", default=True, help="Usar GPU")
    parser.add_argument("--no-midas", action="store_true", help="Desactivar MiDaS")
    parser.add_argument("--calibration-samples", type=int, default=10, 
                       help="Muestras mínimas para calibración")
    
    args = parser.parse_args()
    
    # Inicializar sistema
    system = AdvancedSurveillanceSystem(
        use_gpu=args.gpu,
        use_midas=not args.no_midas
    )
    
    # Configurar auto-calibrador
    system.auto_calibrator.min_samples = args.calibration_samples
    
    # Inicializar cámara
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print("❌ Error: No se pudo abrir la cámara")
        return
    
    print("🎥 Sistema iniciado. Presiona 'q' para salir.")
    print("📊 Características activas:")
    print("   ✅ Auto-calibración")
    print("   ✅ Suavizado adaptativo")
    print("   ✅ Clasificación de género mejorada")
    if not args.no_midas:
        print("   ✅ MiDaS para profundidad")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar frame
            processed_frame = system.process_frame(frame)
            
            # Mostrar frame
            cv2.imshow('Sistema de Vigilancia Avanzado', processed_frame)
            
            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Sistema detenido por usuario")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Sistema cerrado correctamente")


if __name__ == "__main__":
    main()

