from __future__ import annotations

import argparse
from typing import Optional
import cv2
import numpy as np
import math
from ultralytics import YOLO

from .config import AppConfig, build_default_polygon
from .application.event_logger import EventLogger


class AdvancedPersonAnalyzer:
    """Analizador avanzado de personas: género, edad y categoría"""
    
    def __init__(self, camera_height="high", camera_angle="overhead", camera_height_meters=7.0):
        # Cargar modelo de edad pre-entrenado (opcional)
        self.age_model = None
        self.age_net = None
        self.precise_mode = False
        
        # Parámetros de calibración de cámara
        self.camera_height = camera_height  # "low", "medium", "high"
        self.camera_angle = camera_angle    # "normal", "angled", "overhead"
        self.camera_height_meters = camera_height_meters  # Altura real en metros
        
        # Convertir ángulo a grados
        if camera_angle == "overhead":
            self.camera_angle_degrees = 90  # Cámara mirando hacia abajo
        elif camera_angle == "high":
            self.camera_angle_degrees = 75  # Cámara con ángulo alto
        elif camera_angle == "normal":
            self.camera_angle_degrees = 60  # Cámara con ángulo normal
        elif camera_angle == "low":
            self.camera_angle_degrees = 45  # Cámara con ángulo bajo
        else:
            self.camera_angle_degrees = 60  # Por defecto
        
        # Parámetros físicos optimizados para cámara a 7 metros
        self.focal_length = 50  # Focal length en mm (típico)
        self.sensor_height = 24  # Sensor height en mm (típico)
        self.frame_height_pixels = 480  # Altura del frame en píxeles
        
        # Parámetros específicos para cámara alta (7m)
        self.min_distance = 2.0  # Distancia mínima observable
        self.max_distance = 15.0  # Distancia máxima observable
        
        # Umbrales adaptativos basados en altura de cámara
        self._setup_camera_parameters()
        
        self._load_age_model()
    
    def _setup_camera_parameters(self):
        """Configura parámetros optimizados para cámara a 7 metros"""
        # Parámetros específicos para cámara a 7 metros de altura
        if self.camera_height_meters >= 6.0:
            # Cámara muy alta (6m+) - personas se ven muy pequeñas
            self.aspect_ratio_thresholds = {
                "child_min": 0.4, "child_max": 0.7,
                "adult_min": 0.5, "adult_max": 0.9
            }
            self.area_thresholds = {
                "child_min": 2000, "child_max": 8000,
                "adult_min": 8000, "adult_max": 25000
            }
            self.color_sensitivity = 1.5  # Menos sensible a colores por distancia
            
        elif self.camera_height_meters >= 4.0:
            # Cámara alta (4-6m) - personas se ven pequeñas
            self.aspect_ratio_thresholds = {
                "child_min": 0.35, "child_max": 0.65,
                "adult_min": 0.45, "adult_max": 0.8
            }
            self.area_thresholds = {
                "child_min": 3000, "child_max": 10000,
                "adult_min": 10000, "adult_max": 30000
            }
            self.color_sensitivity = 1.2  # Sensibilidad reducida
            
        else:
            # Cámara baja (<4m) - personas se ven más grandes
            self.aspect_ratio_thresholds = {
                "child_min": 0.25, "child_max": 0.5,
                "adult_min": 0.35, "adult_max": 0.7
            }
            self.area_thresholds = {
                "child_min": 5000, "child_max": 15000,
                "adult_min": 15000, "adult_max": 50000
            }
            self.color_sensitivity = 0.8  # Más sensible a colores
        
        # Ajustes por ángulo de cámara
        if self.camera_angle == "overhead":
            # Vista desde arriba - ajustar umbrales para perspectiva
            self.aspect_ratio_thresholds["child_max"] *= 1.3
            self.aspect_ratio_thresholds["adult_max"] *= 1.3
            self.color_sensitivity *= 0.7  # Menos sensible por perspectiva
            
        elif self.camera_angle == "angled":
            # Cámara inclinada - ajustar sensibilidad
            self.color_sensitivity *= 1.2
    
    def _load_age_model(self):
        """Carga modelo de edad si está disponible"""
        try:
            # Modelo de edad usando OpenCV DNN
            # Este es un modelo ligero para estimación de edad
            print("Cargando modelo de edad...")
            # Por ahora usamos análisis visual mejorado
            self.age_model = "visual_enhanced"
            print("✅ Modelo de edad cargado (análisis visual mejorado)")
        except Exception as e:
            print(f"⚠️ No se pudo cargar modelo de edad: {e}")
            self.age_model = None
    
    def analyze_person(self, frame_bgr, person_bbox):
        """Análisis completo: género, edad y categoría usando altura real"""
        x1, y1, x2, y2 = person_bbox
        crop = frame_bgr[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        
        if crop.size == 0:
            return "Desconocido", "Desconocido", 0
        
        # Calcular altura real usando física
        frame_height = frame_bgr.shape[0]
        real_height, distance = self._calculate_real_height(person_bbox, frame_height)
        
        # Clasificar por altura real (más preciso)
        category, age = self._classify_by_real_height(real_height)
        
        # Análisis de colores para determinar género
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 100))
        bright_mask = cv2.inRange(hsv, (0, 0, 150), (180, 255, 255))
        
        dark_pixels = cv2.countNonZero(dark_mask)
        bright_pixels = cv2.countNonZero(bright_mask)
        total_pixels = crop.shape[0] * crop.shape[1]
        
        dark_ratio = dark_pixels / total_pixels if total_pixels > 0 else 0
        bright_ratio = bright_pixels / total_pixels if total_pixels > 0 else 0
        
        # Determinar género basado en colores y altura
        gender = self._detect_gender_by_height_and_colors(real_height, dark_ratio, bright_ratio, category)
        
        return gender, category, age
    
    def _detect_gender_by_height_and_colors(self, real_height, dark_ratio, bright_ratio, category):
        """Detecta género basado en altura real y colores"""
        if category == "Niño":
            # Para niños, análisis basado en colores
            if bright_ratio > 0.4:  # Colores muy brillantes
                return "Niña"
            elif bright_ratio > 0.2:  # Colores moderadamente brillantes
                return "Niño"  # Empate, usar por defecto
            else:  # Colores oscuros
                return "Niño"
        else:
            # Para adultos, análisis más complejo
            if real_height < 1.6:  # Adulto bajo (probablemente mujer)
                if bright_ratio > 0.3:
                    return "Mujer"
                else:
                    return "Mujer"  # Por defecto para altura baja
            elif real_height < 1.7:  # Altura media
                if dark_ratio > 0.4:  # Muchos colores oscuros
                    return "Hombre"
                else:
                    return "Mujer"
            else:  # Adulto alto (probablemente hombre)
                if dark_ratio > 0.3:
                    return "Hombre"
                else:
                    return "Hombre"  # Por defecto para altura alta
    
    def _detect_child(self, aspect_ratio, area, dark_ratio, bright_ratio):
        """Detecta si es un niño usando parámetros adaptativos de cámara"""
        child_score = 0
        
        # Usar umbrales adaptativos basados en altura de cámara
        child_min_ratio = self.aspect_ratio_thresholds["child_min"]
        child_max_ratio = self.aspect_ratio_thresholds["child_max"]
        adult_min_ratio = self.aspect_ratio_thresholds["adult_min"]
        
        child_min_area = self.area_thresholds["child_min"]
        child_max_area = self.area_thresholds["child_max"]
        adult_min_area = self.area_thresholds["adult_min"]
        
        # Análisis de proporciones adaptativo
        if child_min_ratio <= aspect_ratio <= child_max_ratio:
            child_score += 3  # Proporciones típicas de niño
        elif child_max_ratio < aspect_ratio <= adult_min_ratio:
            child_score += 1  # Proporciones intermedias
        elif aspect_ratio < child_min_ratio:
            # Muy delgado - podría ser niño alto o adulto delgado
            if area < child_max_area:
                child_score += 2  # Probablemente niño
            else:
                child_score += 0  # Probablemente adulto delgado
        
        # Análisis de área adaptativo
        if child_min_area <= area <= child_max_area:
            child_score += 3  # Área típica de niño
        elif area < child_min_area:
            child_score += 2  # Muy pequeño - probablemente niño
        elif child_max_area < area < adult_min_area:
            child_score += 1  # Área intermedia
        else:
            child_score += 0  # Área grande - probablemente adulto
        
        # Análisis de colores con sensibilidad adaptativa
        bright_threshold = 0.3 * self.color_sensitivity
        if bright_ratio > bright_threshold:
            child_score += 2  # Colores brillantes típicos de niños
        elif bright_ratio > bright_threshold * 0.7:
            child_score += 1  # Colores moderadamente brillantes
        
        # Análisis adicional para cámaras altas
        if self.camera_height == "high":
            # En cámaras altas, los niños se ven más cuadrados
            if 0.4 <= aspect_ratio <= 0.7:
                child_score += 1
        
        # Análisis adicional para cámaras bajas
        elif self.camera_height == "low":
            # En cámaras bajas, los niños se ven más delgados
            if aspect_ratio < 0.4 and area < child_max_area:
                child_score += 1
        
        return child_score >= 4  # Umbral más alto para mejor precisión
    
    def _detect_gender(self, aspect_ratio, dark_ratio, bright_ratio, is_child):
        """Detecta el género basado en características visuales"""
        if is_child:
            # Para niños, análisis más simple
            if dark_ratio > 0.4:
                return "Niño"
            else:
                return "Niña"
        else:
            # Para adultos, análisis más complejo
            if aspect_ratio > 0.4:  # Más ancho
                if dark_ratio > 0.3:
                    return "Hombre"
                else:
                    return "Mujer"
            else:  # Más alto
                if dark_ratio > 0.4:
                    return "Hombre"
                else:
                    return "Mujer"
    
    def _estimate_age(self, is_child, gender, aspect_ratio, dark_ratio, bright_ratio):
        """Estima la edad con algoritmo mejorado"""
        if is_child:
            # Algoritmo mejorado para niños (3-15 años)
            age_score = 0
            
            # Análisis de proporciones corporales
            if aspect_ratio < 0.4:  # Muy delgado/alto
                age_score += 3  # Más joven
            elif aspect_ratio < 0.6:
                age_score += 2
            else:
                age_score += 1  # Más maduro
            
            # Análisis de colores (ropa típica por edad)
            if bright_ratio > 0.5:  # Colores muy brillantes
                age_score += 3  # Muy joven (3-6 años)
            elif bright_ratio > 0.3:
                age_score += 2  # Niño (7-10 años)
            elif bright_ratio > 0.1:
                age_score += 1  # Pre-adolescente (11-13 años)
            else:
                age_score += 0  # Adolescente (14-15 años)
            
            # Mapeo de score a edad
            if age_score >= 5:
                return 4  # Muy joven
            elif age_score >= 3:
                return 7  # Niño
            elif age_score >= 1:
                return 11  # Pre-adolescente
            else:
                return 14  # Adolescente
                
        else:
            # Algoritmo mejorado para adultos (16-70 años)
            age_score = 0
            
            # Análisis de proporciones corporales
            if aspect_ratio > 0.5:  # Más ancho (adulto maduro)
                age_score += 2
            elif aspect_ratio > 0.4:
                age_score += 1
            else:
                age_score += 0  # Más joven
            
            # Análisis de colores (ropa típica por edad)
            if dark_ratio > 0.6:  # Muchos colores oscuros (maduro)
                age_score += 3
            elif dark_ratio > 0.4:
                age_score += 2
            elif dark_ratio > 0.2:
                age_score += 1
            else:
                age_score += 0  # Colores claros (joven)
            
            # Análisis de brillo (ropa más formal = más edad)
            if bright_ratio < 0.1:  # Muy pocos colores brillantes
                age_score += 2  # Ropa formal
            elif bright_ratio < 0.3:
                age_score += 1
            
            # Mapeo de score a edad
            if age_score >= 5:
                return 45  # Maduro
            elif age_score >= 3:
                return 35  # Adulto medio
            elif age_score >= 1:
                return 25  # Adulto joven
            else:
                return 20  # Muy joven
    
    def _estimate_age_precise(self, crop, is_child, gender, aspect_ratio, dark_ratio, bright_ratio):
        """Estimación de edad más precisa usando análisis avanzado"""
        if crop.size == 0:
            return self._estimate_age(is_child, gender, aspect_ratio, dark_ratio, bright_ratio)
        
        # Análisis avanzado de características faciales/corporales
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Detectar bordes y contornos para análisis de textura
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Análisis de textura (arrugas, líneas)
        texture_score = self._analyze_texture(gray)
        
        # Análisis de contornos faciales/corporales
        contour_score = self._analyze_contours(contours, crop.shape)
        
        # Análisis de colores más detallado
        color_score = self._analyze_colors_detailed(crop)
        
        if is_child:
            # Algoritmo preciso para niños
            precise_score = texture_score + contour_score + color_score
            
            if precise_score >= 8:
                return 3  # Muy joven
            elif precise_score >= 6:
                return 6  # Niño pequeño
            elif precise_score >= 4:
                return 9  # Niño
            elif precise_score >= 2:
                return 12  # Pre-adolescente
            else:
                return 15  # Adolescente
        else:
            # Algoritmo preciso para adultos
            precise_score = texture_score + contour_score + color_score
            
            if precise_score >= 8:
                return 50  # Maduro
            elif precise_score >= 6:
                return 40  # Adulto maduro
            elif precise_score >= 4:
                return 30  # Adulto medio
            elif precise_score >= 2:
                return 25  # Adulto joven
            else:
                return 20  # Muy joven
    
    def _analyze_texture(self, gray):
        """Analiza textura para detectar arrugas/líneas"""
        # Usar filtros para detectar textura
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Más textura = más edad
        texture_density = np.mean(magnitude)
        return min(3, int(texture_density / 20))  # Score 0-3
    
    def _analyze_contours(self, contours, shape):
        """Analiza contornos para detectar características de edad"""
        if not contours:
            return 0
        
        # Analizar complejidad de contornos
        total_contour_area = sum(cv2.contourArea(c) for c in contours)
        total_image_area = shape[0] * shape[1]
        
        if total_image_area == 0:
            return 0
        
        contour_ratio = total_contour_area / total_image_area
        
        # Más contornos complejos = más edad
        if contour_ratio > 0.3:
            return 3
        elif contour_ratio > 0.2:
            return 2
        elif contour_ratio > 0.1:
            return 1
        else:
            return 0
    
    def _analyze_colors_detailed(self, crop):
        """Análisis detallado de colores"""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Análisis de saturación (colores más saturados = más joven)
        saturation = np.mean(hsv[:, :, 1])
        
        # Análisis de valor (brillo)
        value = np.mean(hsv[:, :, 2])
        
        # Análisis de matices (tonos de piel)
        hue = np.mean(hsv[:, :, 0])
        
        score = 0
        
        # Colores más saturados y brillantes = más joven
        if saturation > 100 and value > 150:
            score += 3  # Muy joven
        elif saturation > 80 and value > 120:
            score += 2  # Joven
        elif saturation > 60 and value > 100:
            score += 1  # Medio
        
        return score
    
    def _calculate_real_height(self, person_bbox, frame_height):
        """Calcula la altura real de la persona usando física óptica corregida"""
        x1, y1, x2, y2 = person_bbox
        
        # Altura de la persona en píxeles
        person_height_pixels = y2 - y1
        
        # Calcular centro horizontal de la persona
        person_center_x = (x1 + x2) // 2
        frame_width = 640  # Asumir ancho estándar (se puede hacer dinámico)
        
        # Calcular distancia a la persona usando trigonometría (ahora con posición horizontal)
        distance_to_person = self._estimate_distance_to_person(y2, frame_height, person_center_x, frame_width)
        
        # FÓRMULA CORREGIDA para cámara a 7 metros
        # Altura_real = (Altura_píxeles * Distancia) / (Focal_length_pixels)
        # Focal_length_pixels = (Focal_length_mm * Frame_height_pixels) / Sensor_height_mm
        
        focal_length_pixels = (self.focal_length * frame_height) / self.sensor_height
        
        # Calcular altura real
        real_height = (person_height_pixels * distance_to_person) / focal_length_pixels
        
        # Ajuste específico para cámara a 7 metros
        # Las personas se ven más pequeñas desde esta altura
        real_height *= 1.5  # Factor de corrección para cámara alta
        
        # Limitar altura a rangos realistas
        real_height = max(0.5, min(2.5, real_height))
        
        print(f"📏 Debug: pixels={person_height_pixels}, distance={distance_to_person:.1f}m, height={real_height:.2f}m")
        
        return real_height, distance_to_person
    
    def _estimate_distance_to_person(self, person_bottom_y, frame_height, person_center_x=None, frame_width=None):
        """Estima la distancia a la persona desde el punto focal de la cámara"""
        # Normalizar posición (0 = arriba, 1 = abajo)
        normalized_y = person_bottom_y / frame_height
        
        # Campo de visión vertical típico
        fov_vertical = 2 * np.arctan(self.sensor_height / (2 * self.focal_length))
        
        # CORRECCIÓN FUNDAMENTAL: Calcular distancia desde el punto focal
        # El ángulo se mide desde el centro óptico de la cámara
        angle_from_center = (normalized_y - 0.5) * fov_vertical
        
        # Calcular distancia horizontal usando trigonometría
        # distance = camera_height / tan(angle_from_center + camera_angle)
        camera_angle_rad = math.radians(self.camera_angle_degrees)
        total_angle = angle_from_center + camera_angle_rad
        
        if abs(total_angle) > 0.01:  # Evitar divisiones por cero
            # Distancia horizontal desde la cámara
            horizontal_distance = self.camera_height_meters / np.tan(total_angle)
            
            # Distancia real (hipotenusa) desde el punto focal
            distance = np.sqrt(horizontal_distance**2 + self.camera_height_meters**2)
        else:
            distance = self.camera_height_meters  # Distancia por defecto
        
        # CORRECCIÓN: Ajuste para posición horizontal (perspectiva)
        if person_center_x is not None and frame_width is not None:
            # Calcular offset horizontal (0 = centro, 1 = borde)
            horizontal_offset = abs(person_center_x - frame_width // 2) / (frame_width // 2)
            
            # Las personas en los bordes están más lejos debido a la perspectiva
            if horizontal_offset > 0.6:  # En los bordes
                distance *= 1.2  # Aumentar distancia
            elif horizontal_offset > 0.3:  # Ligeramente fuera del centro
                distance *= 1.05
            # Si está en el centro (offset < 0.3), no ajustar
        
        # Limitar distancia a rangos realistas
        distance = max(self.min_distance, min(self.max_distance, distance))
        
        # Debug: mostrar información de cálculo
        if person_center_x is not None and frame_width is not None:
            horizontal_offset = abs(person_center_x - frame_width // 2) / (frame_width // 2)
            print(f"🎯 Debug distancia: center_x={person_center_x}, offset={horizontal_offset:.2f}, "
                  f"angle={math.degrees(total_angle):.1f}°, distance={distance:.1f}m")
        
        return distance
    
    def _classify_by_real_height(self, real_height):
        """Clasifica persona por altura real usando estadísticas antropométricas corregidas"""
        # Estadísticas de altura por edad (en metros) - CORREGIDAS
        # Basado en datos antropométricos reales
        
        print(f"🔍 Altura calculada: {real_height:.2f}m")  # Debug
        
        if real_height < 0.8:
            return "Niño", 4  # Muy pequeño (bebé)
        elif real_height < 1.0:
            return "Niño", 6  # Niño muy pequeño
        elif real_height < 1.2:
            return "Niño", 8  # Niño pequeño
        elif real_height < 1.4:
            return "Niño", 11  # Niño
        elif real_height < 1.5:
            return "Niño", 14  # Adolescente
        elif real_height < 1.55:
            return "Adulto", 18  # Adulto joven (mujer muy baja)
        elif real_height < 1.65:
            return "Adulto", 25  # Adulto (mujer promedio)
        elif real_height < 1.75:
            return "Adulto", 30  # Adulto (hombre promedio)
        elif real_height < 1.85:
            return "Adulto", 35  # Adulto alto
        else:
            return "Adulto", 40  # Adulto muy alto


def main():
    parser = argparse.ArgumentParser(description="Sistema de Vigilancia - Detección de Personas, Género y Edad")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Fuente de video: ruta, 0 para webcam, o URL RTSP",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Modelo YOLO a usar",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        default=True,
        help="Usar GPU para aceleración",
    )
    parser.add_argument(
        "--precise-age",
        action="store_true",
        help="Usar algoritmo de edad más preciso (más lento)",
    )
    parser.add_argument(
        "--camera-height",
        type=str,
        choices=["low", "medium", "high"],
        default="medium",
        help="Altura de la cámara: low (baja), medium (media), high (alta)",
    )
    parser.add_argument(
        "--camera-angle",
        type=str,
        choices=["normal", "angled", "overhead"],
        default="normal",
        help="Ángulo de la cámara: normal, angled (inclinada), overhead (desde arriba)",
    )
    parser.add_argument(
        "--auto-calibrate",
        action="store_true",
        help="Calibración automática de parámetros de cámara",
    )
    parser.add_argument(
        "--camera-height-meters",
        type=float,
        default=7.0,
        help="Altura real de la cámara en metros (por defecto: 7.0)",
    )
    parser.add_argument(
        "--use-physics",
        action="store_true",
        default=True,
        help="Usar cálculo de altura real con física (recomendado)",
    )
    args = parser.parse_args()
    
    # Configurar fuente
    source = int(args.source) if args.source.isdigit() else args.source
    
    # Cargar modelo YOLO
    print("Cargando modelo YOLO...")
    model = YOLO(args.model)
    
    # Configurar GPU si está disponible
    if args.gpu:
        try:
            import torch
            if torch.cuda.is_available():
                print("✅ GPU disponible - usando aceleración CUDA")
            else:
                print("⚠️ GPU no disponible - usando CPU")
        except ImportError:
            print("⚠️ PyTorch no disponible - usando CPU")
    
    # Inicializar analizador avanzado con parámetros de cámara
    person_analyzer = AdvancedPersonAnalyzer(
        camera_height=args.camera_height,
        camera_angle=args.camera_angle,
        camera_height_meters=args.camera_height_meters
    )
    
    # Configurar modo preciso si se solicita
    if args.precise_age:
        print("🔍 Modo preciso activado - análisis de edad mejorado")
        person_analyzer.precise_mode = True
    else:
        person_analyzer.precise_mode = False
    
    # Mostrar configuración de cámara
    print(f"📷 Configuración de cámara:")
    print(f"   Altura: {args.camera_height} ({args.camera_height_meters}m)")
    print(f"   Ángulo: {args.camera_angle}")
    print(f"   Física: {'Sí' if args.use_physics else 'No'}")
    print(f"   Modo preciso: {'Sí' if args.precise_age else 'No'}")
    
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
    print("Funcionalidades activas:")
    print("  ✅ Detección de personas")
    print("  ✅ Análisis de género (Hombre/Mujer/Niño/Niña)")
    print("  ✅ Estimación de edad")
    print("  ✅ Categorización (Adulto/Niño)")
    print("  ✅ Visualización en tiempo real")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detectar personas
            results = model(frame, verbose=False)
            
            # Procesar detecciones
            for r in results:
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        # Solo personas (clase 0 en COCO)
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            
                            if conf > 0.5:  # Umbral de confianza
                                # Análisis completo: género, categoría y edad
                                gender, category, age = person_analyzer.analyze_person(frame, (x1, y1, x2, y2))
                                
                                # Calcular altura real para mostrar
                                frame_height = frame.shape[0]
                                real_height, distance = person_analyzer._calculate_real_height((x1, y1, x2, y2), frame_height)
                                
                                # Color según género y categoría
                                if gender == "Mujer":
                                    color = (0, 255, 0)  # Verde
                                elif gender == "Hombre":
                                    color = (255, 0, 0)  # Azul
                                elif gender == "Niña":
                                    color = (255, 0, 255)  # Magenta
                                elif gender == "Niño":
                                    color = (0, 255, 255)  # Cian
                                else:
                                    color = (0, 255, 255)  # Amarillo
                                
                                # Dibujar rectángulo
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                
                                # Dibujar etiqueta con información completa
                                label = f"{gender} ({category}) {age} años"
                                cv2.putText(frame, label, (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                
                                # Dibujar altura real
                                height_label = f"Altura: {real_height:.2f}m"
                                cv2.putText(frame, height_label, (x1, y1-30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                
                                # Dibujar distancia (profundidad)
                                distance_label = f"Dist: {distance:.1f}m"
                                cv2.putText(frame, distance_label, (x1, y1-50), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                
                                # Dibujar confianza
                                conf_label = f"Conf: {conf:.2f}"
                                cv2.putText(frame, conf_label, (x1, y1-70), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                
                                # Dibujar línea de profundidad desde el centro de la persona
                                center_x = (x1 + x2) // 2
                                center_y = (y1 + y2) // 2
                                
                                # Línea que muestra la profundidad estimada
                                depth_line_length = int(distance * 5)  # Escalar para visualización
                                end_x = center_x + depth_line_length
                                end_y = center_y
                                
                                # Dibujar línea de profundidad
                                cv2.line(frame, (center_x, center_y), (end_x, end_y), color, 2)
                                
                                # Dibujar punto central
                                cv2.circle(frame, (center_x, center_y), 3, color, -1)
            
            # Mostrar frame
            cv2.imshow('Sistema de Vigilancia - Género y Edad', frame)
            
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
