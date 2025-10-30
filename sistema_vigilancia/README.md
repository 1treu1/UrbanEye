## Sistema de Vigilancia Inteligente (Arquitectura Hexagonal)

Este proyecto implementa un sistema de vigilancia en tiempo real con detección de personas (YOLOv8), seguimiento por ID (DeepSORT), estimación de edad y género (DeepFace), detección de objetos portados (bolsa/mochila/compras), estimación de profundidad (MiDaS) y conteo de entradas/salidas en un área poligonal. La arquitectura sigue el patrón de puertos y adaptadores (hexagonal) para facilitar la extensibilidad y el reemplazo de componentes.

### Características
- Detección de personas con YOLOv8 (Ultralytics)
- Seguimiento multi-objeto con DeepSORT (ID único por persona)
- Estimación de edad y género con DeepFace
- Detección de objetos portados: bolsa, mochila, cartera, maleta (aprox.)
- Estimación de profundidad monocular (MiDaS)
- Área poligonal de conteo: eventos de entrada/salida
- Visualización en tiempo real (OpenCV) y registro CSV

### Requisitos
Consulte `requirements.txt` para dependencias de Python. Requiere Python 3.9+ (recomendado), y una GPU CUDA es opcional pero recomendable para rendimiento.

### Instalación
```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Si utiliza Windows, ejecute los comandos con PowerShell o CMD; el script `run.sh` puede ejecutarse con Git Bash/WSL. Alternativamente, lance manualmente `python -m src.main`.

### Estructura
```
sistema_vigilancia/
├── README.md
├── requirements.txt
├── run.sh
└── src/
    ├── main.py
    ├── config.py
    ├── domain/
    │   ├── models.py
    │   ├── services.py
    │   └── utils/
    │       ├── area_utils.py
    │       ├── depth_utils.py
    │       ├── tracking_utils.py
    │       └── draw_utils.py
    ├── ports/
    │   ├── detection_port.py
    │   ├── tracking_port.py
    │   ├── age_gender_port.py
    │   └── depth_port.py
    ├── adapters/
    │   ├── yolo_adapter.py
    │   ├── deepsort_adapter.py
    │   ├── deepface_adapter.py
    │   └── midas_adapter.py
    └── application/
        ├── video_service.py
        └── event_logger.py
```

### Configuración
Edite `src/config.py` para definir:
- Fuente de video: archivo local, cámara (`0`) o RTSP.
- Área poligonal de conteo (lista de puntos `[(x,y), ...]`).
- Umbrales y opciones de rendimiento.

Puede pasar la fuente por CLI:
```bash
python -m src.main --source data/video.mp4  # o rtsp://user:pass@host/...
```

### Ejecución
```bash
bash run.sh  # Linux/macOS/WSL
# En Windows sin bash:
python -m src.main --source 0  # webcam
```

### Extensibilidad
- Para cambiar el detector, cree un nuevo adaptador que implemente `DetectionPort` y ajústelo en `main.py` mediante inyección de dependencias.
- Lo mismo para seguimiento, edad/género y profundidad implementando sus puertos correspondientes.

### Salida
- Ventana OpenCV con anotaciones (IDs, edad/género, estado de objeto portado, profundidad y área).
- CSV en `data/logs/events.csv` con columnas: `timestamp,track_id,age,gender,carries_object,depth,inside_area`.

### Notas de rendimiento
- Para RTSP en tiempo real, prefiera modelos livianos (`yolov8n.pt`), y considere reducir resolución de entrada.
- DeepFace puede ser costoso por frame; por defecto se evalúa por pista cada N frames (ajustable en `config.py`).





cd /workspace/UrbanEye/sistema_vigilancia && . ../.venv/bin/activate 2>/dev/null || true && PORT=7860 python -m src.main --ui http --port 7860