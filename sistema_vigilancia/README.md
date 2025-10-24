# Sistema de Vigilancia Avanzado 🚀

Sistema de vigilancia inteligente con auto-calibración, estimación de profundidad y clasificación de género mejorada.

## 🌟 Características

- **Auto-calibración**: Se calibra automáticamente basándose en alturas observadas
- **Suavizado adaptativo**: Reduce el ruido en las mediciones en tiempo real
- **MiDaS Integration**: Estimación de profundidad usando modelos de deep learning (GPU)
- **Clasificación avanzada**: Fusión de altura + probabilidad para determinar género
- **Estructura modular**: Código organizado y fácil de mantener

## 🚀 Instalación Rápida

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd sistema_vigilancia
```

### 2. Crear entorno virtual
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
```

### 3. Instalar dependencias

#### Opción A: Instalación automática (recomendada)
```bash
pip install -r requirements.txt
```

#### Opción B: Con GPU (CUDA 11.8)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

#### Opción C: Con GPU (CUDA 12.1)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

#### Opción D: Solo CPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## 🎯 Uso en Cursor

### Ejecutar sistema básico
```bash
python -m src.main_simple_gender --source 0 --gpu
```

### Ejecutar sistema avanzado
```bash
python -m src.advanced_surveillance --source 0 --gpu
```

### Opciones disponibles
```bash
# Sistema básico
python -m src.main_simple_gender --source 0 --gpu --camera-height-meters 7.0 --use-physics

# Sistema avanzado
python -m src.advanced_surveillance --source 0 --gpu --calibration-samples 15

# Sin MiDaS (más rápido, menos preciso)
python -m src.advanced_surveillance --source 0 --gpu --no-midas
```

## 📊 Parámetros de Configuración

### Sistema Básico
- `--source`: Fuente de video (0=cámara, ruta=archivo)
- `--gpu`: Usar aceleración GPU
- `--camera-height-meters`: Altura de la cámara en metros
- `--camera-height`: Altura relativa (low/medium/high)
- `--camera-angle`: Ángulo de la cámara (normal/high/overhead)
- `--use-physics`: Usar cálculos de física óptica

### Sistema Avanzado
- `--source`: Fuente de video
- `--gpu`: Usar GPU para aceleración
- `--no-midas`: Desactivar MiDaS (más rápido)
- `--calibration-samples`: Muestras mínimas para auto-calibración

## 🔧 Arquitectura del Sistema

```
sistema_vigilancia/
├── src/
│   ├── main_simple_gender.py      # Sistema básico
│   ├── advanced_surveillance.py   # Sistema avanzado
│   ├── config.py                  # Configuración
│   └── application/
│       └── event_logger.py        # Logger de eventos
├── requirements.txt               # Dependencias
└── README.md                      # Este archivo
```

## 🧠 Componentes Principales

### 1. AutoCalibrator
- Se calibra automáticamente basándose en alturas observadas
- Usa estadísticas antropométricas para ajustar parámetros
- Mejora la precisión con el tiempo

### 2. AdaptiveSmoother
- Suavizado exponencial adaptativo
- Reduce ruido en mediciones
- Ajusta sensibilidad según variabilidad

### 3. MiDaSDepthEstimator
- Estimación de profundidad usando deep learning
- Funciona con GPU para mayor velocidad
- Mejora precisión de cálculos de altura

### 4. AdvancedGenderClassifier
- Fusión de múltiples características
- Análisis de altura, proporciones y colores
- Clasificación probabilística

## 📈 Mejoras Implementadas

### Auto-calibración
- ✅ Detecta automáticamente altura de cámara
- ✅ Ajusta ángulo de visión
- ✅ Usa estadísticas antropométricas
- ✅ Filtra outliers para mayor precisión

### Suavizado Adaptativo
- ✅ Reduce ruido en mediciones
- ✅ Ajusta sensibilidad automáticamente
- ✅ Mantiene responsividad

### MiDaS Integration
- ✅ Estimación de profundidad con deep learning
- ✅ Aceleración GPU
- ✅ Mejor precisión que métodos geométricos

### Clasificación Mejorada
- ✅ Fusión de altura + probabilidad
- ✅ Análisis de colores de ropa
- ✅ Proporciones corporales
- ✅ Clasificación probabilística

## 🎮 Controles

- **'q'**: Salir del sistema
- **ESC**: Salir del sistema
- **Ctrl+C**: Salir del sistema

## 🔍 Debugging

### Verificar GPU
```python
import torch
print(f"CUDA disponible: {torch.cuda.is_available()}")
print(f"Dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Verificar MiDaS
```python
import torch
model = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
print("MiDaS cargado correctamente")
```

## 📝 Logs y Debugging

El sistema muestra información detallada en consola:
- 🎯 Estado de calibración
- 📏 Mediciones de altura y distancia
- 🔍 Debug de cálculos
- ✅/❌ Estado de componentes

## 🚨 Solución de Problemas

### Error: "No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Error: "CUDA out of memory"
- Reducir resolución de video
- Usar `--no-midas` para desactivar MiDaS
- Cerrar otras aplicaciones que usen GPU

### Cámara lenta
- Verificar que GPU esté habilitado
- Usar `--gpu` flag
- Verificar drivers de CUDA

### Calibración incorrecta
- Aumentar `--calibration-samples`
- Asegurar que hay personas adultas en el frame
- Verificar altura de cámara real

## 📊 Rendimiento

### Con GPU + MiDaS
- **FPS**: 15-25 (dependiendo de hardware)
- **Precisión**: Alta
- **Uso GPU**: 2-4GB VRAM

### Con GPU sin MiDaS
- **FPS**: 25-35
- **Precisión**: Media-Alta
- **Uso GPU**: 1-2GB VRAM

### Solo CPU
- **FPS**: 5-10
- **Precisión**: Media
- **Uso CPU**: 80-100%

## 🤝 Contribuir

1. Fork el repositorio
2. Crear rama feature (`git checkout -b feature/nueva-caracteristica`)
3. Commit cambios (`git commit -am 'Agregar nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crear Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 🆘 Soporte

Para soporte técnico:
- Crear issue en GitHub
- Incluir logs de error
- Especificar configuración de hardware
- Incluir versión de Python y dependencias