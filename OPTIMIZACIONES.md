# Optimizaciones Implementadas en el Sistema de Vigilancia

## Resumen de Optimizaciones

Se implementaron las siguientes optimizaciones para mejorar significativamente el rendimiento del sistema:

### 1. **Throttling de DeepFace** ⚡
- **Antes**: DeepFace se ejecutaba en cada frame (muy costoso)
- **Ahora**: DeepFace se ejecuta cada 2 frames por defecto (`_DEEPFACE_FRAME_SKIP = 2`)
- **Beneficio**: Reduce el uso de GPU/CPU en ~50% para DeepFace
- **Impacto**: Mínimo impacto visual ya que las caras no cambian significativamente entre frames consecutivos

### 2. **Caché de Resultados de DeepFace** 💾
- **Antes**: Se recalculaban atributos en cada frame
- **Ahora**: Los resultados se cachean y reutilizan por 5 frames (`_DEEPFACE_CACHE_AGE = 5`)
- **Beneficio**: Evita recálculos innecesarios cuando una persona está estática o moviéndose lentamente
- **Impacto**: Reduce drásticamente el número de llamadas a DeepFace

### 3. **Eliminación de Doble Ejecución de DeepFace** 🔄
- **Antes**: DeepFace se ejecutaba:
  1. Una vez por cada track de YOLO (N llamadas)
  2. Una vez más en toda la ROI (1 llamada adicional)
- **Ahora**: Solo se ejecuta una vez en toda la ROI y se hace matching con IoU
- **Beneficio**: Reduce de (N+1) llamadas a DeepFace a solo 1 llamada por frame procesado
- **Impacto**: Si hay 5 personas, esto reduce de 6 llamadas a 1 llamada (83% menos)

### 4. **Conversión RGB Lazy (Pereza)** 🎨
- **Antes**: Se convertía el frame completo a RGB en cada frame
- **Ahora**: Solo se convierte cuando realmente se necesita ejecutar DeepFace
- **Beneficio**: Ahorra la conversión de color en ~50% de los frames
- **Impacto**: Reduce el tiempo de procesamiento en frames donde no se ejecuta DeepFace

### 5. **Limpieza Automática de Caché** 🧹
- Se eliminan automáticamente entradas de caché viejas (> 5 frames)
- Previene acumulación de memoria

## Mejoras de Rendimiento Estimadas

### Antes de las Optimizaciones:
- DeepFace: 6-10 llamadas por frame (dependiendo de número de personas)
- Conversión RGB: Cada frame
- Sin caché: Todo se recalcula

### Después de las Optimizaciones:
- DeepFace: 0.5 llamadas por frame en promedio (1 cada 2 frames)
- Conversión RGB: Solo cuando se ejecuta DeepFace
- Con caché: Reutilización inteligente de resultados

### Mejora Total Estimada:
- **Reducción de ~80-90%** en el tiempo de procesamiento de DeepFace
- **Reducción de ~50%** en conversiones de color innecesarias
- **Ahorro de memoria**: Caché con limpieza automática

## Configuración Ajustable

Los parámetros de optimización se pueden ajustar en el código:

```python
_DEEPFACE_FRAME_SKIP: int = 2      # Aumentar para más rendimiento, disminuir para más precisión
_DEEPFACE_CACHE_AGE: int = 5       # Frames antes de que expire el caché
```

### Recomendaciones:
- **Alta calidad**: `FRAME_SKIP = 1`, `CACHE_AGE = 3`
- **Balanceado** (default): `FRAME_SKIP = 2`, `CACHE_AGE = 5`
- **Alto rendimiento**: `FRAME_SKIP = 3`, `CACHE_AGE = 8`

## Notas Adicionales

- El tracking de YOLO no se ve afectado (sigue siendo cada frame)
- Las estelas de tracking se mantienen fluidas
- La calidad de detección se mantiene gracias al matching inteligente con IoU
- El sistema es más responsive y puede manejar más personas simultáneamente

