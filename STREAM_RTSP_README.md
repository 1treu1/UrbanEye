# Exponer Stream RTSP con ngrok

Este script convierte tu stream RTSP a HLS y lo expone públicamente a través de ngrok.

## Requisitos

1. **ffmpeg**: Necesario para convertir RTSP a HLS
   - Windows: `choco install ffmpeg` o descargar desde https://ffmpeg.org/download.html
   - Mac: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg`

2. **ngrok** (opcional, una de estas opciones):
   - **Opción A**: Instalar pyngrok (recomendado): `pip install pyngrok`
   - **Opción B**: Instalar ngrok CLI desde https://ngrok.com/download

## Uso

### Opción 1: Script con pyngrok (Recomendado)

```bash
python stream_rtsp_to_ngrok_v2.py
```

### Opción 2: Script con ngrok CLI

```bash
python stream_rtsp_to_ngrok.py
```

## Configuración

Edita el archivo para cambiar el stream RTSP:

```python
RTSP_URL = "rtsp://1treu1:treutapo@192.168.1.40:554/stream1"
```

## Cómo funciona

1. **ffmpeg** convierte el stream RTSP a HLS (HTTP Live Streaming)
2. Un **servidor HTTP** local sirve los archivos HLS
3. **ngrok** expone el servidor HTTP públicamente

El script mostrará la URL pública donde puedes acceder al stream.

## Usar el stream en prueba.py

Una vez que ejecutes el script, copia la URL que muestra (algo como `https://xxxx.ngrok-free.dev/stream.m3u8`) y úsala en `prueba.py`:

```python
url = "https://xxxx.ngrok-free.dev/stream.m3u8"
```

## Notas

- El script crea una carpeta `hls_output` con los segmentos HLS
- Presiona `Ctrl+C` para detener el script
- El stream tiene un pequeño delay (2-6 segundos) debido a la conversión HLS















