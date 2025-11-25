import os
import subprocess
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Verificar si pyngrok está disponible
try:
    from pyngrok import ngrok
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False
    print("⚠️  pyngrok no está instalado. Instálalo con: pip install pyngrok")

# ---------------- CONFIGURACIÓN ---------------- #
RTSP_URL = "rtsp://1treu1:treutapo@192.168.1.40:554/stream2"
HLS_OUTPUT_DIR = "hls"
HLS_PLAYLIST = os.path.join(HLS_OUTPUT_DIR, "stream.m3u8")
HTTP_PORT = 8888
# ------------------------------------------------ #

# Crear directorio HLS si no existe
Path(HLS_OUTPUT_DIR).mkdir(exist_ok=True)

class HLSRequestHandler(SimpleHTTPRequestHandler):
    """Servir archivos HLS con headers correctos"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HLS_OUTPUT_DIR, **kwargs)

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

def start_http_server():
    """Servidor HTTP para los archivos HLS"""
    httpd = HTTPServer(("0.0.0.0", HTTP_PORT), HLSRequestHandler)
    print(f"✅ Servidor HTTP iniciado en LAN: http://<TU_IP_LOCAL>:{HTTP_PORT}")
    httpd.serve_forever()

def start_ffmpeg_stream():
    """Convertir RTSP a HLS usando FFmpeg"""
    # Limpiar archivos HLS anteriores
    for file in Path(HLS_OUTPUT_DIR).glob("*"):
        file.unlink()

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-rtsp_transport", "tcp",
        "-timeout", "5000000",  # Timeout de socket (5s)
        "-i", RTSP_URL,
        "-c:v", "copy",
        "-c:a", "aac",
        "-f", "hls",
        "-hls_flags", "delete_segments+append_list",
        "-hls_time", "2",
        "-hls_list_size", "10", # Mantener lista pequeña para evitar fugas de memoria
        HLS_PLAYLIST
    ]

    print(f"🔄 Iniciando conversión RTSP → HLS...")
    print(f"   Stream RTSP: {RTSP_URL}")

    try:
        # IMPORTANTE: Usar DEVNULL para evitar que el buffer del pipe se llene y cuelgue el proceso
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return process
    except FileNotFoundError:
        print("❌ Error: FFmpeg no está instalado o no está en el PATH")
        return None

def main():
    # Verificar FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Error: FFmpeg no está instalado")
        return

    # Iniciar servidor HTTP en hilo separado
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)

    # Iniciar FFmpeg
    ffmpeg_process = start_ffmpeg_stream()
    if not ffmpeg_process:
        return

    # Esperar que se genere el primer archivo HLS
    print("⏳ Esperando a que se genere el stream HLS...")
    
    waited = 0
    max_wait = 30
    while not Path(HLS_PLAYLIST).exists() and waited < max_wait:
        time.sleep(1)
        waited += 1

    if not Path(HLS_PLAYLIST).exists():
        print("❌ Error: No se pudo generar el stream HLS")
        ffmpeg_process.terminate()
        return

    print("✅ Stream HLS generado correctamente")
    print(f"🔗 URL LAN: http://<TU_IP_LOCAL>:{HTTP_PORT}/stream.m3u8")

    # Exponer vía ngrok si está disponible
    if PYNGROK_AVAILABLE:
        try:
            public_url = ngrok.connect(HTTP_PORT)
            print(f"🌐 Stream público con ngrok: {public_url}/stream.m3u8")
        except Exception as e:
            print(f"❌ Error al conectar con ngrok: {e}")

    # Mantener el script corriendo y reiniciar FFMPEG si muere o se cuelga
    try:
        while True:
            time.sleep(1)
            
            # 1. Verificar si el proceso murió
            if ffmpeg_process.poll() is not None:
                print("❌ FFmpeg se detuvo inesperadamente. Reiniciando en 5 segundos...")
                time.sleep(5)
                ffmpeg_process = start_ffmpeg_stream()
                continue

            # 2. Watchdog: Verificar si el archivo m3u8 se está actualizando
            # Si no se actualiza en X segundos, FFmpeg está colgado
            try:
                if Path(HLS_PLAYLIST).exists():
                    mtime = Path(HLS_PLAYLIST).stat().st_mtime
                    age = time.time() - mtime
                    if age > 20: # Si el archivo tiene más de 20s de antigüedad
                        print(f"⚠️ Watchdog: El stream parece colgado (sin cambios en {int(age)}s). Reiniciando FFmpeg...")
                        ffmpeg_process.kill()
                        ffmpeg_process.wait()
                        time.sleep(2)
                        ffmpeg_process = start_ffmpeg_stream()
            except Exception as e:
                print(f"⚠️ Error en watchdog: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Deteniendo stream...")
        if ffmpeg_process:
            ffmpeg_process.terminate()
        if PYNGROK_AVAILABLE:
            ngrok.kill()
        print("✅ Stream detenido correctamente")

if __name__ == "__main__":
    main()
