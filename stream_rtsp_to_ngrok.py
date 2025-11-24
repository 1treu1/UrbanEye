import os
import subprocess
import time
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

try:
    from pyngrok import ngrok
    PYNGROK_AVAILABLE = True
except ImportError:
    PYNGROK_AVAILABLE = False
    print("⚠️  pyngrok no está instalado. Instálalo con: pip install pyngrok")
    print("   O usa ngrok CLI manualmente")

# Configuración
RTSP_URL = "rtsp://1treu1:treutapo@192.168.1.40:554/stream1"
HLS_OUTPUT_DIR = "hls_output"
HLS_PLAYLIST = os.path.join(HLS_OUTPUT_DIR, "stream.m3u8")
HTTP_PORT = 8888

# Crear directorio de salida
Path(HLS_OUTPUT_DIR).mkdir(exist_ok=True)

class HLSRequestHandler(SimpleHTTPRequestHandler):
    """Handler personalizado para servir archivos HLS desde el directorio correcto"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=HLS_OUTPUT_DIR, **kwargs)
    
    def end_headers(self):
        # Agregar headers CORS y cache para HLS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache')
        super().end_headers()

def start_http_server():
    """Inicia un servidor HTTP simple para servir los archivos HLS"""
    handler = HLSRequestHandler
    httpd = HTTPServer(("localhost", HTTP_PORT), handler)
    print(f"✅ Servidor HTTP iniciado en http://localhost:{HTTP_PORT}")
    httpd.serve_forever()

def start_ffmpeg_stream():
    """Inicia FFmpeg para convertir RTSP a HLS"""
    # Limpiar archivos HLS anteriores
    for file in Path(HLS_OUTPUT_DIR).glob("*.ts"):
        file.unlink()
    if Path(HLS_PLAYLIST).exists():
        Path(HLS_PLAYLIST).unlink()
    
    ffmpeg_cmd = [
    "ffmpeg",
    "-rtsp_transport", "tcp",
    "-i", RTSP_URL,
    "-c:v", "copy",
    "-c:a", "aac",
    "-f", "hls",
    HLS_PLAYLIST
]


    
    print(f"🔄 Iniciando conversión RTSP → HLS...")
    print(f"   Stream RTSP: {RTSP_URL}")
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    except FileNotFoundError:
        print("❌ Error: FFmpeg no está instalado o no está en el PATH")
        print("   Instálalo desde: https://ffmpeg.org/download.html")
        return None

def main():
    print("=" * 60)
    print("🚀 Exponiendo Stream RTSP con ngrok")
    print("=" * 60)
    
    # Verificar FFmpeg
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("❌ Error: FFmpeg no está instalado")
        print("   Windows: choco install ffmpeg")
        print("   O descarga desde: https://ffmpeg.org/download.html")
        return
    
    # Iniciar servidor HTTP en un hilo separado
    http_thread = threading.Thread(target=start_http_server, daemon=True)
    http_thread.start()
    time.sleep(1)  # Dar tiempo al servidor para iniciar
    
    # Iniciar FFmpeg
    ffmpeg_process = start_ffmpeg_stream()
    if not ffmpeg_process:
        return
    
    # Esperar a que se genere el primer archivo HLS
    print("⏳ Esperando a que se genere el stream HLS...")
    max_wait = 30
    waited = 0
    while not Path(HLS_PLAYLIST).exists() and waited < max_wait:
        time.sleep(1)
        waited += 1
    
    if not Path(HLS_PLAYLIST).exists():
        print("❌ Error: No se pudo generar el stream HLS")
        print("   Verifica que el stream RTSP sea accesible")
        ffmpeg_process.terminate()
        return
    
    print("✅ Stream HLS generado correctamente")
    
    # Exponer con ngrok
    if PYNGROK_AVAILABLE:
        try:
            # Configurar ngrok (si tienes un token, configúralo aquí)
            # ngrok.set_auth_token("tu_token_aqui")
            
            public_url = ngrok.connect(HTTP_PORT)
            ngrok_url = f"{public_url}/stream.m3u8"
            
            print("\n" + "=" * 60)
            print("🌐 Stream expuesto públicamente:")
            print(f"   {ngrok_url}")
            print("=" * 60)
            print("\n💡 Usa esta URL en prueba.py:")
            print(f'   url = "{ngrok_url}"')
            print("\n⏹️  Presiona Ctrl+C para detener el stream")
            print("=" * 60)
            
            # Mantener el script corriendo
            try:
                while True:
                    time.sleep(1)
                    # Verificar que FFmpeg sigue corriendo
                    if ffmpeg_process.poll() is not None:
                        print("❌ FFmpeg se detuvo inesperadamente")
                        break
            except KeyboardInterrupt:
                print("\n\n🛑 Deteniendo stream...")
                ffmpeg_process.terminate()
                ngrok.kill()
                print("✅ Stream detenido correctamente")
        except Exception as e:
            print(f"❌ Error al conectar con ngrok: {e}")
            print("\n💡 Alternativa: Usa ngrok CLI manualmente:")
            print(f"   ngrok http {HTTP_PORT}")
            print(f"\n   Luego usa: http://localhost:{HTTP_PORT}/stream.m3u8")
            print("   (reemplaza localhost con la URL de ngrok)")
    else:
        print(f"\n💡 Inicia ngrok manualmente en otra terminal:")
        print(f"   ngrok http {HTTP_PORT}")
        print(f"\n   Luego usa: http://localhost:{HTTP_PORT}/stream.m3u8")
        print("   (reemplaza localhost con la URL de ngrok)")
        print("\n⏹️  Presiona Ctrl+C para detener el stream")
        
        try:
            while True:
                time.sleep(1)
                if ffmpeg_process.poll() is not None:
                    print("❌ FFmpeg se detuvo inesperadamente")
                    break
        except KeyboardInterrupt:
            print("\n\n🛑 Deteniendo stream...")
            ffmpeg_process.terminate()
            print("✅ Stream detenido correctamente")

if __name__ == "__main__":
    main()

