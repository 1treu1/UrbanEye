import os
import subprocess
import time
import threading
import argparse
import glob
import shutil
from pathlib import Path
from google.cloud import storage

# ---------------- CONFIGURACIÓN DEFAULT ---------------- #
RTSP_URL_DEFAULT = "rtsp://1treu1:treutapo@192.168.1.40:554/stream2"
# ------------------------------------------------------- #

class GCSUploader:
    def __init__(self, bucket_name, credentials_path, remote_prefix=""):
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.remote_prefix = remote_prefix
        self.storage_client = storage.Client.from_service_account_json(credentials_path)
        self.bucket = self.storage_client.bucket(bucket_name)
        self.uploaded_files = set()

    def upload_file(self, local_path, is_playlist=False):
        """Sube un archivo a GCS en un hilo separado."""
        def _upload():
            try:
                filename = os.path.basename(local_path)
                # Construir path remoto: lugar/camara/archivo
                remote_path = f"{self.remote_prefix}/{filename}" if self.remote_prefix else filename
                
                blob = self.bucket.blob(remote_path)
                
                # Configurar headers de caché
                if is_playlist:
                    blob.cache_control = "no-cache, max-age=0"
                else:
                    blob.cache_control = "public, max-age=3600"

                blob.upload_from_filename(local_path)
                print(f"☁️  [Upload] Subido: {filename} -> gs://{self.bucket_name}/{remote_path}")
                
                if not is_playlist:
                    self.uploaded_files.add(filename)
                    # Opcional: Borrar archivo local después de subir (para ahorrar espacio)
                    # os.remove(local_path) 

            except Exception as e:
                print(f"❌ [Upload] Error subiendo {local_path}: {e}")

        thread = threading.Thread(target=_upload)
        thread.daemon = True
        thread.start()

def start_ffmpeg_stream(rtsp_url, output_dir, hls_playlist_path):
    """Convertir RTSP a HLS usando FFmpeg"""
    # Limpiar archivos HLS anteriores
    if os.path.exists(output_dir):
        # Borrar contenido pero mantener directorio
        for file in Path(output_dir).glob("*"):
            try:
                file.unlink()
            except:
                pass
    else:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-rtsp_transport", "tcp",
        "-timeout", "5000000",  # Timeout de socket (5s)
        "-i", rtsp_url,
        "-c:v", "copy",         # Copiar video sin transcodificar (muy rápido)
        "-c:a", "aac",          # Audio AAC
        "-f", "hls",
        "-hls_flags", "delete_segments+append_list", # Borrar segmentos viejos del m3u8 local
        "-hls_time", "4",       # Duración del segmento (s)
        "-hls_list_size", "5",  # Número de segmentos en la lista
        hls_playlist_path
    ]

    print(f"🔄 Iniciando conversión RTSP → HLS...")
    print(f"   Stream RTSP: {rtsp_url}")
    print(f"   Output Dir:  {output_dir}")

    try:
        process = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return process
    except FileNotFoundError:
        print("❌ Error: FFmpeg no está instalado o no está en el PATH")
        return None

def get_credentials_path():
    """Busca el archivo de credenciales en la carpeta config"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(base_dir, "config")
    cred_files = glob.glob(os.path.join(config_dir, "*.json"))
    if not cred_files:
        raise FileNotFoundError("No se encontraron credenciales JSON en la carpeta config/")
    return cred_files[0]

def main():
    parser = argparse.ArgumentParser(description="Stream RTSP to GCS HLS")
    parser.add_argument("--url", type=str, default=RTSP_URL_DEFAULT, help="URL RTSP de entrada")
    parser.add_argument("--minutes", type=int, default=1, help="Duración (No usado para streaming continuo, mantenido por compatibilidad)")
    parser.add_argument("--out", type=str, default="hls_stream", help="Directorio de salida local")
    parser.add_argument("--bucket", type=str, required=True, help="Nombre del Bucket GCS")
    parser.add_argument("--lugar", type=str, default="default_place", help="Nombre del lugar (para ruta GCS)")
    parser.add_argument("--camara", type=str, default="default_cam", help="Nombre de la cámara (para ruta GCS)")
    
    args = parser.parse_args()

    try:
        credentials_path = get_credentials_path()
        print(f"🔑 Usando credenciales: {credentials_path}")
    except Exception as e:
        print(f"❌ {e}")
        return

    # Construir prefijo remoto: lugar/camara
    # Ejemplo: viva_envigado/camara_01
    remote_prefix = f"{args.lugar}/{args.camara}"
    
    # Configurar paths locales
    hls_output_dir = args.out
    hls_playlist = os.path.join(hls_output_dir, "stream.m3u8")

    uploader = GCSUploader(args.bucket, credentials_path, remote_prefix=remote_prefix)

    # Iniciar FFmpeg
    ffmpeg_process = start_ffmpeg_stream(args.url, hls_output_dir, hls_playlist)
    if not ffmpeg_process:
        return

    print("⏳ Esperando a que se genere el stream HLS...")
    
    # Loop principal
    try:
        last_m3u8_mtime = 0
        
        while True:
            time.sleep(1)
            
            # 1. Verificar si FFmpeg murió
            if ffmpeg_process.poll() is not None:
                print("❌ FFmpeg se detuvo. Reiniciando en 5s...")
                time.sleep(5)
                ffmpeg_process = start_ffmpeg_stream(args.url, hls_output_dir, hls_playlist)
                continue

            # 2. Verificar archivos nuevos para subir
            if os.path.exists(hls_output_dir):
                # Subir segmentos .ts nuevos
                for ts_file in Path(hls_output_dir).glob("*.ts"):
                    filename = ts_file.name
                    if filename not in uploader.uploaded_files:
                        uploader.upload_file(str(ts_file), is_playlist=False)

                # Subir playlist .m3u8 si cambió
                if os.path.exists(hls_playlist):
                    mtime = os.path.getmtime(hls_playlist)
                    if mtime > last_m3u8_mtime:
                        uploader.upload_file(hls_playlist, is_playlist=True)
                        last_m3u8_mtime = mtime
            
    except KeyboardInterrupt:
        print("\n🛑 Deteniendo stream...")
        ffmpeg_process.terminate()
        print("✅ Listo.")

if __name__ == "__main__":
    main()
