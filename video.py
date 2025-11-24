import cv2
import time

# URL del stream HLS
stream_url = "https://hazel-unsacked-hester.ngrok-free.dev/stream.m3u8"

# Abrir el stream con OpenCV
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("No se pudo abrir el stream")
    exit()

# Obtener ancho, alto y FPS del stream
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 15  # Valor por defecto si el stream no reporta FPS

# Configurar VideoWriter para guardar los primeros 30 segundos
output_filename = "video_30s.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec MP4
out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

print(f"📹 Grabando los primeros 30 segundos en {output_filename}...")

start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        print("No hay más frames o error en el stream")
        break

    # Mostrar el frame
    cv2.imshow("Stream RTSP/HLS", frame)

    # Guardar frame en el video mientras no hayan pasado 30s
    elapsed_time = time.time() - start_time
    if elapsed_time <= 30:
        out.write(frame)
    else:
        print("⏹️  Se alcanzaron los 30 segundos de grabación")
        break

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
