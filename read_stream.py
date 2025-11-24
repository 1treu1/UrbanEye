import cv2

# URL del stream HLS
# Local: "http://<IP_RaspberryPi>:8888/stream.m3u8"
# Remoto (ngrok): "https://hazel-unsacked-hester.ngrok-free.dev/stream.m3u8"
stream_url = "https://hazel-unsacked-hester.ngrok-free.dev/stream.m3u8"

# Abrir el stream con OpenCV
cap = cv2.VideoCapture(stream_url)

if not cap.isOpened():
    print("No se pudo abrir el stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("No hay más frames o error en el stream")
        break

    # Mostrar el frame
    cv2.imshow("Stream RTSP/HLS", frame)

    # Salir con la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
