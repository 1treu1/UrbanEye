import cv2
import numpy as np

def draw_heart():
    # Dimensiones de la imagen
    width, height = 600, 600
    
    # Crear una imagen blanca (3 canales, 8 bits, valor 255)
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Puntos para el corazón
    # Usaremos ecuaciones paramétricas para generar una forma de corazón suave
    t = np.linspace(0, 2 * np.pi, 400)
    x = 16 * np.sin(t)**3
    y = 13 * np.cos(t) - 5 * np.cos(2*t) - 2 * np.cos(3*t) - np.cos(4*t)
    
    # Escalar y centrar
    scale = 15
    x = x * scale + width / 2
    # Invertir Y porque en imágenes la coordenada Y crece hacia abajo
    y = -y * scale + height / 2
    
    # Convertir a formato de puntos para polígonos (int32)
    # Necesita forma (n_puntos, 1, 2)
    points = np.stack((x, y), axis=1).astype(np.int32)
    points = points.reshape((-1, 1, 2))
    
    # Dibujar el corazón lleno de rojo
    # Color BGR: Rojo es (0, 0, 255)
    cv2.fillPoly(image, [points], color=(0, 0, 255))
    
    # Mostrar la imagen
    cv2.imshow("Corazon", image)
    
    # Esperar una tecla para cerrar
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    draw_heart()
