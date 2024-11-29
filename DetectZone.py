# Nombre del código: DetectZone.py
# Descripción: Este script captura una imagen estática desde la cámara, espera a que la cámara estabilice
# su enfoque, permite al usuario hacer clic en 4 esquinas en un orden específico, guarda las coordenadas en un archivo .pkl,
# y realiza una transformación homográfica para generar una imagen de 480x480 píxeles.

import cv2
import numpy as np
import pickle
import time

# Lista para almacenar las coordenadas de las esquinas
points = []

def select_points(event, x, y, flags, param):
    """
    Función de callback para capturar los clics del mouse.
    """
    global points, static_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:  # Solo permite seleccionar 4 puntos
            points.append((x, y))
            print(f"Punto seleccionado: {x}, {y}")
            # Dibujar un punto en la imagen donde se hizo clic
            cv2.circle(static_frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Selecciona las esquinas", static_frame)
        else:
            print("Ya seleccionaste 4 puntos. Pulsa 'q' para continuar.")

# Función para configurar la cámara
def configurar_camara(camara, resolucion=(1280, 720), fps=30):
    """
    Configura la resolución y los FPS de la cámara.
    """
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, resolucion[0])
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, resolucion[1])
    camara.set(cv2.CAP_PROP_FPS, fps)

# Abrir la cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Configurar resolución y FPS
configurar_camara(cap, resolucion=(1280, 720), fps=30)

print("""
Esperando que la cámara estabilice el enfoque...
Haz clic en las 4 esquinas de la región en el siguiente orden:
1. Superior izquierda
2. Superior derecha
3. Inferior derecha
4. Inferior izquierda
Presiona 'q' para confirmar la selección una vez hayas seleccionado los puntos.
""")

# Esperar unos segundos para que la cámara estabilice su autoenfoque
time.sleep(2)

# Capturar un solo frame como imagen estática después de estabilizar el autoenfoque
for _ in range(30):  # Leer algunos frames para estabilizar
    ret, static_frame = cap.read()
if not ret:
    print("No se puede capturar la imagen.")
    cap.release()
    exit()

cap.release()

cv2.namedWindow("Selecciona las esquinas")
cv2.setMouseCallback("Selecciona las esquinas", select_points)

# Mostrar la imagen estática para la selección
cv2.imshow("Selecciona las esquinas", static_frame)

while True:
    # Esperar a que el usuario presione 'q' para finalizar
    if cv2.waitKey(1) & 0xFF == ord('q') and len(points) == 4:
        break

cv2.destroyAllWindows()

if len(points) == 4:
    # Guardar las coordenadas en un archivo .pkl
    with open('points.pkl', 'wb') as file:
        pickle.dump(points, file)
    print(f"Coordenadas guardadas en 'points.pkl': {points}")
    
    # Definir el rectángulo destino (480x480)
    dest_points = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], dtype="float32")
    
    # Calcular la matriz de transformación (homografía)
    points = np.array(points, dtype="float32")
    M = cv2.getPerspectiveTransform(points, dest_points)
    
    # Aplicar la transformación a la imagen
    warped = cv2.warpPerspective(static_frame, M, (480, 480))
    
    # Mostrar la imagen transformada
    cv2.imshow("Transformacion Homografica (480x480)", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Guardar la imagen transformada
    cv2.imwrite("output_480x480.jpg", warped)
    print("Imagen transformada guardada como 'output_480x480.jpg'.")
else:
    print("No se seleccionaron 4 puntos. Intenta de nuevo.")
