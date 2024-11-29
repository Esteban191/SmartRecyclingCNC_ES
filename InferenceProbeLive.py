# InferenceProbeLive.py
# Este script abre la cámara y realiza inferencia en tiempo real usando un modelo YOLOv8 de segmentación.
# Los parámetros como opacidad, FPS y umbral de confianza se configuran directamente en el código.
# Se aplican diferentes máscaras para cada etiqueta (clase).
# Basado en CaptureImage.py para la interfaz y apertura de cámara, y en InferenceProbe.py para la inferencia.

import cv2
import tkinter as tk
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
import pickle
from ultralytics import YOLO
import torch

# Configuración de parámetros
OPACITY = 0.3                   # Opacidad de la máscara (entre 0.0 y 1.0)
FPS = 30                        # FPS de procesamiento
CONFIDENCE_THRESHOLD = 0.87     # Umbral de confianza para detecciones (entre 0.0 y 1.0)

# Forzar el uso de CPU
device = torch.device('cpu')

# Cargar el modelo YOLOv8 de segmentación en CPU
model = YOLO('YOLO8nSegPlaticB.pt')  # Reemplaza con la ruta de tu modelo

# Definir colores para cada clase
class_colors = {}
unique_classes = model.names  # Obtener las clases únicas del modelo
np.random.seed(42)  # Para obtener colores consistentes en cada ejecución
for cls_id, cls_name in unique_classes.items():
    # Generar un color aleatorio para cada clase
    color = tuple(np.random.randint(0, 255, size=3).tolist())
    class_colors[cls_id] = color

# Leer las coordenadas del archivo 'points.pkl' para la transformación de perspectiva
try:
    with open('points.pkl', 'rb') as file:
        points = pickle.load(file)
        if len(points) != 4:
            raise ValueError("El archivo 'points.pkl' no contiene exactamente 4 puntos.")
except FileNotFoundError:
    print("Error: El archivo 'points.pkl' no se encuentra. Genera primero las coordenadas.")
    exit()
except Exception as e:
    print(f"Error al cargar 'points.pkl': {e}")
    exit()

# Configuración del destino para la transformación (480x480)
dest_points = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], dtype="float32")
points = np.array(points, dtype="float32")
M = cv2.getPerspectiveTransform(points, dest_points)

# Abrir la cámara en el índice 1
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Configuración de la cámara (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Segmentación en Tiempo Real con YOLOv8")

# Etiqueta para mostrar el video
video_label = Label(root)
video_label.pack()

# Cálculo del intervalo entre fotogramas en milisegundos
interval = int(1000 / FPS)

def update_video():
    ret, frame = cap.read()
    if ret:
        # Aplicar la transformación de perspectiva
        transformed = cv2.warpPerspective(frame, M, (480, 480))

        # Convertir a RGB
        transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)

        # Realizar la inferencia
        results = model(transformed_rgb, device='cpu', conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Crear una imagen overlay para las máscaras
        overlay = transformed_rgb.copy()

        # Procesar las detecciones y aplicar las máscaras
        if results:
            result = results[0]
            masks = result.masks  # Máscaras de segmentación
            boxes = result.boxes  # Cajas delimitadoras
            if masks is not None and boxes is not None:
                overlay = overlay.astype(np.float32)
                for mask, box, score, cls_id in zip(masks.data, boxes.xyxy, boxes.conf, boxes.cls):
                    if score >= CONFIDENCE_THRESHOLD:
                        # Convertir la máscara a numpy
                        mask = mask.cpu().numpy().astype(np.uint8)

                        # Obtener el color de la clase
                        cls_id_int = int(cls_id)
                        color = class_colors[cls_id_int]

                        # Aplicar la máscara con el color y opacidad especificados
                        for c in range(3):
                            overlay[:, :, c] = np.where(mask == 1,
                                                        overlay[:, :, c] * (1 - OPACITY) + color[c] * OPACITY,
                                                        overlay[:, :, c])
                        # Dibujar la caja delimitadora
                        x1, y1, x2, y2 = map(int, box.cpu().numpy())
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                        # Poner la etiqueta de la clase y confianza
                        class_label = f"{model.names[cls_id_int]} {score:.2f}"
                        cv2.putText(overlay, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Convertir overlay de vuelta a uint8
                overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            else:
                # No hay máscaras detectadas
                overlay = transformed_rgb
        else:
            # No hay resultados
            overlay = transformed_rgb

        # Convertir de nuevo a BGR para mostrar en Tkinter
        display_image = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        # Convertir a imagen de PIL y luego a ImageTk
        img = Image.fromarray(display_image)
        imgtk = ImageTk.PhotoImage(image=img)

        # Actualizar la etiqueta con la nueva imagen
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Programar la siguiente llamada
    video_label.after(interval, update_video)

# Iniciar el bucle de actualización de video
update_video()

# Iniciar el bucle principal de Tkinter
root.mainloop()

# Liberar la cámara al cerrar la aplicación
cap.release()
