# SmartRecyclingCNC.py
# Este script realiza detección en tiempo real de botellas y tapas utilizando un modelo YOLOv8 de segmentación.
# Muestra las segmentaciones de ambas clases, pero solo las botellas inician el proceso de reciclaje.
# Cuando se detecta una botella quieta, se envía un G-code a la CNC para su ejecución.

import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import pickle
from ultralytics import YOLO
import torch
import serial
import time
import os

# Configuración de parámetros
OPACITY = 0.3                   # Opacidad de las máscaras (0.0 a 1.0)
FPS = 12                        # Fotogramas por segundo
CONFIDENCE_THRESHOLD = 0.87     # Umbral de confianza para detecciones (0.0 a 1.0)
TOLERANCE = 5                   # Tolerancia en píxeles para considerar que el objeto está quieto
STILL_FRAMES = 27              # Número de frames consecutivos para confirmar que el objeto está quieto
MAX_FRAMES_NO_BOTTLE = 15       # Frames máximos sin detectar botella antes de mostrar mensaje
MESSAGE_DISPLAY_FRAMES = 18     # Frames para mostrar el mensaje después de detener el análisis
CNC_PORT = 'COM20'              # Puerto serie para la CNC
GCODE_FILE = 'gcode.txt'        # Ruta al archivo de G-code

# Forzar el uso de CPU
device = torch.device('cpu')

# Cargar el modelo YOLOv8 de segmentación
model = YOLO('YOLO8nSegPlaticB.pt')  # Reemplaza con la ruta de tu modelo

# Definir colores para las clases
MASK_COLORS = {
    'Bottles': (0, 255, 0),  # Verde
    'Lids': (0, 0, 255)      # Azul
}

# Obtener los IDs de clase para "Bottles" y "Lids"
class_names = model.names
bottle_class_id = None
lid_class_id = None
for cls_id, cls_name in class_names.items():
    if cls_name.lower() == "bottles":
        bottle_class_id = cls_id
    elif cls_name.lower() == "lids":
        lid_class_id = cls_id

if bottle_class_id is None:
    print("Error: La clase 'Bottles' no se encuentra en el modelo.")
    exit()

if lid_class_id is None:
    print("Error: La clase 'Lids' no se encuentra en el modelo.")
    exit()

# Leer las coordenadas para la transformación de perspectiva
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

# Configuración de la transformación de perspectiva (480x480)
dest_points = np.array([[0, 0], [480, 0], [480, 480], [0, 480]], dtype="float32")
points = np.array(points, dtype="float32")
M = cv2.getPerspectiveTransform(points, dest_points)

# Abrir la cámara en el índice 1
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Configuración de la cámara
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Conectar a la CNC
try:
    cnc_serial = serial.Serial(CNC_PORT, 115200, timeout=1)
    time.sleep(2)  # Esperar a que la conexión se establezca
    print(f"Conectado a la CNC en {CNC_PORT}")
except serial.SerialException as e:
    print(f"Error al conectar con la CNC en {CNC_PORT}: {e}")
    exit()

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Smart Recycling CNC")

# Etiquetas para mostrar el video y mensajes
video_label = Label(root)
video_label.pack()
message_label = Label(root, text="", font=("Arial", 16), fg="red")
message_label.pack()

# Variables de control
analyzing = False
recycling_started = False
centroid_history = []
frame_counter = 0
message_display_counter = 0

# Función para calcular el centroide de una máscara
def calculate_centroid(mask):
    moments = cv2.moments(mask)
    if moments['m00'] == 0:
        return None
    cx = int(moments['m10'] / moments['m00'])
    cy = int(moments['m01'] / moments['m00'])
    return (cx, cy)

# Intervalo entre frames en milisegundos
interval = int(1000 / FPS)

def start_analysis():
    global analyzing, centroid_history, recycling_started, frame_counter, message_display_counter
    analyzing = True
    recycling_started = False
    centroid_history = []
    frame_counter = 0
    message_display_counter = 0
    message_label.config(text="Analizando...")

def update_video():
    global analyzing, centroid_history, recycling_started, frame_counter, message_display_counter
    ret, frame = cap.read()
    if ret:
        # Inicializar found_bottle como False al inicio
        found_bottle = False

        # Aplicar la transformación de perspectiva
        transformed = cv2.warpPerspective(frame, M, (480, 480))

        # Convertir a RGB
        transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)

        # Crear una imagen overlay para las máscaras
        overlay = transformed_rgb.copy()

        if analyzing and not recycling_started:
            frame_counter += 1  # Incrementar el contador de frames

            # Realizar la inferencia
            results = model(transformed_rgb, device='cpu', conf=CONFIDENCE_THRESHOLD, verbose=False)

            # Procesar las detecciones y aplicar las máscaras
            if results:
                result = results[0]
                masks = result.masks
                boxes = result.boxes
                if masks is not None and boxes is not None:
                    overlay = overlay.astype(np.float32)
                    current_centroid = None

                    for mask, box, score, cls_id in zip(masks.data, boxes.xyxy, boxes.conf, boxes.cls):
                        cls_id_int = int(cls_id)
                        cls_name = model.names[cls_id_int]

                        if cls_name in MASK_COLORS and score >= CONFIDENCE_THRESHOLD:
                            # Convertir la máscara a numpy
                            mask_np = mask.cpu().numpy().astype(np.uint8)

                            # Obtener el color correspondiente a la clase
                            color = MASK_COLORS[cls_name]

                            # Aplicar la máscara
                            for c in range(3):
                                overlay[:, :, c] = np.where(mask_np == 1,
                                                            overlay[:, :, c] * (1 - OPACITY) + color[c] * OPACITY,
                                                            overlay[:, :, c])
                            # Dibujar la caja delimitadora
                            x1, y1, x2, y2 = map(int, box.cpu().numpy())
                            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                            # Etiqueta de la clase y confianza
                            class_label = f"{cls_name} {score:.2f}"
                            cv2.putText(overlay, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 0, 0), 2)

                            if cls_id_int == bottle_class_id:
                                found_bottle = True
                                # Calcular el centroide de la botella
                                centroid = calculate_centroid(mask_np)
                                if centroid:
                                    current_centroid = centroid

                    if found_bottle and current_centroid:
                        # Agregar el centroide a la historia
                        centroid_history.append(current_centroid)
                        if len(centroid_history) > STILL_FRAMES:
                            centroid_history.pop(0)

                        # Evaluar si el objeto está quieto
                        if len(centroid_history) == STILL_FRAMES:
                            # Calcular el desplazamiento máximo
                            displacements = [np.linalg.norm(np.array(centroid_history[i]) - np.array(centroid_history[i - 1]))
                                             for i in range(1, len(centroid_history))]
                            max_displacement = max(displacements)
                            if max_displacement <= TOLERANCE:
                                message_label.config(text="Iniciando reciclaje")
                                analyzing = False
                                recycling_started = True
                                message_display_counter = MESSAGE_DISPLAY_FRAMES
                                send_gcode_to_cnc()
                    else:
                        # Reiniciar si no se detecta la botella
                        centroid_history = []
                else:
                    centroid_history = []
            else:
                centroid_history = []

            # Verificar si ha pasado el máximo de frames sin detectar botella
            if frame_counter >= MAX_FRAMES_NO_BOTTLE and not found_bottle:
                message_label.config(text="Reciclaje no encontrado")
                analyzing = False
                message_display_counter = MESSAGE_DISPLAY_FRAMES
        elif recycling_started:
            if message_display_counter > 0:
                message_display_counter -= 1
            else:
                message_label.config(text="")
        else:
            if message_display_counter > 0:
                message_display_counter -= 1
            else:
                message_label.config(text="")

        # Convertir overlay a uint8
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)

        # Crear imagen para Tkinter
        img = Image.fromarray(overlay)
        imgtk = ImageTk.PhotoImage(image=img)

        # Actualizar la etiqueta con la nueva imagen
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Programar la siguiente llamada
    video_label.after(interval, update_video)

def send_gcode_to_cnc():
    try:
        # Leer el archivo de G-code
        if not os.path.isfile(GCODE_FILE):
            print(f"Error: El archivo de G-code '{GCODE_FILE}' no existe.")
            return

        with open(GCODE_FILE, 'r') as file:
            gcode_lines = file.readlines()

        # Enviar cada línea de G-code a la CNC
        for line in gcode_lines:
            # Eliminar espacios en blanco y líneas vacías
            line = line.strip()
            if line == '':
                continue
            # Enviar la línea al CNC
            cnc_serial.write((line + '\n').encode())
            print(f"Enviado: {line}")
            # Esperar la respuesta de la CNC
            response = cnc_serial.readline().decode().strip()
            print(f"Respuesta CNC: {response}")
            # Agregar un pequeño retraso si es necesario
            time.sleep(0.1)

        print("G-code enviado exitosamente a la CNC.")
    except Exception as e:
        print(f"Error al enviar G-code a la CNC: {e}")

# Botón para iniciar el análisis
analyze_button = Button(root, text="Analizar", command=start_analysis)
analyze_button.pack()

# Iniciar el bucle de actualización de video
update_video()

# Iniciar el bucle principal de Tkinter
root.mainloop()

# Liberar recursos al cerrar la aplicación
cap.release()
cnc_serial.close()
