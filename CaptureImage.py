# Nombre del código: CaptureImage.py
# Descripción: Este script utiliza Tkinter para mostrar una interfaz gráfica. 
# Carga los puntos de perspectiva desde un archivo .pkl, abre la cámara, y muestra
# en tiempo real la imagen transformada (480x480). Incluye un botón para capturar y guardar
# la imagen transformada con un nombre incremental.

import cv2
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import pickle
import os

# Leer las coordenadas del archivo .pkl
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

# Abrir la cámara
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error al abrir la cámara.")
    exit()

# Configuración de resolución y FPS (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# Crear la ventana principal de Tkinter
root = tk.Tk()
root.title("Transformación de Perspectiva en Tiempo Real")

# Etiqueta para mostrar el video
video_label = Label(root)
video_label.pack()

# Variable global para almacenar el frame transformado
transformed_frame = None

def obtener_siguiente_nombre():
    """
    Determina el siguiente número disponible para nombrar la captura.
    Busca el número más alto en la carpeta 'capturas' y retorna el siguiente.
    """
    os.makedirs("capturas", exist_ok=True)
    archivos = os.listdir("capturas")
    numeros = []
    
    for archivo in archivos:
        try:
            # Extraer el número del nombre del archivo
            numero = int(os.path.splitext(archivo)[0])
            numeros.append(numero)
        except ValueError:
            # Ignorar archivos que no sean numéricos
            pass
    
    # Determinar el siguiente número
    if numeros:
        return max(numeros) + 1
    else:
        return 0

def capturar_imagen():
    """
    Función para capturar y guardar la imagen transformada con un nombre incremental.
    """
    global transformed_frame
    if transformed_frame is not None:
        siguiente_numero = obtener_siguiente_nombre()
        filename = os.path.join("capturas", f"{siguiente_numero}.jpg")
        cv2.imwrite(filename, transformed_frame)
        print(f"Imagen guardada como: {filename}")
    else:
        print("No hay imagen transformada disponible para guardar.")

def actualizar_video():
    """
    Función para capturar el video de la cámara y mostrar la imagen transformada.
    """
    global transformed_frame
    ret, frame = cap.read()
    if ret:
        # Aplicar la transformación de perspectiva
        transformed = cv2.warpPerspective(frame, M, (480, 480))
        transformed_frame = transformed  # Almacenar el frame actual para captura
        
        # Convertir la imagen de OpenCV a un formato compatible con Tkinter
        transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(transformed_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Actualizar la etiqueta con la nueva imagen
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    
    # Llamar a la función de nuevo después de un breve retraso
    video_label.after(10, actualizar_video)

# Botón para capturar la imagen
capture_button = Button(root, text="Capturar Imagen", command=capturar_imagen)
capture_button.pack()

# Llamar a la función por primera vez
actualizar_video()

# Ejecutar el bucle principal de Tkinter
root.mainloop()

# Liberar la cámara al cerrar la aplicación
cap.release()
