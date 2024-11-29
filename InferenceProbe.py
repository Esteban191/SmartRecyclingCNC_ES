# YOLOv8 Segmentación Básica

# Importar las librerías
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Forzar uso de CPU (opcional, YOLOv8 lo ajusta automáticamente)
import torch
torch.device('cpu')

# Cargar el modelo YOLOv8 segmentador
model = YOLO('YOLO8nSegPlaticB.pt')  # Cambia por la ruta de tu modelo

# Ruta de la imagen
image_path = 'capturas/54.jpg'  # Cambia por la ruta de tu imagen

# Procesar la imagen con el modelo
results = model(image_path)

# Leer la imagen original
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Crear una copia de la imagen para aplicar las máscaras translúcidas
overlay = image.copy()

# Colores únicos para cada clase
class_colors = {}
unique_classes = model.names  # Obtener las clases únicas del modelo
for i, cls_name in unique_classes.items():
    class_colors[cls_name] = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

# Dibujar las máscaras translúcidas y las etiquetas
for mask, box, score, cls_id in zip(results[0].masks.data, results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
    # Transferir la máscara de la GPU a la CPU y convertir a numpy
    mask = mask.cpu().numpy().astype(np.uint8)

    # Aplicar la máscara translúcida al overlay
    color = class_colors[unique_classes[int(cls_id)]]
    for c in range(3):  # Aplicar el color en los tres canales (RGB)
        overlay[:, :, c] = np.where(mask == 1, overlay[:, :, c] * 0.6 + color[c] * 0.4, overlay[:, :, c])

    # Obtener las coordenadas del bounding box
    x1, y1, x2, y2 = map(int, box.cpu().numpy())
    confidence = score.cpu().numpy()

    # Dibujar el bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Poner la etiqueta de la clase y la confianza
    class_label = f"{unique_classes[int(cls_id)]} {confidence:.2f}"
    cv2.putText(image, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Combinar la imagen original con el overlay usando transparencia
image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)

# Mostrar la imagen con las máscaras, colores y etiquetas
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.axis("off")
plt.title("YOLOv8 Segmentación con Máscaras Translúcidas y Etiquetas")
plt.show()
