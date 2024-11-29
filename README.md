# Smart Recycling CNC

Este proyecto implementa un sistema básico de reciclaje inteligente utilizando **Python** y un modelo de **segmentación**. El objetivo es inspirar a otros en sus proyectos relacionados con la visión por computadora y la automatización.

## Descripción del Proyecto

El sistema está diseñado para detectar **botellas** y **tapas** en tiempo real utilizando un modelo **YOLOv8** de segmentación. Al detectar una botella quieta en una zona específica, el sistema inicia un proceso de reciclaje enviando instrucciones G-code a una máquina **CNC**.

### Características:

- **Detección en tiempo real** de botellas y tapas utilizando un modelo YOLOv8.
- **Integración con máquina CNC** para automatizar el proceso de reciclaje.
- **Interfaz gráfica** desarrollada con Tkinter para visualizar las detecciones y controlar el sistema.
- **Cargado de puntos** para definir una Región de Interés (ROI) y limitar la detección a un área específica redimensionada a 480x480 píxeles.
- **Diseñado para correr en CPU**, facilitando su ejecución en equipos sin GPU.

## Video de Funcionamiento

Puedes ver un breve video del funcionamiento del sistema en el siguiente enlace:

[Video de Funcionamiento](https://drive.google.com/file/d/1D1MtQpHgZ8GRgle3CkPdneBa_89ZZzaj/view?usp=sharing)

## Dataset Utilizado

El modelo fue entrenado utilizando el siguiente dataset de Roboflow:

[Dataset de Botellas y Tapas](https://universe.roboflow.com/labs-odu3x/plastic-bottles-4bt1x)

## Contenido del Repositorio

- **Código principal**: `SmartRecyclingCNC.py` es el script principal que ejecuta el sistema de detección y control de la CNC.
- **Código básico para accionar la CNC**:`gcode`  comandos G-code para controlar la máquina CNC para ejecutar acciones específicas.
- **Notebook de Colab**: `YOLOTrain.ipynb` para entrenar el modelo en la nube utilizando Google Colab.
- **Capturador de Puntos**: `DetectZone.py` Herramienta para definir una Región de Interés (ROI) y limitar la detección a un área específica. Los puntos se guardan en un archivo `points.pkl`.
- **Algoritmo para captura imagenes**: `CaptureImage.py` Carga la información del ROI desde el archivo `points.pkl` para capturar imágenes en el formato especificado.
- **Algoritmo para probar el modelo**: `InferenceProbe.py` y `InferenceProbeLive.py` Para probar el modelo.
- **Modelo segmentado YOLO version 8 nano**: `YOLO8nSegPlaticB.pt` tiene 2 clases Bottles y Lids.



## Uso

1. **Conexión de la CNC:**

   - Asegúrate de que la máquina CNC esté conectada al puerto especificado (por defecto `COM20`) y que esté configurada correctamente.

2. **Configuración de la Región de Interés (ROI):**

   - Ejecuta el **capturador de puntos** para definir el ROI y generar el archivo `points.pkl`.

3. **Entrenamiento del Modelo (opcional):**

   - Utiliza el **notebook de Colab** incluido para entrenar el modelo en la nube con el dataset proporcionado.

4. **Ejecución del Sistema:**

   - Ejecuta el script principal `SmartRecyclingCNC.py`

5. **Iniciar Análisis:**

   - Presiona el botón **"Analizar"** en la interfaz.
   - El sistema comenzará a detectar botellas y tapas en tiempo real.

6. **Proceso de Reciclaje:**

   - Al detectar una **botella quieta** en el ROI, el sistema mostrará el mensaje **"Iniciando reciclaje"**.
   - Se enviará el **G-code** a la CNC para iniciar el proceso de reciclaje.

7. **Notas Adicionales:**

   - Si después de un tiempo no se detecta una botella, el sistema mostrará **"Reciclaje no encontrado"** y podrás volver a intentar el análisis.

## Notas

- **Ejecución en CPU:**

  - El proyecto está diseñado para correr en CPU, por lo que no requiere una GPU dedicada.

- **Personalización:**

  - Puedes ajustar los parámetros en los scripts según tus necesidades, como el puerto de la CNC, el umbral de confianza, la tolerancia de movimiento, etc.

- **Formato de Archivos:**

  - El archivo de G-code debe llamarse `gcode.txt` y estar ubicado en la raíz del proyecto.

## Recursos Adicionales

- **Entrenamiento del Modelo:**

  - Utiliza el notebook de Colab incluido para entrenar el modelo con tus propios datos o ajustar el existente.

- **Dataset:**

  - El dataset utilizado se encuentra en Roboflow y contiene imágenes de botellas y tapas etiquetadas para segmentación.

---

Este proyecto es público y busca inspirar a otros desarrolladores y entusiastas en el área de la visión por computadora y la automatización.
