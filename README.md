# detector-de-bullying,inasistencia de estudiantes

este es un detector por camara que funciona con python , usando tensorflow , cv2 , funcional en phyton
import cv2
import numpy as np
from collections import deque
import tensorflow as tf


# ===== Cargar modelo TFLite =====
interpreter = tf.lite.Interpreter(model_path="model_pelea.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===== Cargar etiquetas =====
with open("labels.txt", "r") as f:
    labels = [line.strip().lower() for line in f.readlines()]

# ===== Configuración de cámara =====
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("⚠️ No se puede acceder a la cámara")
    exit()
print("✅ Cámara iniciada. Presiona ESC para salir.")

# ===== Variables para suavizado =====
frame_window = 10  # cantidad de frames para promediar
pred_queue = deque(maxlen=frame_window)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    # ===== Preprocesar imagen =====
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)  # normalizar entre 0 y 1

    # ===== Ejecutar predicción =====
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    # ===== Suavizado de predicción =====
    pred_queue.append(prediction)
    avg_pred = np.mean(pred_queue, axis=0)

    clase_id = np.argmax(avg_pred)
    clase_nombre = labels[clase_id]
    prob = avg_pred[clase_id]

    # ===== Colores por clase =====
    color = (0, 255, 0)  # normal
    if clase_nombre == "pelea":
        color = (0, 0, 255)
    elif clase_nombre == "bullying":
        color = (0, 165, 255)
    elif clase_nombre == "faltan estudiantes":
        color = (255, 0, 0)

    # ===== Mostrar resultado =====
    cv2.putText(frame, f"{clase_nombre.capitalize()} ({prob*100:.1f}%)",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Detector de Movimientos", frame)

    # ===== Salir con ESC =====
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
