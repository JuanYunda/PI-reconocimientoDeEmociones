# Importamos las librerias
from deepface import DeepFace
import cv2
import mediapipe as mp
import random
import time

# Declaramos la deteccion de rostros
detros = mp.solutions.face_detection
rostros = detros.FaceDetection(min_detection_confidence=0.8, model_selection=0)
# Dibujo
dibujorostro = mp.solutions.drawing_utils

# Realizamos VideoCaptura
cap = cv2.VideoCapture(0)

# Emociones disponibles
emociones_disponibles = ['angry','happy', 'sad', 'surprise']

# Cargar imágenes para cada emoción
imagenes_emociones = {
    'angry': cv2.imread('angry_image.png'),
    'happy': cv2.imread('happy_image.png'),
    'sad': cv2.imread('sad_image.png'),
    'surprise': cv2.imread('surprise_image.png'),
}

# Empezamos
while True:
    # Leemos los fotogramas
    ret, frame = cap.read()
    # Elegimos una emoción al azar
    emocion_elegida = random.choice(emociones_disponibles)
    print("Emoción elegida:", emocion_elegida)

    # Mostramos la imagen correspondiente a la emoción elegida
    img_emocion = imagenes_emociones.get(emocion_elegida, None)
    if img_emocion is not None:
        # Imprimimos las dimensiones de la imagen
        print("Dimensiones de la imagen original:", img_emocion.shape)

        # Verificamos si la imagen se cargó correctamente y tiene dimensiones válidas
        if img_emocion.shape[0] > 0 and img_emocion.shape[1] > 0:
            # Redimensionamos la imagen
            img_emocion = cv2.resize(img_emocion, (0, 0), None, 0.25, 0.25)
            ani, ali, c = img_emocion.shape

            # Superponemos la imagen en la esquina superior izquierda de frame
            frame[10:ani + 10, 10:ali + 10] = img_emocion
        else:
            print("Error: La imagen de la emoción no tiene dimensiones válidas.")

    # Correccion de color
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Esperamos a que la emoción sea expresada
    emocion_detectada = None
    while emocion_detectada != emocion_elegida:
        ret, frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resrostros = rostros.process(rgb)

        if resrostros.detections is not None:
            for rostro in resrostros.detections:
                box = rostro.location_data.relative_bounding_box
                xi, yi, w, h = int(box.xmin * frame.shape[1]), int(box.ymin * frame.shape[0]), int(box.width * frame.shape[1]), int(box.height * frame.shape[0])
                xf, yf = xi + w, yi + h

                # Dibujamos
                cv2.rectangle(frame, (xi, yi), (xf, yf), (255, 255, 0), 1)
                frame[10:ani + 10, 10:ali+10] = img_emocion

                # Analizamos la emoción
                info = DeepFace.analyze(rgb, actions=['emotion'], enforce_detection=False)
                emocion_detectada = info['dominant_emotion'] if info and 'dominant_emotion' in info else None

                if emocion_detectada:
                    print("Emoción detectada:", emocion_detectada)

                # Verificamos si la emoción detectada es igual a la emoción elegida
                if emocion_detectada == emocion_elegida:
                    es_coincidente = True
                else:
                    es_coincidente = False
                
                print("Coincidencia de emoción:", es_coincidente)

                # Mostramos el resultado en la interfaz gráfica
                texto_resultado = f"Coincidencia: {es_coincidente}"
                cv2.putText(frame, texto_resultado, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow("Deteccion de Emocion", frame)
        cv2.waitKey(1)

    # Esperamos 2 segundos antes de elegir otra emoción
    time.sleep(2)

cv2.destroyAllWindows()
cap.release()
