import cv2
import mediapipe as mp
import numpy as np

# Inizializza il modulo FaceMesh di Mediapipe
mp_face_mesh = mp.solutions.face_mesh

# Inizializza la webcam
cap = cv2.VideoCapture(0)

# Imposta il rilevatore di landmark del viso di Mediapipe
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)

orizontal_flip = True
vertical_flip = False
active_blur = True

while True:
    # Leggi il frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converte l'immagine da BGR a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva i landmark del viso nell'immagine
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Prendi le coordinate dei landmark del viso
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmarks.append((x, y))

            # Calcola il rettangolo di delimitazione intorno al volto
            x, y, w, h = cv2.boundingRect(np.array(landmarks))

            if active_blur:
                # Applica l'effetto di sfocatura all'area del volto
                blurred_face = frame[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(blurred_face, (99, 99), 0)

                # Sostituisci l'area del volto con il volto sfocato
                frame[y:y+h, x:x+w] = blurred_face

            # Disegna i landmark del viso sull'immagine
            for landmark in landmarks:
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)

    if orizontal_flip:
        frame = cv2.flip(frame, 1)
    if vertical_flip:
        frame = cv2.flip(frame, 0)

    # Mostra l'immagine con i landmark del viso e il volto sfocato
    cv2.imshow('Face Landmark Detection with Blur', frame)

    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
