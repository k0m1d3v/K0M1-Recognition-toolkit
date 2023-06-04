# hand tracking script

import cv2
import mediapipe as mp
import numpy as np

# Variabile booleana per il flip del frame orizzontalmente (consiglio di lasciare orizontal_flip = True e vertical_flip = False, come impostato di default)
orizontal_flip = True
vertical_flip = False
active_blur = True

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inizializza il modulo Hands di MediaPipe
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh

# Inizializza la webcam
cap = cv2.VideoCapture(0)

# Imposta il rilevatore di landmark del viso di Mediapipe
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    # Leggi il frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converte l'immagine da BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # è opzionale ma facendo dei test il programma risponde meglio agli input e risulta più veloce

    # Rileva le mani nell'immagine
    hands_results = hands.process(image)

    # Rileva i landmark del viso nell'immagine
    face_results = face_mesh.process(image)

    # Disegna i landmark delle mani sul frame
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
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

    # Qui mi sono ricollegato alla variabile booleana per il flip del frame verticale o orizzontale
    if orizontal_flip:
        frame = cv2.flip(frame, 1)
    if vertical_flip:
        frame = cv2.flip(frame, 0)

    # Mostra il frame con i landmark delle mani e della faccia
    cv2.imshow('Hand and face Tracking', frame)

    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
