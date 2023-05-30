# hand tracking

import cv2
import mediapipe as mp

print(cv2.__version__)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inizializza il modulo Hands di MediaPipe
hands = mp_hands.Hands()

# Inizializza la webcam
cap = cv2.VideoCapture(0)

while True:
    # Leggi il frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flippa il frame orizzontalmente
    frame = cv2.flip(frame, 1)

    # Converte l'immagine da BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # è opzionale ma facendo dei test il programma risponde meglio agli input e risulta più veloce

    # Rileva le mani nell'immagine
    results = hands.process(image)

    # Disegna i landmark delle mani sul frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Mostra il frame con i landmark delle mani
    cv2.imshow('Hand Tracking', frame)

    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
