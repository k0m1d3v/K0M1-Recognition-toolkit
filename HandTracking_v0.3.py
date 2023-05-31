# hand tracking script

import cv2
import mediapipe as mp

print(cv2.__version__)

# Variabile booleana per il flip del frame orizzontalmente (consiglio di lasciare orizontal_flip = True e vertical_flip = False, come impostato di default)
orizontal_flip = True
vertical_flip = False

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

    # Converte l'immagine da BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # è opzionale ma facendo dei test il programma risponde meglio agli input e risulta più veloce

    # Rileva le mani nell'immagine
    results = hands.process(image)

    # Disegna i landmark delle mani sul frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Qui mi sono ricollegato alla variabile booleana per il flip del frame verticale o orizzontale
    if orizontal_flip:
        frame = cv2.flip(frame, 1)
    if vertical_flip:
        frame = cv2.flip(frame, 0)

    # Mostra il frame con i landmark delle mani
    cv2.imshow('Hand Tracking', frame)

    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
