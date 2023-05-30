# Linux utilizza un sistema di gestione dell'audio diverso da windows di conseguenza è necessario usare un altra libreria

import cv2
import mediapipe as mp
import numpy as np
import math
import sounddevice as sd
import os

# Imposta la variabile di ambiente per indicare la posizione della libreria PortAudio
os.environ["PYAUDIO_PORTAUDIO_LIB"] = "/path/to/libportaudio.so"  # Sostituisci "/path/to/libportaudio.so" con il percorso reale

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inizializza il modulo Hands di MediaPipe
hands = mp_hands.Hands()

# Inizializza la webcam
cap = cv2.VideoCapture(0)

# Imposta i parametri per il controllo del volume
volume_range = [-65, 0]  # Intervallo di regolazione del volume in decibel (dB)
min_distance = 20  # Distanza minima tra le dita per il rilevamento del pizzico
max_volume = 200  # Volume massimo del sistema #Nota il volume è duplicato per una questione di sensibilità
min_volume = 0  # Volume minimo del sistema

# Funzione per regolare il volume del sistema
def set_system_volume(volume):
    # Calcola il volume in scala lineare nell'intervallo [0, 1]
    volume_level = (volume_range[1] - volume_range[0]) * (volume / 100.0) + volume_range[0]
    # Gestisci i limiti del volume
    volume_level = max(volume_range[0], min(volume_range[1], volume_level))
    # Imposta il volume del sistema utilizzando sounddevice
    sd._set_output_volume(volume_level)

while True:
    # Leggi il frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converte l'immagine da BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva le mani nell'immagine
    results = hands.process(image)

    # Disegna i landmark delle mani sul frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Rileva le coordinate del pollice e dell'indice
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]

            # Calcola la distanza euclidea tra il pollice e l'indice
            distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            # Regola il volume in base alla distanza tra il pollice e l'indice
            if distance < min_distance:
                distance = min_distance
            elif distance > frame.shape[1] // 2:
                distance = frame.shape[1] // 2

            # Mappa la distanza al range del volume
            volume = np.interp(distance, [min_distance, frame.shape[1] // 2], [min_volume, max_volume])

            # Regola il volume del sistema
            set_system_volume(volume)

            # Disegna un indicatore di volume sul frame
            cv2.rectangle(frame, (20, 400), (20 + int(volume * 3), 440), (0, 255, 0), -1)

            # Disegna una linea che collega il pollice e l'indice
            cv2.line(frame, (int(thumb_x), int(thumb_y)), (int(index_x), int(index_y)), (0, 0, 255), 2)

    # Mostra il frame con i landmark delle mani, l'indicatore di volume e la linea che collega il pollice e l'indice
    cv2.imshow('Hand Tracking', frame)

    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ripristina il volume al valore iniziale
set_system_volume(min_volume)

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
