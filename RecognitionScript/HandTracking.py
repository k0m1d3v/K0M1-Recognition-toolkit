
import cv2
import mediapipe as mp
import numpy as np
import math
import os
import serial

# Inizializza la comunicazione seriale con l'Arduino
# Sostituisci 'COM3' con la porta seriale corretta e 9600 con la velocità di comunicazione corretta
ser = serial.Serial('COM3', 9600)


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inizializza il modulo Hands di MediaPipe
hands = mp_hands.Hands()

# Inizializza la webcam
cap = cv2.VideoCapture(1)

font = cv2.FONT_HERSHEY_SIMPLEX
scala = 1
colore = (0, 255, 0)  # Colore del testo (in formato BGR)
spessore = 2
oldLettura=0
cont=0
precisione=3


while True:
    # Leggi il frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converte l'immagine da BGR a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva le mani nell'immagine
    results = hands.process(image)

    # Variabile booleana per il flip del frame orizzontalmente (consiglio di lasciare orizontal_flip = True e vertical_flip = False, come impostato di default)q


    # Disegna i landmark delle mani sul frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            #mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=5),
                                      mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2))

            # Rileva le coordinate del pollice e dell'indice
            thumb_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[1]
            thumb_y = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * frame.shape[0]
            index_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1]
            index_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]


            # Conta le dita alzate mano destra
            thumbT_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[0]
            thumbI_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * frame.shape[0]
            indexT_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0]
            indexD_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * frame.shape[0]
            middleT_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * frame.shape[0]
            middleD_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * frame.shape[0]
            ringT_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * frame.shape[0]
            ringD_y = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * frame.shape[0]
            pinkyT_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * frame.shape[0]
            pinkyD_y = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * frame.shape[0]

            # Inizializza un array per le coordinate y dei landmark delle mani
            landmark_y = []

            # Rileva le coordinate y dei landmark delle mani
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for landmark in hand_landmarks.landmark:
                        landmark_y.append(landmark.y * frame.shape[0])


            # Stampa le coordinate y dei landmark delle mani
            print(landmark_y)

            numero = 0
            freq = []

            if thumbT_x>thumbI_x:
                numero=numero+1
            if indexT_y<indexD_y:
             numero=numero+1
            if middleT_y<middleD_y:
                numero=numero+1
            if ringT_y<ringD_y:
                numero=numero+1
            if pinkyT_y<pinkyD_y:
                numero=numero+1

            if oldLettura==numero:
                cont=cont+1
            else:
                cont=0
                oldLettura = numero



            if cont>=precisione:
             frame_con_testo = cv2.putText(frame, str(numero),(10,150), font, scala, colore, spessore, cv2.LINE_AA)
             ser.write(str(numero).encode())




            # Calcola la distanza euclidea tra il pollice e l'indice
            distance = math.sqrt((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2)

            frame_con_testo1 = cv2.putText(frame, str("Py="+str(int(thumb_y))+" Iy= "+str(int(index_y))+" mY="+str(int(((index_y)+(thumb_y))/2))),(10, 50), font, scala, colore,spessore, cv2.LINE_AA)
            frame_con_testo = cv2.putText(frame, str(distance-14), (int(thumb_x), int(((index_y)+(thumb_y))/2)), font, scala, colore, spessore,cv2.LINE_AA)
            # Disegna una linea che collega il pollice e l'indice
            cv2.line(frame, (int(thumb_x), int(thumb_y)), (int(index_x), int(index_y)), (0, 0, 255), 2)

    # Mostra il frame con i landmark delle mani, l'indicatore di volume e la linea che collega il pollice e l'indice
    cv2.imshow('Hand Tracking', frame)

    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
