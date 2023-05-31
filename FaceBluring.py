import cv2
import mediapipe as mp

# Inizializza il modulo FaceDetection di Mediapipe
mp_face_detection = mp.solutions.face_detection

# Inizializza la webcam
cap = cv2.VideoCapture(0)

# Imposta il rilevatore di volti di Mediapipe
face_detection = mp_face_detection.FaceDetection()

orizontal_flip = True
vertical_flip = False
show_circle = False

while True:
    # Leggi il frame dalla webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Converte l'immagine da BGR a RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Rileva i volti nell'immagine
    results = face_detection.process(image_rgb)

    # Se sono stati rilevati volti, applica l'effetto di sfocatura al contenuto dell'ovale intorno al volto
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            # Calcola le coordinate dell'ovale intorno al volto
            x = int(bbox.xmin * frame.shape[1])
            y = int(bbox.ymin * frame.shape[0])
            w = int(bbox.width * frame.shape[1])
            h = int(bbox.height * frame.shape[0])

            # Applica l'effetto di sfocatura al contenuto dell'ovale
            blurred_roi = frame[y:y + h, x:x + w].copy()
            blurred_roi = cv2.GaussianBlur(blurred_roi, (101, 101), 0)

            # Sostituisci il contenuto dell'ovale con l'immagine sfocata
            frame[y:y + h, x:x + w] = blurred_roi

            if show_circle:
                # Disegna l'ovale intorno al volto
                center_x = x + w // 2
                center_y = y + h // 2
                radius_x = w // 2
                radius_y = h // 2
                cv2.ellipse(frame, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, (0, 255, 0), 2)

    if orizontal_flip:
        frame = cv2.flip(frame, 1)
    if vertical_flip:
        frame = cv2.flip(frame, 0)

    # Mostra il frame con l'ovale tracciato e il contenuto sfocato
    cv2.imshow('Face Blur', frame)

    # Interrompi il ciclo se viene premuto il tasto 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Rilascia le risorse
cap.release()
cv2.destroyAllWindows()
