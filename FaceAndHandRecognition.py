import cv2
import mediapipe as mp
import numpy as np

# Settings
orizontal_flip = True
vertical_flip = False
active_blur = False
main_camera = 0         # 0 = webcam, 1 = external camera

# Variable declaration for a better understanding of the code
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Starting the mediapipe modules
hands = mp_hands.Hands()
mp_face_mesh = mp.solutions.face_mesh

# Start the webcam (reference to the settings)
cap = cv2.VideoCapture(main_camera)

# Set the mediapipe face landmark detector
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=4, min_detection_confidence=0.5, min_tracking_confidence=0.5)

while True:
    # Read the frame from the webcam and check if it's working
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the image from BGR to RGB (for better stats)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # è opzionale ma facendo dei test il programma risponde meglio agli input e risulta più veloce

    # Find the hands landmarks in the image
    hands_results = hands.process(image)

    # Find the face landmarks in the image
    face_results = face_mesh.process(image)

    # Draw the landmarks on the image
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # take the coordinates of the landmarks
            landmarks = []
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                landmarks.append((x, y))

            # Calculate the bounding rectangle around the face
            x, y, w, h = cv2.boundingRect(np.array(landmarks))

            if active_blur:
                # Blur the selected area
                blurred_face = frame[y:y+h, x:x+w]
                blurred_face = cv2.GaussianBlur(blurred_face, (99, 99), 0)

                # Switch the blurred face with the original one
                frame[y:y+h, x:x+w] = blurred_face

            # Draw the landmarks on the image
            for landmark in landmarks:
                cv2.circle(frame, landmark, 2, (0, 255, 0), -1)

    # Boolean variable for the flip of the frame (reference to the settings)
    if orizontal_flip:
        frame = cv2.flip(frame, 1)
    if vertical_flip:
        frame = cv2.flip(frame, 0)

    # Show the frame with the landmarks
    cv2.imshow('Hand and face Tracking', frame)

    # Listen to the 'q' key to kill the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# End the program and release the resources
cap.release()
cv2.destroyAllWindows()
