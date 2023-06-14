wrist_x = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * frame.shape[0]
thumbT_x = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * frame.shape[0]

class CustomLib:
    def riconoscimentoMano(self):
        if thumbT_x > wrist_x:
            return True
        else:
            return False