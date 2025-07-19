import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import joblib

# Load model and scaler
model = keras.models.load_model('thumbs_mlp_model.keras')
scaler = joblib.load('scaler.save')

# Label map (update this if you change class IDs)
label_map = {
    0: "Thumbs Down",
    1: "Thumbs Up",
    2: "Other"
}

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(hand_landmarks, handedness):
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
    keypoints.append(1 if handedness.classification[0].label == "Right" else 0)
    return keypoints

def predict_gesture(keypoints):
    x = np.array(keypoints).reshape(1, -1)
    x_scaled = scaler.transform(x)
    probs = model.predict(x_scaled)[0]
    class_id = np.argmax(probs)
    confidence = probs[class_id]
    return label_map[class_id], confidence

# Start webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                keypoints = extract_keypoints(hand_landmarks, handedness)
                prediction, confidence = predict_gesture(keypoints)

                # Display prediction
                display_text = f"{prediction} ({confidence*100:.1f}%)"
                cv2.putText(frame, display_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "Show your hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        cv2.imshow('Gesture Predictor', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
