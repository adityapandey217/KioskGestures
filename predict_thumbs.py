import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import joblib

# Load model and scaler
model = keras.models.load_model('thumbs_mlp_model.keras')
scaler = joblib.load('scaler.save')

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(hand_landmarks, handedness):
    keypoints = []
    for lm in hand_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z])
    # Add handedness: 1 if Right, 0 if Left
    keypoints.append(1 if handedness.classification[0].label == "Right" else 0)
    return keypoints

def predict_thumb(keypoints):
    x = np.array(keypoints).reshape(1, -1)
    x_scaled = scaler.transform(x)
    prob = model.predict(x_scaled)[0][0]
    return "Thumbs Up" if prob >= 0.5 else "Thumbs Down"

# Start webcam capture
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

        # Flip and convert color
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract keypoints and predict
                keypoints = extract_keypoints(hand_landmarks, handedness)
                prediction = predict_thumb(keypoints)

                # Show prediction on frame
                cv2.putText(frame, prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        else:
            cv2.putText(frame, "Show your hand", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

        cv2.imshow('Thumbs Up/Down Predictor', frame)

        if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
            break

cap.release()
cv2.destroyAllWindows()
