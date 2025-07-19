import cv2
import mediapipe as mp
import csv
import time

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Output file
CSV_PATH = "data.csv"

# Labels
LABELS = {
    "u": 1,  # thumbs up
    "d": 0   # thumbs down
}

# Prepare CSV (create header if doesn't exist)
try:
    with open(CSV_PATH, 'x', newline='') as f:
        writer = csv.writer(f)
        header = [f'{axis}{i}' for i in range(21) for axis in ('x', 'y', 'z')] + ['handedness', 'label']
        writer.writerow(header)
except FileExistsError:
    pass  # File already exists

cap = cv2.VideoCapture(0)

print("Press 'u' to collect THUMBS UP")
print("Press 'd' to collect THUMBS DOWN")
print("Each key will record continuously for 60 seconds.")
print("Press 'q' to quit.")

collecting = False
label = None
start_time = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting.")
        break

    if not collecting and chr(key) in LABELS:
        label = LABELS[chr(key)]
        start_time = time.time()
        collecting = True
        print(f"\n▶️ Started collecting for label {label} ({'Thumbs Up' if label else 'Thumbs Down'})...")

    if collecting and time.time() - start_time >= 60:
        collecting = False
        print(f"✅ Done collecting 1 minute of data for label {label}.\n")

    if result.multi_hand_landmarks and result.multi_handedness:
        for hand_landmarks, hand_handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            keypoints = []
            for lm in hand_landmarks.landmark:
                keypoints.extend([lm.x, lm.y, lm.z])

            handedness = hand_handedness.classification[0].label
            handedness_flag = 1 if handedness == "Right" else 0

            if collecting:
                row = keypoints + [handedness_flag, label]
                with open(CSV_PATH, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

    # Display message while collecting
    if collecting:
        elapsed = int(time.time() - start_time)
        cv2.putText(image, f"Collecting {'Thumbs Up' if label else 'Thumbs Down'}: {elapsed}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Collector", image)

cap.release()
cv2.destroyAllWindows()
