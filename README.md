# 👍👎 Thumbs Up/Down Gesture Classifier

This project uses [MediaPipe](https://google.github.io/mediapipe/) to extract hand keypoints and a custom-trained machine learning model (TensorFlow MLP) to classify hand gestures as **Thumbs Up** or **Thumbs Down** in real-time.

---

## 📌 Features

- Real-time webcam-based gesture recognition
- Uses 21 hand landmarks from MediaPipe + handedness info (Left/Right)
- Trained on a custom dataset with over 5,500 samples
- Achieves >99% accuracy on validation set
- Predicts "Thumbs Up" or "Thumbs Down" live using a trained model

---

## 🧠 Model Details

- **Model:** MLP (Multi-layer Perceptron) using TensorFlow/Keras
- **Input:** 63 hand keypoints (x, y, z) + 1 handedness flag = 64 features
- **Output:** Binary classification (Thumbs Up = 1, Down = 0)
- **Scaler:** `StandardScaler` from `sklearn`, saved via `joblib`

---

## 🧪 Dataset

- Collected using MediaPipe by capturing:
  - 21 hand landmarks per frame → 63 values
  - Handedness (1 for Right, 0 for Left)
- Labels:
  - `1` → Thumbs Up  
  - `0` → Thumbs Down
- CSV Format: 64 features + 1 label → 65 columns total

---

## 📸 Demo

> Coming soon! (Insert a GIF or video of your model in action here.)

---

## ⚙️ How to Run

### 1. Install dependencies

```bash
pip install mediapipe opencv-python tensorflow joblib numpy
