# 👍👎 Thumbs Up/Down Gesture Classifier

This project uses [MediaPipe](https://google.github.io/mediapipe/) to extract hand keypoints and a custom-trained machine learning model (TensorFlow MLP) to classify hand gestures as **Thumbs Up**, **Thumbs Down**, or **Other gestures** in real-time.

---

## 📌 Features

- **Real-time webcam-based gesture recognition**
- **Custom dataset collection** with over 12,800 samples
- **High accuracy**: Achieves >98% accuracy on validation set
- **Multi-class classification**: Thumbs Up, Thumbs Down, and Other gestures
- **Extensible architecture** for adding new gesture types
- **Complete pipeline**: Data collection → Training → Real-time prediction
- **Standardized preprocessing** with saved scaler for consistent predictions

---

## 🧠 Model Architecture

- **Model Type:** Multi-layer Perceptron (MLP) using TensorFlow/Keras
- **Input Features:** 64 total features
  - 63 hand keypoints (21 landmarks × 3 coordinates: x, y, z)
  - 1 handedness flag (1 for Right hand, 0 for Left hand)
- **Architecture:**
  - Dense layer: 128 neurons + ReLU + Dropout (0.3)
  - Dense layer: 64 neurons + ReLU + Dropout (0.3)
  - Output layer: 3 neurons + Softmax (for 3 classes)
- **Preprocessing:** StandardScaler normalization
- **Training:** 30 epochs, batch size 32, Adam optimizer

---

## 📊 Dataset

| Gesture Class | Label | Sample Count | Description |
|---------------|-------|--------------|-------------|
| Thumbs Down   | 0     | 4,614        | 👎 Downward thumb gesture |
| Thumbs Up     | 1     | 4,561        | 👍 Upward thumb gesture |
| Other         | 2     | 4540        | ✋ Random/neutral gestures |
| **Total**     | -     | **13,716**   | Complete dataset |

**Data Format:** CSV with 65 columns (64 features + 1 label)

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/adityapandey217/KioskGestures.git
cd KioskGestures

# Install dependencies
pip install mediapipe opencv-python tensorflow joblib numpy scikit-learn matplotlib pandas
```

### 2. Run Real-time Prediction (Pre-trained Model)

```bash
python predict_thumbs.py
```

- Show your hand to the webcam
- Make thumbs up 👍, thumbs down 👎, or other gestures ✋
- Press `ESC` to quit

### 3. Collect Your Own Data (Optional)

```bash
python data_collect.py
```

**Controls:**
- Press `u` → Collect Thumbs Up data (60 seconds)
- Press `d` → Collect Thumbs Down data (60 seconds)  
- Press `o` → Collect Other gesture data (60 seconds)
- Press `q` → Quit collection

### 4. Train Your Own Model (Optional)

Open and run the Jupyter notebook:
```bash
jupyter notebook gesture_detector_train.ipynb
```

Or train directly with the existing `data.csv` file.


---

## 🔧 Usage Details

### Data Collection

The `data_collect.py` script captures hand landmarks using MediaPipe and saves them to a CSV file:

- **Automatic landmark extraction:** 21 hand keypoints per frame
- **Handedness detection:** Left/Right hand classification
- **Continuous recording:** 60-second sessions per gesture class
- **CSV output:** Appends data to `data.csv` with proper headers

### Model Training

The `gesture_detector_train.ipynb` notebook includes:

- **Data loading and exploration:** Analysis of class distribution
- **Preprocessing:** StandardScaler normalization
- **Model definition:** 3-layer MLP with dropout regularization
- **Training:** 30 epochs with validation split
- **Evaluation:** Accuracy metrics and training curves
- **Model export:** Saves both model and scaler for deployment

### Real-time Prediction

The `predict_thumbs.py` script provides:

- **Live webcam feed:** Real-time hand detection
- **Gesture classification:** Instant prediction with confidence scores
- **Visual feedback:** Hand landmarks and prediction overlay
- **Performance optimized:** Smooth real-time operation