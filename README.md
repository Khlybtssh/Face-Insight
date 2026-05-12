# 🎭 Face Recognition + Expression Detection

A real-time desktop application that performs **face identity recognition** and **facial expression detection** simultaneously using deep learning. Built with a dual-model architecture — MobileNetV2 for identity and MobileNetV2 + CBAM attention for emotion — delivered through an interactive Tkinter GUI with live webcam feed.

---

## ✨ Features

- **Real-Time Face Detection** — Haar Cascade-based detection running on every webcam frame
- **Identity Recognition** — Register users via webcam, train a MobileNetV2 classifier, and recognize faces in real time
- **Expression Detection** — Detects 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using a MobileNetV2 + CBAM model trained on FER2013
- **Combined Predictions** — Overlays both identity and emotion on each detected face with color-coded bounding boxes
- **Interactive GUI** — Clean, dark-themed Tkinter interface with one-click registration, training, and recognition controls

---

## 🏗️ Architecture

### Identity Recognition Pipeline

```
Webcam Frame → Haar Cascade (face detection) → Crop + Resize (224×224)
  → MobileNetV2 (ImageNet pretrained, fine-tuned) → Softmax → Identity + Confidence
```

- **Backbone:** MobileNetV2 with frozen base layers + custom dense head (1024 → N classes)
- **Framework:** TensorFlow / Keras
- **Training:** On-the-fly from webcam-collected face images with data augmentation

### Expression Recognition Pipeline

```
Face Crop → Grayscale → Resize (224×224) → Normalize (ImageNet stats)
  → MobileNetV2 backbone (partially frozen) → CBAM Attention → Classifier → 7 Emotions
```

- **Backbone:** MobileNetV2 (first 14/19 blocks frozen for stable fine-tuning)
- **Attention:** [CBAM](https://arxiv.org/abs/1807.06521) (Convolutional Block Attention Module) — applies channel attention + spatial attention to focus on discriminative facial regions
- **Framework:** PyTorch
- **Dataset:** [FER2013](https://www.kaggle.com/datasets/msambare/fer2013) (~28K training images, 7 emotion classes)

### CBAM Attention Module

```
Features (1280×7×7) → Channel Attention (squeeze-excite style) → Spatial Attention (7×7 conv) → Attended Features
```

| Component | Mechanism |
|---|---|
| **Channel Attention** | Global Avg Pool + Global Max Pool → shared MLP → sigmoid gating |
| **Spatial Attention** | Channel-wise Avg + Max Pool → 7×7 Conv → sigmoid gating |

---

## 📁 Project Structure

```
face-rego-main/
├── main.py                  # GUI application (Tkinter + OpenCV)
├── face_lib.py              # Core engine: detection, identity, expression, combined prediction
├── expression_model.py      # PyTorch model definition (MobileNetV2 + CBAM)
├── train_expression.py      # Training script for expression model (GPU required)
├── evaluate_expression.py   # Evaluation script with classification report & confusion matrix
├── download_fer2013.py      # FER2013 dataset downloader (Kaggle)
├── experiments.ipynb        # Notebook for quick experiments
├── requirements.txt         # Python dependencies
├── data/                    # Webcam-collected face images (per-user folders)
├── fer2013/                 # FER2013 dataset (train/test splits by emotion)
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── sad/
│   │   ├── surprise/
│   │   └── neutral/
│   └── test/
│       └── ...
└── models/                  # Saved model weights & class mappings
    ├── face_model.keras      # Trained identity model (TF/Keras)
    ├── classes.pkl           # Identity class name mapping
    ├── expression_model.pth  # Trained expression model (PyTorch)
    └── expression_classes.pkl# Expression class name mapping
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- **Webcam** (for the GUI application)
- **NVIDIA GPU + CUDA** (required for expression model training only — inference runs on CPU)

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/<your-username>/face-rego.git
cd face-rego
pip install -r requirements.txt
```

Then install PyTorch with CUDA support (for training):

```bash
# Visit https://pytorch.org/get-started/locally/ for your specific CUDA version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Download the FER2013 Dataset

```bash
python download_fer2013.py
```

This will attempt to download via `kagglehub` or the Kaggle API. If automatic download fails, follow the manual instructions printed by the script (requires a free [Kaggle](https://www.kaggle.com/) account).

### 3. Train the Expression Model

```bash
python train_expression.py
```

- Requires an NVIDIA GPU
- Trains MobileNetV2 + CBAM on FER2013 for up to 50 epochs (with early stopping)
- Saves the best model to `models/expression_model.pth`
- Estimated training time: **5–15 minutes** on a modern GPU

### 4. Launch the Application

```bash
python main.py
```

---

## 🎮 Usage

The GUI provides three main actions:

| Button | Action |
|---|---|
| **📷 Register User** | Prompts for a name, then collects 50 face images from the webcam |
| **🧠 Train Identity** | Trains the MobileNetV2 identity model on all registered users |
| **▶ Start Recognition** | Begins real-time identity + expression recognition |

### Workflow

1. **Register** — Click "Register User", enter a name, and face the camera while it captures 50 images
2. **Train** — Click "Train Identity" to fine-tune MobileNetV2 on your collected data
3. **Recognize** — Click "Start Recognition" to see real-time identity + emotion labels on every detected face

### Bounding Box Color Coding

| Emotion | Box Color |
|---|---|
| 😊 Happy | 🟢 Green |
| 😠 Angry | 🔴 Red |
| 😨 Fear | 🟣 Purple |
| 😢 Sad | 🔵 Blue |
| 😲 Surprise | 🟡 Yellow |
| 🤢 Disgust | 🟢 Dark Green |
| 😐 Neutral | ⚪ Gray |
| ❓ Unknown Identity | 🔴 Red |

---

## 📊 Evaluation

To evaluate the expression model on the FER2013 test set:

```bash
python evaluate_expression.py
```

This outputs:
- Per-class **precision, recall, F1-score**
- **Confusion matrix**
- **Overall test accuracy**

---

## ⚙️ Configuration

Key parameters that can be adjusted:

| Parameter | Location | Default | Description |
|---|---|---|---|
| `collect_limit` | `main.py` | 50 | Number of face images collected per registration |
| `EPOCHS` | `train_expression.py` | 50 | Max training epochs for expression model |
| `LEARNING_RATE` | `train_expression.py` | 0.0005 | Learning rate for expression training |
| `patience` | `train_expression.py` | 15 | Early stopping patience (epochs) |
| `epochs` | `face_lib.py` → `train()` | 10 | Training epochs for identity model |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Face Detection | OpenCV Haar Cascade |
| Identity Model | TensorFlow / Keras — MobileNetV2 |
| Expression Model | PyTorch — MobileNetV2 + CBAM |
| GUI | Tkinter + PIL |
| Dataset | FER2013 (Kaggle) |

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgements

- **FER2013 Dataset** — [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- **CBAM** — Woo et al., *"CBAM: Convolutional Block Attention Module"*, ECCV 2018 ([arXiv](https://arxiv.org/abs/1807.06521))
- **MobileNetV2** — Sandler et al., *"MobileNetV2: Inverted Residuals and Linear Bottlenecks"*, CVPR 2018
