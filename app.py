"""
╔══════════════════════════════════════════════════════════════════╗
║           DEEPFAKE DETECTION SYSTEM — app.py                    ║
║   Run using: streamlit run app.py                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────
import io
import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Safe OpenCV import
try:
    import cv2
except Exception:
    cv2 = None

# Safe matplotlib import
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Streamlit Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔬",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
WEIGHTS_DIR = ROOT_DIR / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

HISTORY_FILE = ROOT_DIR / "history.json"

IMAGE_SIZE = 224
FAKE_THRESHOLD = 0.50

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

CLASS_COLORS = {
    "REAL": "#00e676",
    "FAKE": "#ff3b3b",
    "ERROR": "#ffab00",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background-color: #0f1117;
    color: white;
}

h1,h2,h3 {
    color: white;
}

.stButton>button {
    border-radius: 10px;
    border: 1px solid #ff3b3b;
}

.metric-box {
    padding: 1rem;
    border-radius: 10px;
    background: #181b22;
    border: 1px solid #2a2f3d;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Transform
# ─────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ─────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────
class DeepfakeClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        try:
            import timm

            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=True,
                num_classes=0,
                global_pool="avg",
            )

            in_features = self.backbone.num_features

        except Exception:

            from torchvision.models import efficientnet_b0

            model = efficientnet_b0(pretrained=True)

            in_features = model.classifier[1].in_features

            model.classifier = nn.Identity()

            self.backbone = model

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)

    def predict(self, x):

        self.eval()

        with torch.no_grad():

            out = self(x)

            probs = torch.softmax(out, dim=1)

        return probs

# ─────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():

    model = DeepfakeClassifier().to(DEVICE)

    model_path = WEIGHTS_DIR / "model_best.pth"

    if model_path.exists():

        try:
            model.load_state_dict(
                torch.load(model_path, map_location=DEVICE)
            )

            st.success("Custom weights loaded")

        except Exception as e:

            st.warning(f"Could not load weights: {e}")

    else:
        st.info("Running in demo mode")

    model.eval()

    return model

# ─────────────────────────────────────────────────────────────
# Face Detection
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_mtcnn():

    try:
        from facenet_pytorch import MTCNN

        detector = MTCNN(
            image_size=IMAGE_SIZE,
            keep_all=False,
            device=DEVICE,
        )

        return detector

    except Exception:
        return None

# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────
def detect_face(image: Image.Image, detector):

    if detector is None:
        return image

    try:

        face = detector(image)

        if face is None:
            return image

        face = face.permute(1, 2, 0).numpy()

        face = (face * 255).astype(np.uint8)

        return Image.fromarray(face)

    except Exception:
        return image

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
def predict_image(image: Image.Image):

    model = load_model()

    detector = load_mtcnn()

    start = time.time()

    face = detect_face(image, detector)

    tensor = transform(face).unsqueeze(0).to(DEVICE)

    probs = model.predict(tensor)

    real_prob = float(probs[0][0])
    fake_prob = float(probs[0][1])

    verdict = "FAKE" if fake_prob >= FAKE_THRESHOLD else "REAL"

    confidence = fake_prob if verdict == "FAKE" else real_prob

    latency = time.time() - start

    return {
        "verdict": verdict,
        "confidence": confidence,
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "latency": latency,
        "face": face,
    }

# ─────────────────────────────────────────────────────────────
# Video Prediction
# ─────────────────────────────────────────────────────────────
def predict_video(video_path):

    if cv2 is None:
        return {
            "error": "OpenCV not installed"
        }

    cap = cv2.VideoCapture(video_path)

    frames = []

    count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if count % 15 == 0:

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pil = Image.fromarray(rgb)

            result = predict_image(pil)

            frames.append(result["fake_prob"])

        count += 1

        if len(frames) >= 20:
            break

    cap.release()

    if len(frames) == 0:

        return {
            "error": "No frames processed"
        }

    mean_fake = float(np.mean(frames))

    verdict = "FAKE" if mean_fake >= FAKE_THRESHOLD else "REAL"

    confidence = mean_fake if verdict == "FAKE" else 1 - mean_fake

    return {
        "verdict": verdict,
        "confidence": confidence,
        "fake_prob": mean_fake,
    }

# ─────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────
def load_history():

    if HISTORY_FILE.exists():

        try:
            return json.loads(HISTORY_FILE.read_text())

        except Exception:
            return []

    return []

def save_history(data):

    history = load_history()

    history.insert(0, data)

    history = history[:100]

    HISTORY_FILE.write_text(
        json.dumps(history, indent=2)
    )

# ─────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────
st.title("🔬 DeepFake Detection System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Analyze Image",
        "Analyze Video",
        "History",
        "About"
    ]
)

# ─────────────────────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────────────────────
if page == "Home":

    st.header("AI Powered DeepFake Detection")

    st.write("""
Upload an image or video to detect whether it is REAL or FAKE.

### Features
- EfficientNet Deep Learning Model
- Face Detection using MTCNN
- Streamlit UI
- Video Frame Analysis
- GPU Support
""")

# ─────────────────────────────────────────────────────────────
# IMAGE
# ─────────────────────────────────────────────────────────────
elif page == "Analyze Image":

    uploaded = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:

        image = Image.open(uploaded).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image")

        with st.spinner("Analyzing..."):

            result = predict_image(image)

        with col2:

            color = CLASS_COLORS[result["verdict"]]

            st.markdown(f"""
            <div class="metric-box">
            <h2 style="color:{color}">
            {result['verdict']}
            </h2>

            <p>Confidence: {result['confidence']:.2%}</p>
            <p>Latency: {result['latency']:.2f}s</p>
            </div>
            """, unsafe_allow_html=True)

            st.image(
                result["face"],
                caption="Detected Face"
            )

        save_history({
            "type": "image",
            "verdict": result["verdict"],
            "confidence": result["confidence"],
            "time": str(datetime.now())
        })

# ─────────────────────────────────────────────────────────────
# VIDEO
# ─────────────────────────────────────────────────────────────
elif page == "Analyze Video":

    uploaded = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        ) as tmp:

            tmp.write(uploaded.read())

            temp_path = tmp.name

        st.video(temp_path)

        with st.spinner("Analyzing video..."):

            result = predict_video(temp_path)

        if "error" in result:

            st.error(result["error"])

        else:

            color = CLASS_COLORS[result["verdict"]]

            st.markdown(f"""
            <div class="metric-box">
            <h2 style="color:{color}">
            {result['verdict']}
            </h2>

            <p>Confidence: {result['confidence']:.2%}</p>
            </div>
            """, unsafe_allow_html=True)

            save_history({
                "type": "video",
                "verdict": result["verdict"],
                "confidence": result["confidence"],
                "time": str(datetime.now())
            })

# ─────────────────────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────────────────────
elif page == "History":

    st.header("Detection History")

    history = load_history()

    if len(history) == 0:

        st.info("No history available")

    else:

        for item in history:

            color = CLASS_COLORS.get(
                item["verdict"],
                "#ffffff"
            )

            st.markdown(f"""
            <div class="metric-box">
            <h3 style="color:{color}">
            {item['verdict']}
            </h3>

            <p>Type: {item['type']}</p>
            <p>Confidence: {item['confidence']:.2%}</p>
            <p>Time: {item['time']}</p>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ABOUT
# ─────────────────────────────────────────────────────────────
elif page == "About":

    st.header("About")

    st.write("""
### DeepFake Detection System

This project detects manipulated AI-generated media using Deep Learning.

### Technologies Used
- Python
- Streamlit
- PyTorch
- EfficientNet
- OpenCV
- MTCNN

### Developer Features
- Image Analysis
- Video Analysis
- Face Detection
- Deep Learning Classification

### Note
This is a demo/research system and may not achieve 100% accuracy.
""")
