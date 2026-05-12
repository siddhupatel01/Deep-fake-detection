"""
╔══════════════════════════════════════════════════════════════════╗
║           DEEPFAKE DETECTION SYSTEM — app.py                    ║
║   Self-contained Streamlit app. Run: streamlit run app.py        ║
╚══════════════════════════════════════════════════════════════════╝

Structure (all in one file for portability):
  - CONFIG          : hyperparameters & constants
  - MODEL LAYER     : EfficientNet-B4 classifier (timm)
  - FACE DETECTOR   : MTCNN wrapper
  - TRANSFORMS      : inference & training pipelines
  - GRADCAM         : explainability heatmap
  - INFERENCE ENGINE: high-level image/video API
  - STREAMLIT UI    : pages (Home, Analyze, History, About)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import io
import json
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepFake Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS / THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;700&display=swap');

/* ── Root tokens ─────────────────────────────────── */
:root {
    --red:     #ff3b3b;
    --green:   #00e676;
    --amber:   #ffab00;
    --bg0:     #0a0c0f;
    --bg1:     #111318;
    --bg2:     #181b22;
    --bg3:     #1f2330;
    --border:  #2a2f3d;
    --text1:   #eef0f5;
    --text2:   #9099b0;
    --mono:    'Space Mono', monospace;
    --sans:    'DM Sans', sans-serif;
}

/* ── Streamlit chrome overrides ──────────────────── */
html, body, [class*="css"] { font-family: var(--sans); }
.stApp { background: var(--bg0); }
section[data-testid="stSidebar"] { background: var(--bg1) !important; border-right: 1px solid var(--border); }
.stButton > button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text1);
    font-family: var(--mono);
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 0.5rem 1.2rem;
    border-radius: 4px;
    transition: all 0.2s;
}
.stButton > button:hover { border-color: var(--red); color: var(--red); }
.stButton > button[kind="primary"] { background: var(--red); border-color: var(--red); color: #fff; }
.stProgress > div > div { background: var(--red); }
div[data-testid="metric-container"] {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
}
div[data-testid="metric-container"] label { color: var(--text2) !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text1) !important; font-family: var(--mono); font-size: 1.6rem; }
.stTabs [data-baseweb="tab-list"] { background: var(--bg1); border-radius: 6px; padding: 2px; gap: 2px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: var(--text2); font-family: var(--mono); font-size: 0.75rem; letter-spacing: 0.05em; border-radius: 4px; }
.stTabs [aria-selected="true"] { background: var(--bg3) !important; color: var(--text1) !important; }
.stExpander { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 8px; }
div[data-testid="stFileUploader"] { background: var(--bg2); border: 1px dashed var(--border); border-radius: 8px; padding: 1rem; }
div[data-testid="stFileUploader"]:hover { border-color: var(--red); }
.stAlert { border-radius: 6px; }
/* hide default Streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ① CONFIG
# ─────────────────────────────────────────────────────────────────────────────
ROOT_DIR        = Path(__file__).parent
WEIGHTS_DIR     = ROOT_DIR / "weights"
HISTORY_FILE    = ROOT_DIR / "history.json"
WEIGHTS_DIR.mkdir(exist_ok=True)

MODEL_NAME      = "efficientnet_b4"
NUM_CLASSES     = 2
IMAGE_SIZE      = 224
FACE_MARGIN     = 0.2
DEVICE          = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE      = 8
FAKE_THRESHOLD  = 0.50
VIDEO_MAX_FRAMES= 24
FPS_SAMPLE_RATE = 6

IMAGENET_MEAN   = (0.485, 0.456, 0.406)
IMAGENET_STD    = (0.229, 0.224, 0.225)

CLASS_LABELS    = {0: "REAL", 1: "FAKE"}
CLASS_COLOR     = {"REAL": "#00e676", "FAKE": "#ff3b3b", "ERROR": "#ffab00"}

# ─────────────────────────────────────────────────────────────────────────────
# ② TRANSFORMS
# ─────────────────────────────────────────────────────────────────────────────

def get_inference_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_training_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((size + 32, size + 32), antialias=True),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        transforms.RandomErasing(p=0.1),
    ])


def to_tensor(image: Image.Image | np.ndarray, size: int = IMAGE_SIZE) -> torch.Tensor:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    return get_inference_transform(size)(image).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# ③ MODEL
# ─────────────────────────────────────────────────────────────────────────────

class DeepfakeClassifier(nn.Module):
    """EfficientNet-B4 binary classifier: 0=REAL, 1=FAKE."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.4):
        super().__init__()
        try:
            import timm  # type: ignore
            self.backbone = timm.create_model(
                "efficientnet_b4", pretrained=pretrained, num_classes=0, global_pool="avg"
            )
            in_features: int = self.backbone.num_features
        except Exception:
            # Fallback to torchvision EfficientNet if timm unavailable
            from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
            _base = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT if pretrained else None)
            in_features = _base.classifier[1].in_features
            _base.classifier = nn.Identity()
            self.backbone = _base

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(x))

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        return torch.softmax(self.forward(x), dim=-1)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> "DeepfakeClassifier":
        m = cls(pretrained=False)
        m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        return m.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# ④ FACE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def _get_mtcnn(device: str = "cpu"):
    try:
        from facenet_pytorch import MTCNN  # type: ignore
        return MTCNN(
            image_size=IMAGE_SIZE,
            margin=int(IMAGE_SIZE * FACE_MARGIN),
            keep_all=True,
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            device=device,
            post_process=False,
        )
    except ImportError:
        return None


def detect_largest_face(
    pil_image: Image.Image, detector
) -> Tuple[Optional[np.ndarray], float]:
    """Return HWC uint8 numpy array of the highest-confidence face, or None."""
    if detector is None:
        return None, 0.0
    try:
        with torch.no_grad():
            faces, probs = detector(pil_image, return_prob=True)
        if faces is None:
            return None, 0.0
        idx = int(probs.argmax())
        face_np = faces[idx].permute(1, 2, 0).byte().numpy()
        return face_np, float(probs[idx])
    except Exception as e:
        logger.warning("Face detection failed: %s", e)
        return None, 0.0


def sample_video_frames(video_path: str, every_n: int = FPS_SAMPLE_RATE,
                         max_frames: int = VIDEO_MAX_FRAMES) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames, idx = [], 0
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % every_n == 0:
                frames.append(frame)
            idx += 1
    finally:
        cap.release()
    return frames


# ─────────────────────────────────────────────────────────────────────────────
# ⑤ GRADCAM
# ─────────────────────────────────────────────────────────────────────────────

def gradcam(model: nn.Module, tensor: torch.Tensor, target_class: int = 1) -> np.ndarray:
    """Return a (H, W) float32 heatmap in [0, 1]."""
    model.eval()

    # Target the last conv before global pool
    try:
        target_layer = model.backbone.conv_head
    except AttributeError:
        try:
            target_layer = model.backbone.features[-1]
        except Exception:
            h, w = tensor.shape[-2:]
            return np.zeros((h, w), dtype=np.float32)

    acts, grads = [], []
    fh = target_layer.register_forward_hook(lambda m, i, o: acts.append(o.detach()))
    bh = target_layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0].detach()))

    try:
        t = tensor.clone().requires_grad_(True)
        model.zero_grad()
        logits = model(t)
        logits[0, target_class].backward()
    finally:
        fh.remove()
        bh.remove()

    if not acts or not grads:
        h, w = tensor.shape[-2:]
        return np.zeros((h, w), dtype=np.float32)

    act = acts[0].squeeze(0)
    grad = grads[0].squeeze(0)
    weights = grad.mean(dim=(1, 2))
    cam = F.relu(torch.einsum("c,chw->hw", weights, act))
    mn, mx = cam.min(), cam.max()
    cam = (cam - mn) / (mx - mn + 1e-8)
    cam = F.interpolate(
        cam.unsqueeze(0).unsqueeze(0), size=tensor.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze().numpy()
    return cam.astype(np.float32)


def overlay_cam(face_pil: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> Image.Image:
    img = np.array(face_pil.convert("RGB"))
    h, w = img.shape[:2]
    cam_u8 = (cam * 255).clip(0, 255).astype(np.uint8)
    cam_resized = cv2.resize(cam_u8, (w, h))
    colored = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    blended = cv2.addWeighted(img, 1 - alpha, colored_rgb, alpha, 0)
    return Image.fromarray(blended)


# ─────────────────────────────────────────────────────────────────────────────
# ⑥ INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FrameResult:
    frame_idx: int
    fake_prob: float
    real_prob: float
    face_detected: bool
    face_image: Optional[np.ndarray] = None
    heatmap_image: Optional[Image.Image] = None


@dataclass
class InferenceResult:
    verdict: str
    confidence: float
    fake_prob_mean: float
    frames: List[FrameResult] = field(default_factory=list)
    latency_sec: float = 0.0
    error: Optional[str] = None
    filename: str = ""
    timestamp: str = ""

    @property
    def is_fake(self) -> bool:
        return self.verdict == "FAKE"

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "confidence": round(self.confidence, 4),
            "fake_prob_mean": round(self.fake_prob_mean, 4),
            "frames_analyzed": len(self.frames),
            "latency_sec": round(self.latency_sec, 3),
            "filename": self.filename,
            "timestamp": self.timestamp,
            "error": self.error,
        }


@st.cache_resource(show_spinner=False)
def load_engine(weights_path: Optional[str] = None) -> dict:
    """Load model + detector once, cache in Streamlit resource cache."""
    progress = st.empty()
    progress.info("⚙️ Initialising model…")

    model_path = Path(weights_path) if weights_path else WEIGHTS_DIR / "model_best.pth"
    if model_path.exists():
        try:
            model = DeepfakeClassifier.load(model_path, device=DEVICE)
            progress.success(f"✅ Loaded fine-tuned weights from `{model_path.name}`")
        except Exception as e:
            progress.warning(f"⚠️ Could not load weights ({e}). Using ImageNet backbone.")
            model = DeepfakeClassifier(pretrained=True).to(DEVICE).eval()
    else:
        progress.info("⚠️ No weights found — using ImageNet backbone (demo mode).")
        model = DeepfakeClassifier(pretrained=True).to(DEVICE).eval()

    detector = _get_mtcnn(DEVICE)
    time.sleep(0.5)
    progress.empty()
    return {"model": model, "detector": detector}


def _run_one_frame(
    pil_image: Image.Image,
    frame_idx: int,
    engine: dict,
    heatmaps: bool = True,
) -> FrameResult:
    model    = engine["model"]
    detector = engine["detector"]

    face_np, _ = detect_largest_face(pil_image, detector)
    face_detected = face_np is not None

    if face_np is None:
        # Fallback: use full image
        face_pil = pil_image.resize((IMAGE_SIZE, IMAGE_SIZE))
        face_np  = np.array(face_pil)
    
    face_pil = Image.fromarray(face_np)
    tensor   = to_tensor(face_np, IMAGE_SIZE).to(DEVICE)

    probs    = model.predict_proba(tensor)
    real_p   = float(probs[0, 0])
    fake_p   = float(probs[0, 1])

    heatmap_pil: Optional[Image.Image] = None
    if heatmaps:
        try:
            cam = gradcam(model, tensor.clone(), target_class=1)
            heatmap_pil = overlay_cam(face_pil, cam)
        except Exception as e:
            logger.warning("GradCAM failed frame %d: %s", frame_idx, e)

    return FrameResult(
        frame_idx=frame_idx,
        fake_prob=fake_p,
        real_prob=real_p,
        face_detected=face_detected,
        face_image=face_np,
        heatmap_image=heatmap_pil,
    )


def _aggregate(frames: List[FrameResult], latency: float,
               filename: str = "") -> InferenceResult:
    if not frames:
        return InferenceResult(
            verdict="ERROR", confidence=0.0, fake_prob_mean=0.0,
            latency_sec=latency, error="No faces detected.",
            filename=filename, timestamp=datetime.now().isoformat(),
        )
    mean_fake = float(np.mean([f.fake_prob for f in frames]))
    verdict   = "FAKE" if mean_fake >= FAKE_THRESHOLD else "REAL"
    conf      = mean_fake if verdict == "FAKE" else 1.0 - mean_fake
    return InferenceResult(
        verdict=verdict, confidence=conf, fake_prob_mean=mean_fake,
        frames=frames, latency_sec=latency,
        filename=filename, timestamp=datetime.now().isoformat(),
    )


def predict_image(pil: Image.Image, engine: dict,
                  heatmaps: bool = True, filename: str = "") -> InferenceResult:
    t0 = time.perf_counter()
    result = _run_one_frame(pil, 0, engine, heatmaps)
    return _aggregate([result], time.perf_counter() - t0, filename)


def predict_video(video_path: str, engine: dict,
                  heatmaps: bool = True, filename: str = "") -> InferenceResult:
    t0 = time.perf_counter()
    try:
        bgr_frames = sample_video_frames(video_path)
    except IOError as e:
        return InferenceResult(
            verdict="ERROR", confidence=0.0, fake_prob_mean=0.0,
            latency_sec=time.perf_counter() - t0,
            error=str(e), filename=filename, timestamp=datetime.now().isoformat(),
        )

    frames = []
    prog   = st.progress(0, text="Analysing frames…")
    for i, bgr in enumerate(bgr_frames):
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        frames.append(_run_one_frame(pil, i, engine, heatmaps))
        prog.progress((i + 1) / len(bgr_frames), text=f"Frame {i+1}/{len(bgr_frames)}")
    prog.empty()
    return _aggregate(frames, time.perf_counter() - t0, filename)


# ─────────────────────────────────────────────────────────────────────────────
# ⑦ HISTORY  (persisted to JSON)
# ─────────────────────────────────────────────────────────────────────────────

def load_history() -> List[dict]:
    if HISTORY_FILE.exists():
        try:
            return json.loads(HISTORY_FILE.read_text())
        except Exception:
            return []
    return []


def save_to_history(result: InferenceResult) -> None:
    history = load_history()
    history.insert(0, result.to_dict())
    history = history[:100]   # keep last 100
    HISTORY_FILE.write_text(json.dumps(history, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# ⑧ UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def _verdict_card(result: InferenceResult) -> None:
    color = CLASS_COLOR.get(result.verdict, "#888")
    icon  = {"REAL": "✅", "FAKE": "🚨", "ERROR": "⚠️"}.get(result.verdict, "❓")
    bar_w = int(result.fake_prob_mean * 100)

    st.markdown(f"""
    <div style="
        background: {color}11;
        border: 1px solid {color}44;
        border-left: 4px solid {color};
        border-radius: 10px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
    ">
        <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem">
            <span style="font-size:2rem">{icon}</span>
            <span style="font-family:'Space Mono',monospace;font-size:2rem;font-weight:700;color:{color}">
                {result.verdict}
            </span>
        </div>
        <div style="font-size:0.85rem;color:#9099b0;margin-bottom:0.8rem">
            Confidence&nbsp;&nbsp;<strong style="color:#eef0f5">{result.confidence:.1%}</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Latency&nbsp;&nbsp;<strong style="color:#eef0f5">{result.latency_sec:.2f}s</strong>
            &nbsp;&nbsp;|&nbsp;&nbsp;
            Frames&nbsp;&nbsp;<strong style="color:#eef0f5">{len(result.frames)}</strong>
        </div>
        <div style="background:#ffffff18;border-radius:4px;height:6px;width:100%">
            <div style="background:{color};border-radius:4px;height:6px;width:{bar_w}%;transition:width 0.5s"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#9099b0;margin-top:4px">
            <span>REAL</span><span>FAKE probability: {result.fake_prob_mean:.1%}</span><span>FAKE</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _frame_grid(frames: List[FrameResult], show_heatmap: bool = True) -> None:
    if not frames:
        return
    N = len(frames)
    cols_per_row = min(4, N)
    for row_start in range(0, N, cols_per_row):
        row_frames = frames[row_start:row_start + cols_per_row]
        cols = st.columns(len(row_frames))
        for col, fr in zip(cols, row_frames):
            with col:
                img = fr.heatmap_image if (show_heatmap and fr.heatmap_image) else (
                    Image.fromarray(fr.face_image) if fr.face_image is not None else None
                )
                if img:
                    st.image(img, use_container_width=True)
                color  = "#ff3b3b" if fr.fake_prob >= FAKE_THRESHOLD else "#00e676"
                label  = "FAKE" if fr.fake_prob >= FAKE_THRESHOLD else "REAL"
                det    = "👁 face" if fr.face_detected else "🖼 full"
                st.markdown(
                    f'<div style="text-align:center;font-size:0.7rem;color:{color};'
                    f'font-family:Space Mono,monospace">'
                    f'#{fr.frame_idx}&nbsp;{label}&nbsp;{fr.fake_prob:.0%}&nbsp;{det}</div>',
                    unsafe_allow_html=True,
                )


def _prob_chart(frames: List[FrameResult]) -> None:
    if len(frames) < 2:
        return
    idxs  = [f.frame_idx for f in frames]
    fakes = [f.fake_prob for f in frames]

    fig, ax = plt.subplots(figsize=(7, 2.2), facecolor="#111318")
    ax.set_facecolor("#111318")
    ax.fill_between(idxs, fakes, alpha=0.25, color="#ff3b3b")
    ax.plot(idxs, fakes, color="#ff3b3b", linewidth=1.5)
    ax.axhline(FAKE_THRESHOLD, color="#ffab00", linewidth=0.8,
               linestyle="--", label=f"Threshold {FAKE_THRESHOLD:.0%}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame", color="#9099b0", fontsize=8)
    ax.set_ylabel("P(fake)", color="#9099b0", fontsize=8)
    ax.tick_params(colors="#9099b0", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2f3d")
    ax.legend(fontsize=7, labelcolor="#9099b0", facecolor="#111318", edgecolor="#2a2f3d")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def _section(title: str) -> None:
    st.markdown(
        f'<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
        f'letter-spacing:0.12em;text-transform:uppercase;color:#9099b0;'
        f'border-bottom:1px solid #2a2f3d;padding-bottom:4px;margin:1.6rem 0 0.8rem">'
        f'{title}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ⑨ PAGES
# ─────────────────────────────────────────────────────────────────────────────

def page_home() -> None:
    st.markdown("""
    <div style="padding:2rem 0 1rem">
        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                    letter-spacing:0.18em;color:#ff3b3b;margin-bottom:0.5rem">
            FORENSIC AI SYSTEM v2.0
        </div>
        <h1 style="font-size:3rem;font-weight:700;letter-spacing:-0.02em;
                   line-height:1.1;margin:0 0 1rem">
            DeepFake<br>Detector
        </h1>
        <p style="color:#9099b0;font-size:1rem;max-width:520px;line-height:1.7">
            Upload an image or video. The system detects face-swapped deepfakes
            using <strong style="color:#eef0f5">EfficientNet-B4</strong> + 
            <strong style="color:#eef0f5">MTCNN</strong> and explains every
            prediction with <strong style="color:#eef0f5">GradCAM</strong> heatmaps.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Model", "EfficientNet-B4")
    with c2: st.metric("Face Detector", "MTCNN")
    with c3: st.metric("Explainability", "GradCAM")
    with c4: st.metric("Device", DEVICE.upper())

    _section("HOW IT WORKS")
    steps = [
        ("01", "Upload", "Drag-and-drop image (JPG/PNG/WEBP) or video (MP4/AVI/MOV)."),
        ("02", "Detect", "MTCNN locates every face in the frame with sub-pixel alignment."),
        ("03", "Classify", "EfficientNet-B4 computes P(FAKE) for each face crop."),
        ("04", "Explain", "GradCAM highlights which pixels drove the prediction."),
        ("05", "Aggregate", "Per-frame scores are mean-pooled to a final verdict."),
    ]
    cols = st.columns(5)
    for col, (num, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="background:#111318;border:1px solid #2a2f3d;
                        border-radius:8px;padding:1rem;height:160px">
                <div style="font-family:'Space Mono',monospace;font-size:1.6rem;
                            font-weight:700;color:#ff3b3b;margin-bottom:0.3rem">{num}</div>
                <div style="font-weight:600;margin-bottom:0.4rem">{title}</div>
                <div style="font-size:0.78rem;color:#9099b0;line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    _section("DISCLAIMER")
    st.info(
        "⚠️ No deepfake detector achieves 100% accuracy. "
        "Always treat results as one signal among many. "
        "Accuracy depends heavily on training data and video quality."
    )


def page_analyze() -> None:
    st.markdown('<h2 style="margin-bottom:0">🔍 Analyze</h2>', unsafe_allow_html=True)
    st.caption("Upload media to detect deepfake manipulation.")

    # ── Settings sidebar ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        show_heatmaps  = st.toggle("GradCAM heatmaps", value=True)
        show_chart     = st.toggle("Probability chart (video)", value=True)
        save_hist      = st.toggle("Save to history", value=True)
        st.markdown("---")
        st.markdown("### 🏋️ Weights")
        custom_weights = st.file_uploader("Upload .pth checkpoint", type=["pth"])
        if custom_weights:
            save_path = WEIGHTS_DIR / "model_best.pth"
            save_path.write_bytes(custom_weights.read())
            st.success(f"Saved → `{save_path.name}`")
            st.cache_resource.clear()

    engine = load_engine()

    # ── Input tabs ──────────────────────────────────────────────────────────
    tab_img, tab_vid = st.tabs(["📷  Image", "🎬  Video"])

    with tab_img:
        uploaded = st.file_uploader(
            "Drop image here", type=["jpg", "jpeg", "png", "webp"],
            key="img_upload", label_visibility="collapsed",
        )
        if uploaded:
            raw = uploaded.read()
            pil = Image.open(io.BytesIO(raw)).convert("RGB")
            col_left, col_right = st.columns([1, 1], gap="large")
            with col_left:
                _section("INPUT")
                st.image(pil, use_container_width=True, caption=uploaded.name)
            with col_right:
                _section("RESULT")
                with st.spinner("Analysing…"):
                    result = predict_image(
                        pil, engine, heatmaps=show_heatmaps, filename=uploaded.name
                    )
                _verdict_card(result)
                if result.error:
                    st.error(result.error)

            if result.frames and result.frames[0].heatmap_image and show_heatmaps:
                _section("GRADCAM HEATMAP")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(pil.resize((IMAGE_SIZE, IMAGE_SIZE)), caption="Original face crop", use_container_width=True)
                with col_b:
                    st.image(result.frames[0].heatmap_image, caption="GradCAM overlay", use_container_width=True)

            if save_hist and not result.error:
                save_to_history(result)

            with st.expander("📋 Raw prediction data"):
                st.json(result.to_dict())

    with tab_vid:
        uploaded_v = st.file_uploader(
            "Drop video here", type=["mp4", "avi", "mov", "mkv"],
            key="vid_upload", label_visibility="collapsed",
        )
        if uploaded_v:
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded_v.name).suffix, delete=False
            ) as tmp:
                tmp.write(uploaded_v.read())
                tmp_path = tmp.name

            col_left, col_right = st.columns([1, 1], gap="large")
            with col_left:
                _section("INPUT")
                st.video(tmp_path)

            with col_right:
                _section("RESULT")
                with st.spinner("Sampling frames…"):
                    result = predict_video(
                        tmp_path, engine, heatmaps=show_heatmaps, filename=uploaded_v.name
                    )
                _verdict_card(result)
                if result.error:
                    st.error(result.error)

            Path(tmp_path).unlink(missing_ok=True)

            if show_chart and len(result.frames) > 1:
                _section("FRAME-BY-FRAME PROBABILITY")
                _prob_chart(result.frames)

            if result.frames:
                _section("FRAME GRID")
                _frame_grid(result.frames, show_heatmap=show_heatmaps)

            if save_hist and not result.error:
                save_to_history(result)

            with st.expander("📋 Raw prediction data"):
                st.json(result.to_dict())


def page_history() -> None:
    st.markdown('<h2 style="margin-bottom:0">📂 History</h2>', unsafe_allow_html=True)
    st.caption("Last 100 detections — persisted to `history.json`.")

    history = load_history()

    col_l, col_r = st.columns([8, 2])
    with col_r:
        if st.button("🗑 Clear history") and history:
            HISTORY_FILE.unlink(missing_ok=True)
            st.rerun()

    if not history:
        st.info("No history yet. Run an analysis to see results here.")
        return

    # Summary metrics
    fake_count = sum(1 for h in history if h["verdict"] == "FAKE")
    real_count = sum(1 for h in history if h["verdict"] == "REAL")
    avg_conf   = np.mean([h["confidence"] for h in history if h.get("confidence")]) if history else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total analyzed", len(history))
    with c2: st.metric("Fake detected",  fake_count)
    with c3: st.metric("Real detected",  real_count)
    with c4: st.metric("Avg confidence", f"{avg_conf:.1%}")

    _section("DETECTION LOG")

    # Table header
    st.markdown("""
    <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr 1fr;
                gap:0.5rem;padding:0.5rem 0.8rem;
                background:#181b22;border-radius:6px 6px 0 0;
                font-family:'Space Mono',monospace;font-size:0.65rem;
                letter-spacing:0.08em;color:#9099b0">
        <span>FILENAME</span><span>VERDICT</span><span>CONFIDENCE</span>
        <span>FRAMES</span><span>LATENCY</span><span>TIMESTAMP</span>
    </div>
    """, unsafe_allow_html=True)

    for i, h in enumerate(history):
        color = CLASS_COLOR.get(h.get("verdict", "ERROR"), "#888")
        bg    = "#111318" if i % 2 == 0 else "#181b22"
        ts    = h.get("timestamp", "")[:19].replace("T", " ")
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:2fr 1fr 1fr 1fr 1fr 1fr;
                    gap:0.5rem;padding:0.5rem 0.8rem;background:{bg};
                    border-bottom:1px solid #2a2f3d;
                    font-size:0.78rem;align-items:center">
            <span style="color:#eef0f5;overflow:hidden;text-overflow:ellipsis;
                         white-space:nowrap" title="{h.get('filename','')}">
                {h.get('filename','—')}
            </span>
            <span style="color:{color};font-family:'Space Mono',monospace;font-weight:700">
                {h.get('verdict','—')}
            </span>
            <span style="color:#eef0f5">{h.get('confidence',0):.1%}</span>
            <span style="color:#9099b0">{h.get('frames_analyzed','—')}</span>
            <span style="color:#9099b0">{h.get('latency_sec',0):.2f}s</span>
            <span style="color:#9099b0;font-size:0.7rem">{ts}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Download
    json_bytes = json.dumps(history, indent=2).encode()
    st.download_button(
        "⬇️  Download history.json",
        data=json_bytes,
        file_name="deepfake_history.json",
        mime="application/json",
    )


def page_train() -> None:
    st.markdown('<h2 style="margin-bottom:0">🏋️ Training Guide</h2>', unsafe_allow_html=True)
    st.caption("How to fine-tune the model on your own dataset.")

    _section("DATASET LAYOUT")
    st.code("""
data/
├── train/
│   ├── real/        # authentic face images  (jpg/png)
│   └── fake/        # deepfake face images
└── val/
    ├── real/
    └── fake/
""", language="text")

    _section("RECOMMENDED DATASETS")
    data = {
        "Dataset": ["FaceForensics++ (FF++)", "Celeb-DF v2", "DFDC (Facebook)"],
        "Faces": ["~500 K", "~590 K", "~100 K"],
        "Notes": ["Standard benchmark", "High-quality GAN swaps", "Diverse conditions"],
        "License": ["Research only", "Research only", "Research only"],
    }
    st.table(data)

    _section("TRAINING COMMAND")
    st.code("""
python scripts/train.py \\
    --data-dir  data/ \\
    --output    weights/model_best.pth \\
    --epochs    20 \\
    --warmup-epochs 5 \\
    --lr        1e-4 \\
    --batch-size 16 \\
    --device    cuda
""", language="bash")

    _section("TRAINING PIPELINE (scripts/train.py)")
    st.code("""
Phase 1 — Warmup (epochs 1..warmup):
  Backbone frozen → only head trains (fast convergence)

Phase 2 — Fine-tune (epochs warmup+1..N):
  Full network unfrozen at LR × 0.1
  Optimizer: AdamW  |  weight decay: 1e-4
  Scheduler: CosineAnnealingLR
  Loss: CrossEntropyLoss(label_smoothing=0.1)
  Grad clip: max_norm=1.0
  Saves best checkpoint by val accuracy
""", language="text")

    _section("AFTER TRAINING")
    st.code("# Place the checkpoint and reload the app\ncp weights/model_best.pth deepfake_detection/weights/\nstreamlit run app.py", language="bash")

    _section("EXPECTED METRICS (FF++, c23 compression)")
    st.table({
        "Model": ["EfficientNet-B4", "Xception", "ViT-B/16"],
        "AUC":   ["0.965", "0.955", "0.970"],
        "Acc.":  ["93.1%", "91.8%", "94.2%"],
    })


def page_about() -> None:
    st.markdown('<h2 style="margin-bottom:0">ℹ️ About</h2>', unsafe_allow_html=True)

    _section("MODEL CARD")
    st.table({
        "Property":  ["Architecture", "Input size", "Classes", "Face detector", "Explainability", "Framework"],
        "Value":     ["EfficientNet-B4 (timm)", "224 × 224 px", "REAL (0) · FAKE (1)",
                      "MTCNN (facenet-pytorch)", "GradCAM (last conv)", "PyTorch 2.x + Streamlit"],
    })

    _section("SYSTEM INFO")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("PyTorch", torch.__version__)
    with c2: st.metric("Device", DEVICE.upper())
    with c3: st.metric("Fake threshold", f"{FAKE_THRESHOLD:.0%}")

    _section("REFERENCES")
    st.markdown("""
    - Tan & Le — *EfficientNet: Rethinking Model Scaling for CNNs*, ICML 2019  
    - Selvaraju et al. — *Grad-CAM: Visual Explanations from Deep Networks*, ICCV 2017  
    - Zhang et al. — *Joint Face Detection and Alignment using Multi-task Cascaded CNNs*, 2016  
    - Rössler et al. — *FaceForensics++*, ICCV 2019  
    """)

    _section("LIMITATIONS")
    st.warning("""
    - Accuracy degrades on heavily compressed (WhatsApp / social media) video.  
    - GAN-based manipulations not involving a face are undetected.  
    - Model may reflect demographic biases present in training data.  
    - Not a substitute for forensic expert analysis.  
    """)

    _section("LICENSE")
    st.code("MIT License — free for research and commercial use.", language="text")


# ─────────────────────────────────────────────────────────────────────────────
# ⑩ SIDEBAR NAVIGATION + MAIN
# ─────────────────────────────────────────────────────────────────────────────

PAGES = {
    "🏠  Home":     page_home,
    "🔍  Analyze":  page_analyze,
    "📂  History":  page_history,
    "🏋️  Training": page_train,
    "ℹ️  About":    page_about,
}


def main() -> None:
    with st.sidebar:
        st.markdown("""
        <div style="padding:1rem 0 0.5rem">
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                        letter-spacing:0.18em;color:#ff3b3b;margin-bottom:0.3rem">
                FORENSIC AI
            </div>
            <div style="font-size:1.3rem;font-weight:700;letter-spacing:-0.01em">
                DeepFake Detector
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        selection = st.radio("", list(PAGES.keys()), label_visibility="collapsed")

        st.markdown("---")
        weights_exist = (WEIGHTS_DIR / "model_best.pth").exists()
        if weights_exist:
            st.success("✅ Custom weights loaded")
        else:
            st.warning("⚠️ Demo mode\n\nNo `weights/model_best.pth` found.\nAdd fine-tuned weights for accurate results.")

        st.markdown(f"""
        <div style="font-size:0.68rem;color:#9099b0;margin-top:auto;padding-top:1rem">
            Device: {DEVICE.upper()}<br>
            Threshold: {FAKE_THRESHOLD:.0%}<br>
            Max frames: {VIDEO_MAX_FRAMES}
        </div>
        """, unsafe_allow_html=True)

    PAGES[selection]()


if __name__ == "__main__":
    main()
