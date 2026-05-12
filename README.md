# DeepFake Detector 🔬

Single-file Streamlit deepfake detection app.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at **http://localhost:8501**

## Pages

| Page | Description |
|---|---|
| 🏠 Home | Overview, pipeline steps |
| 🔍 Analyze | Upload image/video → verdict + GradCAM |
| 📂 History | Detection log (persisted to `history.json`) |
| 🏋️ Training | Guide to fine-tune on your own dataset |
| ℹ️ About | Model card, references, limitations |

## Adding your weights

```bash
cp /path/to/your/model_best.pth weights/model_best.pth
# Restart the app — it auto-loads on startup
```

## Without weights (demo mode)

The app runs immediately with an ImageNet-pretrained backbone.
Predictions will be random until you supply fine-tuned weights.

## Directory structure

```
deepfake_streamlit/
├── app.py                  # entire project — one file
├── requirements.txt
├── .streamlit/
│   └── config.toml
├── weights/                # place model_best.pth here
└── history.json            # auto-created on first analysis
```
