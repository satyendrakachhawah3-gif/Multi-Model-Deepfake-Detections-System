# Deepfake Detection Project

A multi-modal deepfake detection system that analyzes text, images, and videos using deep learning techniques with explainable AI.

## 📋 Project Overview

This project implements a comprehensive deepfake detection system with three modalities:
- **Text Detection**: Fine-tuned transformer classifier + SHAP for fake news detection
- **Image Detection**: EfficientNet-B4 + GradCAM for face manipulation detection
- **Video Detection**: Xception + BiLSTM for temporal deepfake detection
- **Fusion Engine**: Decision-level fusion for multi-modal analysis

## 📁 Project Structure

```
deepfake_project/
├── data/                      # Datasets (50GB total)
│   ├── text/                  # Text datasets
│   │   ├── liar_dataset.csv   # LIAR (12K samples) ✓ DOWNLOADED
│   │   ├── fake_news.csv      # Kaggle Fake News (45K samples)
│   │   ├── train/, val/, test/
│   ├── images/                # Face images
│   └── videos/                # Video files
├── models/                    # Trained model checkpoints
├── notebooks/                 # Jupyter experiments
├── src/                       # Production Python modules
├── app/                       # Streamlit deployment
├── docs/                      # Documentation
├── tests/                     # Unit tests
└── outputs/                   # Results & visualizations
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
# DeepFake Detection

Multi-modal deepfake detection system for text, image, and video with a FastAPI backend and frontend dashboard.

## Overview

This project supports three independent detectors and a fused decision workflow:

- Text detector (pretrained Hugging Face model, local copy)
- Image detector (pretrained Hugging Face model, local copy)
- Video detector (local checkpoint)
- Dashboard with per-modality scores and confusion matrices

## Current Model Setup

### Text

- Active model: `models/text_deberta_v3`
- Local path: `models/text_deberta_v3`
- Module: `src/text_detector.py`

### Image

- Active model: `Wvolf/ViT_Deepfake_Detection`
- Local path: `models/image_wvolf_vit_hf`
- Module: `src/image_detector.py`

### Video

- Active checkpoint: `models/video_model.pth`
- Module: `src/video_detector.py`

## Project Structure

```text
Deepfake/
	app/
		api_server.py
		frontend/
			index.html
			analysis.html
			dashboard.html
			script.js
			styles.css
			config.js
	models/
		image_wvolf_vit_hf/
		text_deberta_v3/
		video_model.pth
	outputs/
		metrics/
	scripts/
		eval_text_dataset.py
		eval_image_dataset.py
		eval_video_dataset.py
	src/
		text_detector.py
		image_detector.py
		video_detector.py
```

## Setup

### Python Environment

```bash
python -m venv .venv-1
.venv-1\Scripts\activate
pip install -r requirements.txt
pip install -r app/requirements_app.txt
```

### Run Backend API

```bash
python -m uvicorn app.api_server:app --host 127.0.0.1 --port 8000
```

API docs:

- `http://127.0.0.1:8000/docs`

### Production API Controls

Optional environment variables for production-like behavior:

```powershell
$env:DEEPFAKE_API_KEY='replace-with-strong-key'
$env:DEEPFAKE_RATE_LIMIT_PER_MINUTE='60'
$env:DEEPFAKE_MAX_UPLOAD_MB='50'
```

Threshold and calibration variables (already supported):

```powershell
$env:DEEPFAKE_THRESHOLD_TEXT='0.954'
$env:DEEPFAKE_THRESHOLD_IMAGE='0.568'
$env:DEEPFAKE_THRESHOLD_VIDEO='0.3055'
$env:DEEPFAKE_MARGIN_TEXT='0.0275'
$env:DEEPFAKE_MARGIN_IMAGE='0.0'
$env:DEEPFAKE_MARGIN_VIDEO='0.2'
```

If `DEEPFAKE_API_KEY` is set, pass header:

- `X-API-Key: <your-key>`

### API Endpoints (v1)

- `GET /api/v1/health`
- `GET /api/v1/metrics/confusion-matrices`
- `GET /api/v1/system/status`
- `POST /api/v1/analyze`
- `POST /api/v1/jobs/video` (async submit)
- `GET /api/v1/jobs/{job_id}` (poll status)

`POST /api/v1/analyze` now supports multimodal input in one request:

- Any combination of text, image, and video is accepted.
- If multiple modalities are provided, weighted fusion is applied with weights `(0.3, 0.3, 0.4)`.
- If only one modality is provided, branch-specific threshold and margin are used.

For async video processing:

1. Submit video to `POST /api/v1/jobs/video`.
2. Poll `GET /api/v1/jobs/{job_id}` until status is `completed` or `failed`.

## Frontend

Open these pages from `app/frontend/`:

- `index.html`
- `analysis.html`
- `dashboard.html`

If needed, set API URL in `app/frontend/config.js`:

```js
window.__DEEPFAKE_CONFIG__ = {
		apiBaseUrl: "http://127.0.0.1:8000"
};
```

## Evaluation

### Text

```bash
python scripts/eval_text_dataset.py
```

### Image (balanced sampled evaluation, faster than full set)

```bash
$env:DEEPFAKE_IMAGE_TEST_ROOT='Test_Image'
$env:DEEPFAKE_IMAGE_MAX_PER_CLASS='40'
$env:DEEPFAKE_IMAGE_SAMPLE_SEED='42'
python scripts/eval_image_dataset.py
```

### Video

```bash
$env:DEEPFAKE_VIDEO_TEST_ROOT='test_vedio'
python scripts/eval_video_dataset.py
```

Evaluation outputs are written to `outputs/metrics/`.

## Confusion Matrix UI

Dashboard confusion matrices are loaded from:

- `GET /metrics/confusion-matrices`
- `GET /api/v1/metrics/confusion-matrices`

The API reads the latest summary files for text, image, and video and renders each modality matrix in the dashboard.

## Notes

- Local model directories can be large; keep them out of Git history.
- If inference fails due to model download limits, set a valid `HF_TOKEN` environment variable.