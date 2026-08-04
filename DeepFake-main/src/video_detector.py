import os

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from pathlib import Path
import cv2
import numpy as np


def _resolve_video_model_path(model_path: str | None = None) -> Path:
    env_model_path = os.getenv("DEEPFAKE_VIDEO_MODEL_PATH")
    if env_model_path:
        candidate = Path(env_model_path)
        if candidate.exists():
            return candidate

    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return candidate

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "models" / "video_model.pth",
        project_root / "models" / "video_xception_bilstm_finetuned_bias.pth",
        project_root / "models" / "video_xception_bilstm.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Video model not found. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )


def _resolve_sequence_length(checkpoint: object, default: int = 16) -> int:
    if not isinstance(checkpoint, dict):
        return default

    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, dict):
        return default

    seq_len = cfg.get("seq_len")
    try:
        value = int(seq_len)
    except (TypeError, ValueError):
        return default

    return value if value > 0 else default

class VideoModel(nn.Module):
    def __init__(self, seq_len: int = 16, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.xception = timm.create_model('xception', pretrained=True)
        if hasattr(self.xception, 'fc'):
            self.xception.fc = nn.Identity()
        else:
            self.xception.reset_classifier(0)

        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),
        )
        self.dropout = nn.Dropout(dropout)
        self.seq_len = seq_len
    
    def forward(self, x):  # [B, T, C, H, W]
        B, T = x.shape[:2]
        features = []
        for t in range(T):
            feat = self.xception(x[:, t])
            feat = self.dropout(feat)
            features.append(feat)
        seq_feat = torch.stack(features, dim=1)

        lstm_out, _ = self.lstm(seq_feat)
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)
        pooled = torch.sum(attn_weights * lstm_out, dim=1)
        return self.classifier(pooled)

class VideoDetector:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(42)
        resolved_path = _resolve_video_model_path(model_path)
        self.model_path = resolved_path
        checkpoint = torch.load(resolved_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        hidden_dim = 256
        num_layers = 2
        self.seq_len = _resolve_sequence_length(checkpoint)
        if isinstance(state_dict, dict):
            lstm_hh = state_dict.get("lstm.weight_hh_l0")
            if isinstance(lstm_hh, torch.Tensor) and lstm_hh.ndim == 2:
                hidden_dim = int(lstm_hh.shape[1])
            num_layers = 2 if "lstm.weight_ih_l1" in state_dict else 1

        self.model = VideoModel(seq_len=self.seq_len, hidden_dim=hidden_dim, num_layers=num_layers, dropout=0.5)
        self.model_loaded = False
        self.load_error: str | None = None
        try:
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
        except Exception as exc:
            self.model_loaded = False
            self.load_error = str(exc)
        
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.SEQ_LEN = self.seq_len
    
    def extract_frames(self, video_path):
        """Extract evenly spaced frames from video"""
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []

        if total_frames <= 0:
            cap.release()
            return frames

        if total_frames < self.seq_len:
            indices = list(range(total_frames))
            while indices and len(indices) < self.seq_len:
                indices.extend(indices[: self.seq_len - len(indices)])
        else:
            indices = np.linspace(0, total_frames - 1, self.seq_len, dtype=int).tolist()

        for idx in indices[: self.seq_len]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame_rgb))
        cap.release()
        return frames
    
    def predict(self, video_path):
        """Predict fake probability"""
        frames = self.extract_frames(video_path)
        if len(frames) == 0:
            return 0.5, 0

        if not self.model_loaded:
            return 0.5, len(frames)
        
        # Pad/truncate to SEQ_LEN
        frame_tensors = []
        for frame in frames[:self.seq_len]:
            frame_tensors.append(self.transform(frame))
        while len(frame_tensors) < self.seq_len and frame_tensors:
            frame_tensors.append(frame_tensors[-1])
        
        input_tensor = torch.stack(frame_tensors).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, 1].item()
        return prob, len(frames)
