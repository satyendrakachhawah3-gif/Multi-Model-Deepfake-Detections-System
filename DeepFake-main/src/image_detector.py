from pathlib import Path
import os
from typing import Any

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification


DEFAULT_PRETRAINED_IMAGE_MODEL = "Wvolf/ViT_Deepfake_Detection"


def _looks_like_hf_repo(model_ref: str) -> bool:
    return "/" in model_ref and not model_ref.endswith(".pth")


def _resolve_image_model_source(model_path: str | None = None) -> str:
    env_model_path = os.getenv("DEEPFAKE_IMAGE_MODEL_PATH")
    if env_model_path:
        candidate = Path(env_model_path)
        if candidate.exists():
            return str(candidate)
        if _looks_like_hf_repo(env_model_path):
            return env_model_path

    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return str(candidate)
        if _looks_like_hf_repo(model_path):
            return model_path

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "models" / "image_wvolf_vit_hf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return DEFAULT_PRETRAINED_IMAGE_MODEL

class ImageDetector:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        self.model_source = _resolve_image_model_source(model_path)
        self.image_processor: Any = AutoImageProcessor.from_pretrained(self.model_source)
        self.model: Any = AutoModelForImageClassification.from_pretrained(self.model_source)
        self.model.to(self.device)
        self.model.eval()

        self.fake_class_index = self._resolve_fake_class_index(getattr(self.model.config, "id2label", {}))

    def _resolve_fake_class_index(self, id2label_map: dict[int | str, Any] | None) -> int:
        env_value = os.getenv("DEEPFAKE_IMAGE_FAKE_INDEX")
        if env_value in {"0", "1"}:
            return int(env_value)

        if isinstance(id2label_map, dict):
            normalized = {int(k): str(v).strip().lower() for k, v in id2label_map.items()}
            for idx, label in normalized.items():
                if "fake" in label:
                    return idx
            for idx, label in normalized.items():
                if "real" in label and idx in {0, 1}:
                    return 1 - idx

        return 1

    def predict(self, image_paths: str | Path | list[str] | list[Path] | None) -> np.ndarray:
        """Predict on image paths"""
        if image_paths is None:
            return np.array([])

        if isinstance(image_paths, (str, Path)):
            image_paths = [str(image_paths)]

        probs: list[float] = []
        for path in image_paths:
            img = Image.open(Path(path)).convert("RGB")
            probs.append(self._predict_single_image(img))
        return np.array(probs)

    def _predict_single_image(self, img: Image.Image) -> float:
        inputs = self.image_processor(images=img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = self.model(**inputs)
            prob = torch.softmax(output.logits, dim=1)[0, self.fake_class_index].item()
        return float(prob)

    def predict_with_consistency(self, image_path: str) -> dict[str, Any]:
        img = Image.open(Path(image_path)).convert("RGB")
        variants = [img, ImageOps.mirror(img)]
        scores = [self._predict_single_image(variant) for variant in variants]
        return {
            "score": float(scores[0]),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": [float(score) for score in scores],
        }

    def explain_gradcam(self, image_path: str | Path) -> Image.Image:
        """Return original image when GradCAM is unavailable for transformer backbones."""
        return Image.open(Path(image_path)).convert("RGB")
