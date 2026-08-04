from pathlib import Path
import os
import re
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_PRETRAINED_TEXT_MODEL = "microsoft/deberta-v3-base"


def _split_title_content(text: str) -> tuple[str, str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return "", ""

    sentences = re.split(r"(?<=[.!?])\s+", normalized, maxsplit=1)
    if len(sentences) == 2 and sentences[0]:
        return sentences[0], sentences[1]

    words = normalized.split(" ")
    title = " ".join(words[:12])
    content = " ".join(words[12:]) if len(words) > 12 else normalized
    return title, content


def _format_for_fake_news_model(text: str) -> str:
    title, content = _split_title_content(text)
    return f"<title>{title}<content>{content}<end>"


def _label_to_fake_probability(logits: torch.Tensor, id2label: dict[int, str] | None) -> np.ndarray:
    probs = torch.softmax(logits, dim=-1).cpu().numpy()
    if probs.shape[1] == 2 and id2label:
        normalized = {int(k): str(v).lower() for k, v in id2label.items()}

        # Prefer an explicit 'fake' label when available
        for idx, label in normalized.items():
            if "fake" in label or "manip" in label or "false" in label:
                return probs[:, idx]

        # If labels use truthy names like 'true' or 'real', treat the other index as the FAKE probability
        for idx, label in normalized.items():
            if "real" in label or "true" in label:
                other = 1 - idx
                return probs[:, other]

    # Fallback to previous convention: class index 1 means FAKE.
    return probs[:, 1]


def _resolve_text_model_path(model_path: str | None = None) -> str:
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return str(candidate)
        return model_path

    env_model = os.getenv("DEEPFAKE_TEXT_MODEL")
    if env_model:
        env_candidate = Path(env_model)
        if env_candidate.exists():
            return str(env_candidate)
        return env_model

    project_root = Path(__file__).resolve().parents[1]
    local_candidates = [
        project_root / "models" / "text_deberta_v3",
    ]
    for candidate in local_candidates:
        if candidate.exists():
            return str(candidate)

    return DEFAULT_PRETRAINED_TEXT_MODEL

class TextDetector:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_model = _resolve_text_model_path(model_path)
        self.model_source = resolved_model
        source_lower = resolved_model.lower()
        self._expects_fake_news_prompt = (
            "roberta-fake-news-classification" in source_lower
        )
        self.tokenizer: Any = AutoTokenizer.from_pretrained(resolved_model, use_fast=True)
        self.model: Any = AutoModelForSequenceClassification.from_pretrained(resolved_model)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, texts: str | list[str] | None) -> np.ndarray:
        """Predict fake probability for texts"""
        if texts is None:
            return np.array([])
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return np.array([])

        if self._expects_fake_news_prompt:
            texts = [_format_for_fake_news_model(text) for text in texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            id2label = None
            if hasattr(self.model, "config") and hasattr(self.model.config, "id2label"):
                id2label = self.model.config.id2label
            probs = _label_to_fake_probability(outputs.logits, id2label)
        return probs

    def _text_variants(self, text: str) -> list[str]:
        collapsed = re.sub(r"\s+", " ", text).strip()
        no_emoji = re.sub(r"[^\w\s.,!?;:'\"()\-]", " ", collapsed)
        no_emoji = re.sub(r"\s+", " ", no_emoji).strip()
        return [
            text,
            collapsed,
            collapsed.lower(),
            no_emoji,
        ]

    def predict_with_consistency(self, text: str) -> dict[str, Any]:
        variants = self._text_variants(text)
        scores = self.predict(variants)
        return {
            "score": float(scores[0]),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": [float(score) for score in scores],
        }
    
    def explain_shap(self, background_texts: list[str], test_texts: list[str], n_samples: int = 100):
        """SHAP explanations"""
        try:
            import shap
        except ImportError as error:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            ) from error

        def predict_fn(texts: list[str]) -> np.ndarray:
            return self.predict(texts)

        explainer = shap.KernelExplainer(predict_fn, background_texts[:50])
        shap_values = explainer.shap_values(test_texts, nsamples=n_samples)
        return shap_values
