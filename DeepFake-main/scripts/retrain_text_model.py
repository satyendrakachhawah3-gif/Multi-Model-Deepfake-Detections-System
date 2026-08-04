from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.eval_text_dataset import SAMPLES


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_label(value: object) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None

    text = str(value).strip().lower()
    if text in {"1", "fake", "f", "false", "generated", "manipulated"}:
        return 1
    if text in {"0", "real", "true", "r", "authentic", "genuine"}:
        return 0

    try:
        numeric = int(float(text))
        if numeric in (0, 1):
            return numeric
    except ValueError:
        return None

    return None


def _load_csv_dataset(
    csv_path: str,
    text_column: str,
    label_column: str,
    max_samples: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    frame = pd.read_csv(csv_path, low_memory=False)
    if text_column not in frame.columns:
        raise KeyError(f"Text column '{text_column}' missing in {csv_path}")
    if label_column not in frame.columns:
        raise KeyError(f"Label column '{label_column}' missing in {csv_path}")

    frame = frame[[text_column, label_column]].copy()
    frame[text_column] = frame[text_column].astype(str).map(_normalize_spaces)
    frame[label_column] = frame[label_column].map(_normalize_label)
    frame = frame.dropna(subset=[text_column, label_column])
    frame = frame[frame[text_column].str.len() >= 20].copy()
    frame[label_column] = frame[label_column].astype(int)

    if max_samples > 0 and len(frame) > max_samples:
        frame = frame.sample(n=max_samples, random_state=seed)

    if frame[label_column].nunique() < 2:
        raise RuntimeError(f"Dataset at {csv_path} must contain both classes")

    texts = frame[text_column].tolist()
    labels = frame[label_column].tolist()
    return texts, labels
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _strip_emoji(text: str) -> str:
    text = re.sub(r"[^\w\s.,!?;:'\"()\-]", " ", text)
    return _normalize_spaces(text)


def build_augmented_labeled_samples(max_augs_per_sample: int = 3) -> tuple[list[str], list[int]]:
    clear_samples = [sample for sample in SAMPLES if sample["label"] is not None]

    texts: list[str] = []
    labels: list[int] = []

    for sample in clear_samples:
        text = sample["text"]
        label = int(sample["label"])

        variants = [
            text,
            _normalize_spaces(text),
            text.lower(),
            _strip_emoji(text),
            f"{text} Verified source reports this claim.",
            f"{text} Read before sharing.",
        ]

        seen = set()
        kept = 0
        for variant in variants:
            normalized = _normalize_spaces(variant)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            texts.append(normalized)
            labels.append(label)
            kept += 1
            if kept >= max_augs_per_sample:
                break

    return texts, labels


@dataclass
class EvalMetrics:
    accuracy: float
    real_precision: float
    fake_precision: float
    real_recall: float
    fake_recall: float
    recall_gap_abs: float


class EncodedTextDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def compute_metrics(y_true: list[int], y_pred: list[int]) -> EvalMetrics:
    total = len(y_true)
    tp = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(y_true, y_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(y_true, y_pred) if y == 1 and p == 0)

    accuracy = (tp + tn) / total if total else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    fake_precision = tp / (tp + fp) if (tp + fp) else 0.0

    return EvalMetrics(
        accuracy=accuracy,
        real_precision=real_precision,
        fake_precision=fake_precision,
        real_recall=real_recall,
        fake_recall=fake_recall,
        recall_gap_abs=abs(real_recall - fake_recall),
    )


def evaluate_model(
    model,
    loader: DataLoader,
    device: torch.device,
) -> EvalMetrics:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []

    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    return compute_metrics(y_true, y_pred)


def save_model_artifacts(
    model,
    tokenizer,
    target_dir: Path,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    for stale_name in ["model.safetensors", "model-00001-of-00001.safetensors", "model.safetensors.index.json"]:
        stale_path = target_dir / stale_name
        if stale_path.exists():
            try:
                stale_path.unlink()
            except OSError:
                pass

    model.config.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)

    weights_path = target_dir / "pytorch_model.bin"
    tmp_weights_path = target_dir / "pytorch_model.bin.tmp"
    torch.save(model.state_dict(), tmp_weights_path)
    shutil.move(str(tmp_weights_path), str(weights_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain text detector with fairness-aware validation")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-augs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--base-model", type=str, default=os.getenv("DEEPFAKE_TEXT_BASE_MODEL", "microsoft/deberta-v3-base"))
    parser.add_argument("--dataset-csv", type=str, default="", help="Single external dataset CSV to split into train/val")
    parser.add_argument("--train-csv", type=str, default="", help="External training CSV path")
    parser.add_argument("--val-csv", type=str, default="", help="External validation CSV path")
    parser.add_argument("--text-column", type=str, default="text", help="Text column name for external CSV")
    parser.add_argument("--label-column", type=str, default="label", help="Label column name for external CSV")
    parser.add_argument("--val-size", type=float, default=0.2, help="Validation split size when using --dataset-csv")
    parser.add_argument("--max-samples", type=int, default=0, help="Max external samples (0 means all)")
    parser.add_argument("--input-max-length", type=int, default=128, help="Tokenizer max length for training datasets")
    args = parser.parse_args()

    set_seed(args.seed)

    base_model_path = args.base_model
    output_dir = PROJECT_ROOT / "models" / "text_deberta_v3"
    notebook_output_dir = PROJECT_ROOT / "notebooks" / "models" / "text_deberta_v3"
    report_dir = PROJECT_ROOT / "outputs" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    notebook_output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    data_source = "augmented_inline"
    if args.train_csv and args.val_csv:
        x_train, y_train = _load_csv_dataset(
            args.train_csv,
            text_column=args.text_column,
            label_column=args.label_column,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        x_val, y_val = _load_csv_dataset(
            args.val_csv,
            text_column=args.text_column,
            label_column=args.label_column,
            max_samples=0,
            seed=args.seed,
        )
        data_source = "external_split_csv"
    elif args.dataset_csv:
        texts, labels = _load_csv_dataset(
            args.dataset_csv,
            text_column=args.text_column,
            label_column=args.label_column,
            max_samples=args.max_samples,
            seed=args.seed,
        )
        x_train, x_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=args.val_size,
            random_state=args.seed,
            stratify=labels,
        )
        data_source = "external_single_csv"
    else:
        texts, labels = build_augmented_labeled_samples(max_augs_per_sample=args.max_augs)
        if len(set(labels)) < 2:
            raise RuntimeError("Training data must contain both classes.")

        x_train, x_val, y_train, y_val = train_test_split(
            texts,
            labels,
            test_size=0.25,
            random_state=args.seed,
            stratify=labels,
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(base_model_path, num_labels=2)

    train_enc = tokenizer(
        x_train,
        truncation=True,
        padding=True,
        max_length=args.input_max_length,
        return_tensors="pt",
    )
    val_enc = tokenizer(
        x_val,
        truncation=True,
        padding=True,
        max_length=args.input_max_length,
        return_tensors="pt",
    )

    train_ds = EncodedTextDataset(train_enc, y_train)
    val_ds = EncodedTextDataset(val_enc, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * max(0.0, min(1.0, args.warmup_ratio)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_state = None
    best_objective = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            labels_batch = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch, labels=labels_batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            total_loss += float(loss.item())

        train_loss = total_loss / max(1, len(train_loader))
        val_metrics = evaluate_model(model, val_loader, device)

        balanced_acc = 0.5 * (val_metrics.real_recall + val_metrics.fake_recall)
        objective = balanced_acc - 0.2 * val_metrics.recall_gap_abs

        history.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_accuracy": float(val_metrics.accuracy),
                "val_real_precision": float(val_metrics.real_precision),
                "val_fake_precision": float(val_metrics.fake_precision),
                "val_real_recall": float(val_metrics.real_recall),
                "val_fake_recall": float(val_metrics.fake_recall),
                "val_recall_gap_abs": float(val_metrics.recall_gap_abs),
                "val_objective": float(objective),
            }
        )

        print(
            f"Epoch {epoch}/{args.epochs} | loss={train_loss:.4f} | "
            f"acc={val_metrics.accuracy:.4f} | gap={val_metrics.recall_gap_abs:.4f}"
        )

        if objective > best_objective:
            best_objective = objective
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping triggered after epoch {epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    save_model_artifacts(model, tokenizer, output_dir)
    save_model_artifacts(model, tokenizer, notebook_output_dir)

    final_val_metrics = evaluate_model(model, val_loader, device)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"text_retrain_report_{ts}.json"

    report = {
        "base_model_path": str(base_model_path),
        "saved_model_path": str(output_dir),
        "saved_notebook_model_path": str(notebook_output_dir),
        "device": str(device),
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "patience": args.patience,
        "warmup_ratio": args.warmup_ratio,
        "grad_clip": args.grad_clip,
        "input_max_length": args.input_max_length,
        "max_augs_per_sample": args.max_augs,
        "data_source": data_source,
        "dataset_csv": args.dataset_csv,
        "train_csv": args.train_csv,
        "val_csv": args.val_csv,
        "text_column": args.text_column,
        "label_column": args.label_column,
        "val_size": args.val_size,
        "max_samples": args.max_samples,
        "best_objective": float(best_objective),
        "final_val_metrics": {
            "accuracy": float(final_val_metrics.accuracy),
            "real_precision": float(final_val_metrics.real_precision),
            "fake_precision": float(final_val_metrics.fake_precision),
            "real_recall": float(final_val_metrics.real_recall),
            "fake_recall": float(final_val_metrics.fake_recall),
            "recall_gap_abs": float(final_val_metrics.recall_gap_abs),
        },
        "history": history,
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=== TEXT RETRAIN COMPLETE ===")
    print(f"Train samples: {len(x_train)} | Val samples: {len(x_val)}")
    print(f"Saved model: {output_dir}")
    print(f"Saved notebook model: {notebook_output_dir}")
    print(f"Val accuracy: {final_val_metrics.accuracy:.4f}")
    print(f"Real recall: {final_val_metrics.real_recall:.4f}")
    print(f"Fake recall: {final_val_metrics.fake_recall:.4f}")
    print(f"Recall gap: {final_val_metrics.recall_gap_abs:.4f}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
