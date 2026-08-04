from __future__ import annotations

import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.image_detector import ImageDetector


def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value


def metrics(binary_truth: list[int], binary_pred: list[int]) -> dict[str, float]:
    total = len(binary_truth)
    tp = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 1 and p == 0)

    accuracy = (tp + tn) / total if total else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    balanced_accuracy = (real_recall + fake_recall) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "tp_fake": tp,
        "tn_real": tn,
        "fp_real_as_fake": fp,
        "fn_fake_as_real": fn,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "recall_gap_abs": abs(real_recall - fake_recall),
    }


def iter_images(folder: Path):
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        for path in folder.rglob(ext):
            if path.is_file():
                yield path


def sample_paths(paths: list[Path], max_per_class: int | None, seed: int, class_offset: int) -> list[Path]:
    if max_per_class is None or max_per_class <= 0 or len(paths) <= max_per_class:
        return paths
    rng = random.Random(seed + class_offset)
    return rng.sample(paths, max_per_class)


def main() -> None:
    test_root_name = os.getenv("DEEPFAKE_IMAGE_TEST_ROOT", "Test_Image")
    test_root = PROJECT_ROOT / test_root_name
    real_dir = test_root / "Real"
    fake_dir = test_root / "Fake"

    if not real_dir.exists() or not fake_dir.exists():
        raise FileNotFoundError("Expected Test_Image/Real and Test_Image/Fake folders")

    threshold = get_env_float("DEEPFAKE_THRESHOLD_IMAGE", 0.568)
    margin = get_env_float("DEEPFAKE_MARGIN_IMAGE", 0.0)
    std_limit = get_env_float("DEEPFAKE_IMAGE_STD_LIMIT", 0.12)
    max_per_class_raw = os.getenv("DEEPFAKE_IMAGE_MAX_PER_CLASS")
    sample_seed_raw = os.getenv("DEEPFAKE_IMAGE_SAMPLE_SEED", "42")

    max_per_class: int | None = None
    if max_per_class_raw:
        try:
            parsed = int(max_per_class_raw)
            if parsed > 0:
                max_per_class = parsed
        except ValueError:
            max_per_class = None

    try:
        sample_seed = int(sample_seed_raw)
    except ValueError:
        sample_seed = 42

    detector = ImageDetector()

    real_paths = sorted(iter_images(real_dir))
    fake_paths = sorted(iter_images(fake_dir))
    sampled_real = sample_paths(real_paths, max_per_class=max_per_class, seed=sample_seed, class_offset=0)
    sampled_fake = sample_paths(fake_paths, max_per_class=max_per_class, seed=sample_seed, class_offset=1)

    samples = [(p, 0) for p in sampled_real] + [(p, 1) for p in sampled_fake]

    rows = []
    y_true: list[int] = []
    y_pred: list[int] = []
    skipped = 0

    for image_path, truth in samples:
        try:
            c = detector.predict_with_consistency(str(image_path))
            raw_score = float(c["score"])
            mean_score = float(c["mean_score"])
            std_score = float(c["std_score"])

            if raw_score >= threshold and (mean_score < threshold or std_score > std_limit):
                final_score = mean_score
            else:
                final_score = raw_score

            pred = 1 if final_score > (threshold + margin) else 0

            rows.append(
                {
                    "path": str(image_path.relative_to(PROJECT_ROOT)),
                    "truth_label": truth,
                    "final_fake_probability": final_score,
                    "raw_probability": raw_score,
                    "mean_probability": mean_score,
                    "std_probability": std_score,
                    "pred_label": pred,
                    "pred_text": "FAKE" if pred == 1 else "REAL",
                }
            )
            y_true.append(truth)
            y_pred.append(pred)
        except Exception:
            skipped += 1

    summary = metrics(y_true, y_pred)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "outputs" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"image_dataset_eval_{ts}.csv"
    json_path = out_dir / f"image_dataset_eval_summary_{ts}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "truth_label",
                "final_fake_probability",
                "raw_probability",
                "mean_probability",
                "std_probability",
                "pred_label",
                "pred_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "dataset_root": str(test_root_name),
        "sample_count_total_found": len(samples),
        "sample_count_real_found": len(real_paths),
        "sample_count_fake_found": len(fake_paths),
        "sample_count_real_used": len(sampled_real),
        "sample_count_fake_used": len(sampled_fake),
        "max_per_class": max_per_class,
        "sample_seed": sample_seed,
        "sample_count_evaluated": len(y_true),
        "sample_count_skipped": skipped,
        "threshold": threshold,
        "margin": margin,
        "std_limit": std_limit,
        "metrics": summary,
        "output_csv": str(csv_path),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== IMAGE MODEL EVALUATION COMPLETE ===")
    print(f"Found samples: {len(samples)} | Evaluated: {len(y_true)} | Skipped: {skipped}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.4f}")
    print(f"Real recall: {summary['real_recall']:.4f}")
    print(f"Fake recall: {summary['fake_recall']:.4f}")
    print(f"Recall gap |real-fake|: {summary['recall_gap_abs']:.4f}")
    print(f"FP (real->fake): {summary['fp_real_as_fake']}")
    print(f"FN (fake->real): {summary['fn_fake_as_real']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
