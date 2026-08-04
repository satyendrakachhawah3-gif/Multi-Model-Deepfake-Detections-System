from __future__ import annotations

import csv
import json
import os
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.video_detector import VideoDetector


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
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    fake_precision = tp / (tp + fp) if (tp + fp) else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    balanced_accuracy = (real_recall + fake_recall) / 2
    macro_precision = (real_precision + fake_precision) / 2
    macro_recall = (real_recall + fake_recall) / 2

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "real_precision": real_precision,
        "fake_precision": fake_precision,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "tp_fake": tp,
        "tn_real": tn,
        "fp_real_as_fake": fp,
        "fn_fake_as_real": fn,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "recall_gap_abs": abs(real_recall - fake_recall),
    }


def load_video_defaults(project_root: Path) -> tuple[float, float]:
    threshold_default = 0.3055
    margin_default = 0.2
    meta_path = project_root / "models" / "video_ensemble_meta.json"
    if not meta_path.exists():
        return threshold_default, margin_default

    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return threshold_default, margin_default

    thresholds = payload.get("thresholds", {}) if isinstance(payload, dict) else {}
    if not isinstance(thresholds, dict):
        return threshold_default, margin_default

    threshold = thresholds.get("recommended_env_video_threshold", threshold_default)
    margin = thresholds.get("recommended_env_video_margin", margin_default)

    try:
        threshold_value = float(threshold)
        margin_value = float(margin)
    except (TypeError, ValueError):
        return threshold_default, margin_default

    if not (0.0 <= threshold_value <= 1.0):
        threshold_value = threshold_default
    if not (0.0 <= margin_value <= 1.0):
        margin_value = margin_default

    return threshold_value, margin_value


def iter_videos(folder: Path):
    for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"):
        for path in folder.rglob(ext):
            if path.is_file():
                yield path


def infer_label_from_path(path: Path) -> int | None:
    parts = [segment.lower() for segment in path.parts]
    has_real = any("real" in segment for segment in parts)
    has_fake = any("fake" in segment for segment in parts)
    if has_real and not has_fake:
        return 0
    if has_fake and not has_real:
        return 1
    return None


def main() -> None:
    test_root_name = os.getenv("DEEPFAKE_VIDEO_TEST_ROOT", "Test_Video_Synth")
    test_root = PROJECT_ROOT / test_root_name
    real_dir = test_root / "Real"
    fake_dir = test_root / "Fake"

    threshold_default, margin_default = load_video_defaults(PROJECT_ROOT)
    threshold = get_env_float("DEEPFAKE_THRESHOLD_VIDEO", threshold_default)
    margin = get_env_float("DEEPFAKE_MARGIN_VIDEO", margin_default)

    detector = VideoDetector()

    if not detector.model_loaded:
        raise RuntimeError("Video model failed to load. Cannot run evaluation.")

    samples = []
    if real_dir.exists() and fake_dir.exists():
        for p in iter_videos(real_dir):
            samples.append((p, 0))
        for p in iter_videos(fake_dir):
            samples.append((p, 1))
    else:
        unlabeled = 0
        for p in iter_videos(test_root):
            label = infer_label_from_path(p.relative_to(test_root))
            if label is None:
                unlabeled += 1
                continue
            samples.append((p, label))
        if not samples:
            raise FileNotFoundError(
                f"No labeled videos found under {test_root_name}. "
                "Expected Real/Fake folders or subfolder names containing 'real' or 'fake'."
            )
        if unlabeled:
            print(f"Skipped {unlabeled} unlabeled videos (no clear real/fake folder hint).")

    rows = []
    y_true: list[int] = []
    y_pred: list[int] = []
    skipped = 0

    for video_path, truth in samples:
        try:
            score, frames = detector.predict(str(video_path))
            final_score = float(score)
            pred = 1 if final_score > (threshold + margin) else 0

            rows.append(
                {
                    "path": str(video_path.relative_to(PROJECT_ROOT)),
                    "truth_label": truth,
                    "final_fake_probability": final_score,
                    "frames_analyzed": int(frames),
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

    csv_path = out_dir / f"video_dataset_eval_{ts}.csv"
    json_path = out_dir / f"video_dataset_eval_summary_{ts}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "path",
                "truth_label",
                "final_fake_probability",
                "frames_analyzed",
                "pred_label",
                "pred_text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "dataset_root": str(test_root_name),
        "sample_count_total_found": len(samples),
        "sample_count_evaluated": len(y_true),
        "sample_count_skipped": skipped,
        "threshold": threshold,
        "margin": margin,
        "metrics": summary,
        "output_csv": str(csv_path),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("=== VIDEO MODEL EVALUATION COMPLETE ===")
    print(f"Found samples: {len(samples)} | Evaluated: {len(y_true)} | Skipped: {skipped}")
    print(f"Accuracy: {summary['accuracy']:.4f}")
    print(f"Balanced accuracy: {summary['balanced_accuracy']:.4f}")
    print(f"Real precision: {summary['real_precision']:.4f}")
    print(f"Fake precision: {summary['fake_precision']:.4f}")
    print(f"Macro precision: {summary['macro_precision']:.4f}")
    print(f"Real recall: {summary['real_recall']:.4f}")
    print(f"Fake recall: {summary['fake_recall']:.4f}")
    print(f"Macro recall: {summary['macro_recall']:.4f}")
    print(f"Recall gap |real-fake|: {summary['recall_gap_abs']:.4f}")
    print(f"FP (real->fake): {summary['fp_real_as_fake']}")
    print(f"FN (fake->real): {summary['fn_fake_as_real']}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {json_path}")


if __name__ == "__main__":
    main()
