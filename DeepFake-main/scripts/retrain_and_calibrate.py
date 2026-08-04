from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"


@dataclass
class CalibrationResult:
    modality: str
    threshold: float
    margin: float
    accuracy: float
    real_recall: float
    fake_recall: float
    recall_gap_abs: float
    found_target: bool


def _run_command(command: str, extra_env: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[RUN] {command}")
    completed = subprocess.run(command, shell=True, cwd=PROJECT_ROOT, env=env)
    return int(completed.returncode)


def _find_latest(pattern: str) -> Path:
    files = sorted(METRICS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[-1]


def _calc_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, tn, fp, fn


def _metrics_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp, tn, fp, fn = _calc_confusion(y_true, y_pred)
    total = len(y_true)
    accuracy = float((tp + tn) / total) if total else 0.0
    real_recall = float(tn / (tn + fp)) if (tn + fp) else 0.0
    fake_recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    recall_gap_abs = abs(real_recall - fake_recall)
    return {
        "accuracy": accuracy,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "recall_gap_abs": recall_gap_abs,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def calibrate_image(
    csv_path: Path,
    std_limit: float,
    target_accuracy: float,
    max_margin: float,
) -> CalibrationResult:
    df = pd.read_csv(csv_path)
    y = df["truth_label"].astype(int).to_numpy()
    raw = df["raw_probability"].to_numpy(dtype=float)
    mean = df["mean_probability"].to_numpy(dtype=float)
    std = df["std_probability"].to_numpy(dtype=float)

    best_target: tuple[float, float, float, float, dict[str, float]] | None = None
    best_fallback: tuple[float, float, float, float, dict[str, float]] | None = None

    for threshold in np.linspace(0.0, 1.0, 2001):
        final_scores = np.where((raw >= threshold) & ((mean < threshold) | (std > std_limit)), mean, raw)
        for margin in np.linspace(0.0, max_margin, int(max_margin * 1000) + 1):
            pred = (final_scores > (threshold + margin)).astype(int)
            metrics = _metrics_from_preds(y, pred)

            if metrics["accuracy"] >= target_accuracy:
                candidate = (metrics["recall_gap_abs"], -metrics["accuracy"], threshold, margin, metrics)
                if best_target is None or candidate < best_target:
                    best_target = candidate

            fallback = (metrics["accuracy"], -metrics["recall_gap_abs"], threshold, margin, metrics)
            if best_fallback is None or fallback > best_fallback:
                best_fallback = fallback

    if best_target is not None:
        gap, neg_acc, threshold, margin, metrics = best_target
        return CalibrationResult(
            modality="image",
            threshold=float(threshold),
            margin=float(margin),
            accuracy=float(-neg_acc),
            real_recall=float(metrics["real_recall"]),
            fake_recall=float(metrics["fake_recall"]),
            recall_gap_abs=float(gap),
            found_target=True,
        )

    assert best_fallback is not None
    acc, neg_gap, threshold, margin, metrics = best_fallback
    return CalibrationResult(
        modality="image",
        threshold=float(threshold),
        margin=float(margin),
        accuracy=float(acc),
        real_recall=float(metrics["real_recall"]),
        fake_recall=float(metrics["fake_recall"]),
        recall_gap_abs=float(-neg_gap),
        found_target=False,
    )


def calibrate_video(
    csv_path: Path,
    target_accuracy: float,
    max_margin: float,
) -> CalibrationResult:
    df = pd.read_csv(csv_path)
    y = df["truth_label"].astype(int).to_numpy()
    score = df["final_fake_probability"].to_numpy(dtype=float)

    best_target: tuple[float, float, float, float, dict[str, float]] | None = None
    best_fallback: tuple[float, float, float, float, dict[str, float]] | None = None

    for threshold in np.linspace(0.0, 1.0, 2001):
        for margin in np.linspace(0.0, max_margin, int(max_margin * 1000) + 1):
            pred = (score > (threshold + margin)).astype(int)
            metrics = _metrics_from_preds(y, pred)

            if metrics["accuracy"] >= target_accuracy:
                candidate = (metrics["recall_gap_abs"], -metrics["accuracy"], threshold, margin, metrics)
                if best_target is None or candidate < best_target:
                    best_target = candidate

            fallback = (metrics["accuracy"], -metrics["recall_gap_abs"], threshold, margin, metrics)
            if best_fallback is None or fallback > best_fallback:
                best_fallback = fallback

    if best_target is not None:
        gap, neg_acc, threshold, margin, metrics = best_target
        return CalibrationResult(
            modality="video",
            threshold=float(threshold),
            margin=float(margin),
            accuracy=float(-neg_acc),
            real_recall=float(metrics["real_recall"]),
            fake_recall=float(metrics["fake_recall"]),
            recall_gap_abs=float(gap),
            found_target=True,
        )

    assert best_fallback is not None
    acc, neg_gap, threshold, margin, metrics = best_fallback
    return CalibrationResult(
        modality="video",
        threshold=float(threshold),
        margin=float(margin),
        accuracy=float(acc),
        real_recall=float(metrics["real_recall"]),
        fake_recall=float(metrics["fake_recall"]),
        recall_gap_abs=float(-neg_gap),
        found_target=False,
    )


def _save_outputs(results: dict[str, Any], image_result: CalibrationResult, video_result: CalibrationResult) -> tuple[Path, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = METRICS_DIR / f"retrain_calibration_report_{ts}.json"
    env_path = METRICS_DIR / f"retrain_calibrated_thresholds_{ts}.env"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    env_content = "\n".join(
        [
            f"DEEPFAKE_THRESHOLD_IMAGE={image_result.threshold}",
            f"DEEPFAKE_MARGIN_IMAGE={image_result.margin}",
            f"DEEPFAKE_THRESHOLD_VIDEO={video_result.threshold}",
            f"DEEPFAKE_MARGIN_VIDEO={video_result.margin}",
            "",
        ]
    )
    env_path.write_text(env_content, encoding="utf-8")
    return json_path, env_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated retrain + bias-aware threshold calibration")
    parser.add_argument("--run-retrain", action="store_true", help="Run retraining commands before calibration")
    parser.add_argument("--retrain-image-cmd", type=str, default="", help="Shell command to retrain image model")
    parser.add_argument("--retrain-video-cmd", type=str, default="", help="Shell command to retrain video model")
    parser.add_argument("--image-eval-cmd", type=str, default=f"{sys.executable} scripts/eval_image_dataset.py", help="Shell command to evaluate image model")
    parser.add_argument("--video-eval-cmd", type=str, default=f"{sys.executable} scripts/eval_video_dataset.py", help="Shell command to evaluate video model")
    parser.add_argument("--image-test-root", type=str, default="Test_Image_Synth", help="Image dataset root folder name")
    parser.add_argument("--video-test-root", type=str, default="Test_Video_Synth", help="Video dataset root folder name")
    parser.add_argument("--target-accuracy", type=float, default=0.80, help="Required minimum accuracy")
    parser.add_argument("--max-margin", type=float, default=0.20, help="Maximum margin searched during calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.run_retrain:
        if args.retrain_image_cmd:
            code = _run_command(args.retrain_image_cmd)
            if code != 0:
                raise RuntimeError("Image retraining failed")
        if args.retrain_video_cmd:
            code = _run_command(args.retrain_video_cmd)
            if code != 0:
                raise RuntimeError("Video retraining failed")

    image_env = {"DEEPFAKE_IMAGE_TEST_ROOT": args.image_test_root}
    video_env = {"DEEPFAKE_VIDEO_TEST_ROOT": args.video_test_root}

    code = _run_command(args.image_eval_cmd, image_env)
    if code != 0:
        raise RuntimeError("Image evaluation failed")

    code = _run_command(args.video_eval_cmd, video_env)
    if code != 0:
        raise RuntimeError("Video evaluation failed")

    image_summary = _find_latest("image_dataset_eval_summary_*.json")
    video_summary = _find_latest("video_dataset_eval_summary_*.json")

    image_payload = json.loads(image_summary.read_text(encoding="utf-8"))
    video_payload = json.loads(video_summary.read_text(encoding="utf-8"))

    image_csv = Path(image_payload["output_csv"])
    video_csv = Path(video_payload["output_csv"])
    if not image_csv.is_absolute():
        image_csv = PROJECT_ROOT / image_csv
    if not video_csv.is_absolute():
        video_csv = PROJECT_ROOT / video_csv

    image_std_limit = float(image_payload.get("std_limit", 0.12))

    image_result = calibrate_image(image_csv, image_std_limit, args.target_accuracy, args.max_margin)
    video_result = calibrate_video(video_csv, args.target_accuracy, args.max_margin)

    report: dict[str, Any] = {
        "target_accuracy": args.target_accuracy,
        "image": {
            "dataset_root": args.image_test_root,
            "summary_file": str(image_summary),
            "csv_file": str(image_csv),
            "threshold": image_result.threshold,
            "margin": image_result.margin,
            "accuracy": image_result.accuracy,
            "real_recall": image_result.real_recall,
            "fake_recall": image_result.fake_recall,
            "recall_gap_abs": image_result.recall_gap_abs,
            "target_accuracy_met": image_result.found_target,
        },
        "video": {
            "dataset_root": args.video_test_root,
            "summary_file": str(video_summary),
            "csv_file": str(video_csv),
            "threshold": video_result.threshold,
            "margin": video_result.margin,
            "accuracy": video_result.accuracy,
            "real_recall": video_result.real_recall,
            "fake_recall": video_result.fake_recall,
            "recall_gap_abs": video_result.recall_gap_abs,
            "target_accuracy_met": video_result.found_target,
        },
    }

    report_path, env_path = _save_outputs(report, image_result, video_result)

    print("=== RETRAIN + CALIBRATION COMPLETE ===")
    print(f"Target accuracy: {args.target_accuracy:.2f}")
    print(
        "Image => "
        f"acc={image_result.accuracy:.4f}, gap={image_result.recall_gap_abs:.4f}, "
        f"threshold={image_result.threshold:.4f}, margin={image_result.margin:.4f}, "
        f"target_met={image_result.found_target}"
    )
    print(
        "Video => "
        f"acc={video_result.accuracy:.4f}, gap={video_result.recall_gap_abs:.4f}, "
        f"threshold={video_result.threshold:.4f}, margin={video_result.margin:.4f}, "
        f"target_met={video_result.found_target}"
    )
    print(f"Saved report: {report_path}")
    print(f"Saved env thresholds: {env_path}")


if __name__ == "__main__":
    main()
