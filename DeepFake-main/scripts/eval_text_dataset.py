from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime
import sys
import os
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.text_detector import TextDetector


SAMPLES = [
    {"id": 1, "label": 0, "group": "real_news", "text": "The city council approved a $2 million budget for road repairs in the downtown area during last night's meeting."},
    {"id": 2, "label": 0, "group": "real_news", "text": "Local high school students raised $5,000 for the animal shelter through their annual charity bake sale."},
    {"id": 3, "label": 0, "group": "real_news", "text": "Temperatures are expected to reach 85 degrees on Saturday with a slight chance of afternoon thunderstorms."},
    {"id": 4, "label": 0, "group": "real_news", "text": "The museum will extend its hours during the summer months to accommodate visitors from 9 AM to 8 PM."},
    {"id": 5, "label": 0, "group": "real_news", "text": "Three candidates are running for mayor in the upcoming election, with polls showing a close race."},
    {"id": 6, "label": 0, "group": "real_social", "text": "Just finished my morning run! 5 miles completed and feeling great! 🏃‍♂️ #fitness #morningroutine"},
    {"id": 7, "label": 0, "group": "real_social", "text": "Made homemade pasta for the first time today. It was messy but delicious! 🍝"},
    {"id": 8, "label": 0, "group": "real_social", "text": "Can't believe my daughter took her first steps today. So proud! 👶❤️"},
    {"id": 9, "label": 0, "group": "real_social", "text": "This book is amazing! Halfway through and can't put it down. Highly recommend!"},
    {"id": 10, "label": 0, "group": "real_social", "text": "Rainy day = perfect for staying in with coffee and a movie. ☕🎬"},
    {"id": 11, "label": 0, "group": "real_professional", "text": "Please find attached the quarterly sales report for your review. Let's discuss on Monday."},
    {"id": 12, "label": 0, "group": "real_professional", "text": "Thank you for your application. We will contact you within 5-7 business days to schedule an interview."},
    {"id": 13, "label": 0, "group": "real_professional", "text": "The meeting has been rescheduled to 3 PM due to a scheduling conflict."},
    {"id": 14, "label": 0, "group": "real_professional", "text": "Our team achieved 95% customer satisfaction this month, exceeding our target."},
    {"id": 15, "label": 0, "group": "real_professional", "text": "We regret to inform you that the event has been cancelled due to unforeseen circumstances."},
    {"id": 16, "label": 1, "group": "fake_clickbait", "text": "BREAKING: Scientists discover that drinking lemon water cures all diseases - Big Pharma doesn't want you to know!"},
    {"id": 17, "label": 1, "group": "fake_clickbait", "text": "You won't believe what this mom found in her backyard - the video will shock you!"},
    {"id": 18, "label": 1, "group": "fake_clickbait", "text": "Doctors hate this simple trick that melts belly fat while you sleep!"},
    {"id": 19, "label": 1, "group": "fake_clickbait", "text": "This 12-year-old genius invents device that could end world hunger - watch the inspiring story!"},
    {"id": 20, "label": 1, "group": "fake_clickbait", "text": "Government officials caught in massive cover-up - read before it's deleted!"},
    {"id": 21, "label": 1, "group": "fake_social", "text": "🌟 TRANSFORM YOUR LIFE IN 7 DAYS! 🌟 My quantum manifestation technique attracted $50,000 in just one week! Click link in bio for FREE masterclass! #manifestation #wealth #lawofattraction"},
    {"id": 22, "label": 1, "group": "fake_social", "text": "URGENT: Facebook will start charging $9.99/month tomorrow! Share this post to keep your account FREE forever! ⚠️"},
    {"id": 23, "label": 1, "group": "fake_social", "text": "This crypto hack made me a millionaire at 22 - and it's completely legal! DM me for the secret! 💰🚀"},
    {"id": 24, "label": 1, "group": "fake_social", "text": "Miracle water filter removes 100% of toxins - thousands of doctors are saying this is a scam but we have PROOF!"},
    {"id": 25, "label": 1, "group": "fake_social", "text": "I cured my anxiety in 3 days using this ancient technique - the secret they don't want you to know! 🧠"},
    {"id": 26, "label": 1, "group": "fake_scam", "text": "Dear valued customer, your account has been compromised. Click here immediately to verify your password and banking details."},
    {"id": 27, "label": 1, "group": "fake_scam", "text": "CONGRATULATIONS! You've won $5,000,000 in the international lottery! Send $500 processing fee to claim your prize."},
    {"id": 28, "label": 1, "group": "fake_scam", "text": "I am a prince from Nigeria needing help transferring $50 million out of my country. Share your bank details for 30% commission."},
    {"id": 29, "label": 1, "group": "fake_scam", "text": "Your Netflix subscription will be cancelled in 24 hours unless you update your payment information via this link."},
    {"id": 30, "label": 1, "group": "fake_scam", "text": "Urgent: IRS has detected fraudulent activity on your tax return. Call this number immediately to avoid arrest."},
    {"id": 31, "label": 1, "group": "fake_academic", "text": "In the annals of human achievement, the confluence of technological innovation and societal evolution represents a paradigm shift of unprecedented magnitude that will fundamentally reshape our collective destiny."},
    {"id": 32, "label": 1, "group": "fake_academic", "text": "Leveraging synergistic paradigm shifts in cross-platform optimization, our organization will revolutionize vertical integration strategies moving forward."},
    {"id": 33, "label": 1, "group": "fake_academic", "text": "The utilization of quantum-based holistic approaches to wellness optimization creates transformative opportunities for self-actualization and metaphysical alignment."},
    {"id": 34, "label": 1, "group": "fake_academic", "text": "Our proprietary algorithm utilizes blockchain-enabled AI to revolutionize the synergistic potential of next-generation cryptocurrency ecosystems."},
    {"id": 35, "label": 1, "group": "fake_academic", "text": "Through the application of neuro-linguistic programming and quantum consciousness techniques, you can unlock 1000% of your brain's potential."},
    {"id": 36, "label": None, "group": "borderline", "text": "I wrote most of this myself but used AI to make it sound better: The sunset was pretty and I felt happy. Actually, the golden hour cast its ethereal glow upon the horizon as I experienced a profound sense of tranquility."},
    {"id": 37, "label": None, "group": "borderline", "text": "Human draft: 'Product works well.' AI polished: 'This exceptional product delivers outstanding performance that consistently exceeds expectations in every conceivable metric.'"},
    {"id": 38, "label": None, "group": "borderline", "text": "Original: 'The company had a good quarter.' Modified: 'The company experienced unprecedented success this quarter.'"},
    {"id": 39, "label": None, "group": "borderline", "text": "Original: 'Some customers complained.' Modified: 'A vocal minority expressed concerns.'"},
    {"id": 40, "label": None, "group": "borderline", "text": "Original: 'Sales increased slightly.' Modified: 'Sales skyrocketed to record-breaking levels.'"},
]


def metrics(binary_truth: list[int], binary_pred: list[int]) -> dict[str, float]:
    total = len(binary_truth)
    tp = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(binary_truth, binary_pred) if y == 1 and p == 0)

    accuracy = (tp + tn) / total if total else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    fake_precision = tp / (tp + fp) if (tp + fp) else 0.0

    return {
        "accuracy": accuracy,
        "tp_fake": tp,
        "tn_real": tn,
        "fp_real_as_fake": fp,
        "fn_fake_as_real": fn,
        "real_recall": real_recall,
        "fake_recall": fake_recall,
        "recall_gap_abs": abs(real_recall - fake_recall),
        "real_precision": real_precision,
        "fake_precision": fake_precision,
    }


def _with_default_text(sample: dict[str, Any]) -> str:
    text = sample.get("text")
    return str(text).strip() if text is not None else ""


def _load_unseen_samples(dataset_dir: Path, start_id: int) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    if not dataset_dir.exists():
        return samples

    id_counter = start_id
    mappings = [("Real", 0, "unseen_real"), ("Fake", 1, "unseen_fake")]
    for folder_name, label, group in mappings:
        folder = dataset_dir / folder_name
        if not folder.exists():
            continue

        for text_file in sorted(folder.glob("*.txt")):
            content = text_file.read_text(encoding="utf-8", errors="ignore").strip()
            if not content:
                continue
            samples.append(
                {
                    "id": id_counter,
                    "label": label,
                    "group": group,
                    "text": content,
                    "source": "unseen_folder",
                    "source_file": str(text_file),
                }
            )
            id_counter += 1

    return samples


def _build_eval_samples(
    score_borderline: bool,
    borderline_label: int,
    include_unseen: bool,
    unseen_dataset_dir: Path,
) -> list[dict[str, Any]]:
    eval_samples: list[dict[str, Any]] = []
    for sample in SAMPLES:
        item = dict(sample)
        item["source"] = "curated_base"
        item["source_file"] = "inline"

        if item.get("label") is None:
            if score_borderline:
                item["label"] = borderline_label
                item["group"] = "borderline_scored"
            else:
                item["group"] = "borderline_unscored"

        eval_samples.append(item)

    if include_unseen:
        start_id = max(int(sample["id"]) for sample in eval_samples) + 1
        eval_samples.extend(_load_unseen_samples(unseen_dataset_dir, start_id))

    return eval_samples


def get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text model on curated + unseen datasets")
    parser.add_argument(
        "--score-borderline",
        action="store_true",
        default=True,
        help="Include borderline samples in scored metrics (default: enabled)",
    )
    parser.add_argument(
        "--no-score-borderline",
        action="store_true",
        help="Exclude borderline samples from scored metrics",
    )
    parser.add_argument(
        "--borderline-label",
        choices=["real", "fake"],
        default="fake",
        help="Label to assign when scoring borderline samples (default: fake)",
    )
    parser.add_argument(
        "--include-unseen",
        action="store_true",
        default=True,
        help="Include unseen text files from dataset directory (default: enabled)",
    )
    parser.add_argument(
        "--no-include-unseen",
        action="store_true",
        help="Disable unseen dataset inclusion",
    )
    parser.add_argument(
        "--unseen-dataset-dir",
        type=str,
        default="Test_text",
        help="Path to unseen text dataset with Real/Fake folders (default: Test_text)",
    )
    args = parser.parse_args()

    score_borderline = bool(args.score_borderline and not args.no_score_borderline)
    include_unseen = bool(args.include_unseen and not args.no_include_unseen)
    borderline_label = 1 if args.borderline_label == "fake" else 0
    unseen_dataset_dir = Path(args.unseen_dataset_dir)

    detector = TextDetector()
    text_threshold = get_env_float("DEEPFAKE_THRESHOLD_TEXT", 0.954)
    text_margin = get_env_float("DEEPFAKE_MARGIN_TEXT", 0.0275)
    text_std_limit = get_env_float("DEEPFAKE_TEXT_STD_LIMIT", 0.2)

    eval_samples = _build_eval_samples(
        score_borderline=score_borderline,
        borderline_label=borderline_label,
        include_unseen=include_unseen,
        unseen_dataset_dir=unseen_dataset_dir,
    )

    clear_samples = [sample for sample in SAMPLES if sample["label"] is not None]
    borderline_samples = [sample for sample in SAMPLES if sample["label"] is None]
    scored_samples = [sample for sample in eval_samples if sample["label"] is not None]
    unseen_samples = [sample for sample in eval_samples if sample.get("source") == "unseen_folder"]

    rows = []
    clear_truth: list[int] = []
    clear_pred: list[int] = []

    for sample in eval_samples:
        sample_text = _with_default_text(sample)
        consistency = detector.predict_with_consistency(sample_text)
        raw_prob = float(consistency["score"])
        mean_prob = float(consistency["mean_score"])
        std_prob = float(consistency["std_score"])

        if raw_prob >= text_threshold and (mean_prob < text_threshold or std_prob > text_std_limit):
            final_prob = mean_prob
        else:
            final_prob = raw_prob

        pred_label = 1 if final_prob > (text_threshold + text_margin) else 0
        rows.append(
            {
                "id": sample["id"],
                "group": sample["group"],
                "truth_label": sample["label"],
                "source": sample.get("source", "unknown"),
                "source_file": sample.get("source_file", "inline"),
                "fake_probability": final_prob,
                "raw_probability": raw_prob,
                "mean_probability": mean_prob,
                "std_probability": std_prob,
                "pred_label": pred_label,
                "pred_text": "FAKE" if pred_label == 1 else "REAL",
                "text": sample_text,
            }
        )
        if sample["label"] is not None:
            clear_truth.append(int(sample["label"]))
            clear_pred.append(pred_label)

    summary_scored = metrics(clear_truth, clear_pred)

    core_rows = [row for row in rows if row["source"] == "curated_base" and row["group"] != "borderline_scored" and row["truth_label"] is not None]
    core_truth = [int(row["truth_label"]) for row in core_rows]
    core_pred = [int(row["pred_label"]) for row in core_rows]
    summary_core = metrics(core_truth, core_pred)

    unseen_rows = [row for row in rows if row["source"] == "unseen_folder" and row["truth_label"] is not None]
    unseen_truth = [int(row["truth_label"]) for row in unseen_rows]
    unseen_pred = [int(row["pred_label"]) for row in unseen_rows]
    summary_unseen = metrics(unseen_truth, unseen_pred) if unseen_rows else None

    borderline_predictions = [
        {
            "id": row["id"],
            "fake_probability": row["fake_probability"],
            "raw_probability": row["raw_probability"],
            "mean_probability": row["mean_probability"],
            "std_probability": row["std_probability"],
            "pred_text": row["pred_text"],
            "text": row["text"],
        }
        for row in rows
        if row["group"] == "borderline"
    ]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("outputs") / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"text_dataset_eval_{ts}.csv"
    json_path = out_dir / f"text_dataset_eval_summary_{ts}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "group",
                "truth_label",
                "source",
                "source_file",
                "fake_probability",
                "raw_probability",
                "mean_probability",
                "std_probability",
                "pred_label",
                "pred_text",
                "text",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "sample_count_total": len(eval_samples),
        "sample_count_scored": len(scored_samples),
        "sample_count_clear_labeled": len(clear_samples),
        "sample_count_borderline": len(borderline_samples),
        "sample_count_unseen": len(unseen_samples),
        "score_borderline": score_borderline,
        "borderline_label_assigned": "fake" if borderline_label == 1 else "real",
        "include_unseen": include_unseen,
        "unseen_dataset_dir": str(unseen_dataset_dir),
        "threshold": text_threshold,
        "margin": text_margin,
        "std_limit": text_std_limit,
        "metrics_scored_set": summary_scored,
        "metrics_clear_set": summary_scored,
        "metrics_core_clear_set": summary_core,
        "metrics_unseen_set": summary_unseen,
        "borderline_predictions": borderline_predictions,
        "output_csv": str(csv_path),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=== TEXT MODEL EVALUATION COMPLETE ===")
    print(
        f"Total samples: {len(eval_samples)} "
        f"(scored={len(scored_samples)}, base_clear={len(clear_samples)}, "
        f"borderline={len(borderline_samples)}, unseen={len(unseen_samples)})"
    )
    print(f"Accuracy (scored set): {summary_scored['accuracy']:.4f}")
    print(f"Real recall (scored set): {summary_scored['real_recall']:.4f}")
    print(f"Fake recall (scored set): {summary_scored['fake_recall']:.4f}")
    print(f"Recall gap |real-fake| (scored set): {summary_scored['recall_gap_abs']:.4f}")
    print(f"FP (real->fake, scored set): {summary_scored['fp_real_as_fake']}")
    print(f"FN (fake->real, scored set): {summary_scored['fn_fake_as_real']}")
    if summary_unseen is not None:
        print(f"Unseen set accuracy: {summary_unseen['accuracy']:.4f}")
    print(f"Saved detailed CSV: {csv_path}")
    print(f"Saved summary JSON: {json_path}")


if __name__ == "__main__":
    main()
