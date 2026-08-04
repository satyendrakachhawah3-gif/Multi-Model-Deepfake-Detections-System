from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float, float, float, float, float]:
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    acc = (tp + tn) / len(y)
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    fake_precision = tp / (tp + fp) if (tp + fp) else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1_real = (2 * real_precision * real_recall / (real_precision + real_recall)) if (real_precision + real_recall) else 0.0
    f1_fake = (2 * fake_precision * fake_recall / (fake_precision + fake_recall)) if (fake_precision + fake_recall) else 0.0
    macro_f1 = (f1_real + f1_fake) / 2
    recall_gap = abs(real_recall - fake_recall)

    return acc, real_precision, fake_precision, real_recall, fake_recall, macro_f1, recall_gap


def main() -> None:
    df = pd.read_csv("outputs/metrics/text_dataset_eval_20260303_223414.csv")
    d = df.dropna(subset=["truth_label"]).copy()

    y = d["truth_label"].astype(int).to_numpy()
    raw = d["raw_probability"].to_numpy(dtype=float)
    mean = d["mean_probability"].to_numpy(dtype=float)
    std = d["std_probability"].to_numpy(dtype=float)

    best = None
    for threshold in np.linspace(0.0, 1.0, 1201):
        for margin in np.linspace(0.0, 0.2, 201):
            final = np.where((raw >= threshold) & ((mean < threshold) | (std > 0.2)), mean, raw)
            pred = (final > (threshold + margin)).astype(int)
            m = compute_metrics(y, pred)
            acc, rp, fp, rr, fr, macro_f1, gap = m
            obj = (macro_f1, acc, -gap, rp + fp + rr + fr)
            row = (obj, threshold, margin, acc, rp, fp, rr, fr, macro_f1, gap)
            if best is None or row[0] > best[0]:
                best = row

    assert best is not None
    _, t, m, acc, rp, fp, rr, fr, mf1, gap = best
    print("best_t,best_m,acc,real_precision,fake_precision,real_recall,fake_recall,macro_f1,recall_gap")
    print(f"{t:.5f},{m:.5f},{acc:.4f},{rp:.4f},{fp:.4f},{rr:.4f},{fr:.4f},{mf1:.4f},{gap:.4f}")


if __name__ == "__main__":
    main()
