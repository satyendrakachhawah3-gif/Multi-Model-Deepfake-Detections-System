from __future__ import annotations

import numpy as np
import pandas as pd


def calc(y: np.ndarray, pred: np.ndarray):
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    acc = (tp + tn) / len(y)
    real_precision = tn / (tn + fn) if (tn + fn) else 0.0
    fake_precision = tp / (tp + fp) if (tp + fp) else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) else 0.0
    gap = abs(real_recall - fake_recall)
    f1_real = (2 * real_precision * real_recall / (real_precision + real_recall)) if (real_precision + real_recall) else 0.0
    f1_fake = (2 * fake_precision * fake_recall / (fake_precision + fake_recall)) if (fake_precision + fake_recall) else 0.0
    macro_f1 = (f1_real + f1_fake) / 2
    avg4 = (real_precision + fake_precision + real_recall + fake_recall) / 4
    return acc, real_precision, fake_precision, real_recall, fake_recall, gap, macro_f1, avg4


def main() -> None:
    d = pd.read_csv("outputs/metrics/text_dataset_eval_20260303_223414.csv").dropna(subset=["truth_label"])
    y = d["truth_label"].astype(int).to_numpy()
    raw = d["raw_probability"].to_numpy(dtype=float)
    mean = d["mean_probability"].to_numpy(dtype=float)
    std = d["std_probability"].to_numpy(dtype=float)

    best = None
    for t in np.linspace(0.0, 1.0, 2001):
        for m in np.linspace(0.0, 0.2, 401):
            final = np.where((raw >= t) & ((mean < t) | (std > 0.2)), mean, raw)
            pred = (final > (t + m)).astype(int)
            acc, rp, fp, rr, fr, gap, mf1, avg4 = calc(y, pred)
            if gap <= 0.10:
                row = (avg4, mf1, acc, -gap, t, m, rp, fp, rr, fr)
                if best is None or row > best:
                    best = row

    if best is None:
        print("No solution under gap<=0.10")
        return

    print("avg4,macro_f1,acc,gap,t,m,real_precision,fake_precision,real_recall,fake_recall")
    print(f"{best[0]:.4f},{best[1]:.4f},{best[2]:.4f},{-best[3]:.4f},{best[4]:.5f},{best[5]:.5f},{best[6]:.4f},{best[7]:.4f},{best[8]:.4f},{best[9]:.4f}")


if __name__ == "__main__":
    main()
