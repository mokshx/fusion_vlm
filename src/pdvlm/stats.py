"""Confidence intervals for the paper's metrics.

Frames from the same source sequence are correlated, so we resample whole
sequences (cluster bootstrap) rather than individual frames. With one frame per
sequence this reduces to the ordinary bootstrap.

Every metric we report is a ratio of sums of per-sequence counts (recall =
TP/(TP+FN), FAR = FP/N, precision = TP/(TP+FP), F1 = 2TP/(2TP+FP+FN)), so a
resample is a matrix product of resampling counts with per-sequence counts.
"""

import numpy as np
import pandas as pd

N_BOOT = 10_000


def per_cluster_counts(df: pd.DataFrame, cluster_col: str = "sequence") -> pd.DataFrame:
    """TP/FP/FN/TN and per-label negative counts per cluster from `label`/`answer` columns."""
    pos = df.label == "person_down"
    yes = df.answer == "yes"
    c = pd.DataFrame({
        cluster_col: df[cluster_col],
        "tp": pos & yes, "fn": pos & ~yes, "fp": ~pos & yes, "tn": ~pos & ~yes,
        "fp_hard": (df.label == "hard_negative") & yes, "n_hard": df.label == "hard_negative",
        "fp_normal": (df.label == "normal") & yes, "n_normal": df.label == "normal",
    })
    return c.groupby(cluster_col, sort=False).sum().astype(float)


METRICS = {
    "recall": (lambda c: c.tp, lambda c: c.tp + c.fn),
    "precision": (lambda c: c.tp, lambda c: c.tp + c.fp),
    "f1": (lambda c: 2 * c.tp, lambda c: 2 * c.tp + c.fp + c.fn),
    "far_hard": (lambda c: c.fp_hard, lambda c: c.n_hard),
    "far_normal": (lambda c: c.fp_normal, lambda c: c.n_normal),
}


def bootstrap_ci(counts: pd.DataFrame, metric: str, n_boot: int = N_BOOT,
                 level: float = 0.95, seed: int = 0) -> tuple[float, float, float]:
    """(point estimate, lower, upper) percentile CI for a ratio-of-sums metric."""
    num_f, den_f = METRICS[metric]
    num, den = num_f(counts).to_numpy(), den_f(counts).to_numpy()
    rng = np.random.default_rng(seed)
    k = len(counts)
    # w[b, j] = how many times cluster j appears in resample b
    w = np.zeros((n_boot, k))
    idx = rng.integers(0, k, (n_boot, k))
    np.add.at(w, (np.repeat(np.arange(n_boot), k), idx.ravel()), 1)
    bnum, bden = w @ num, w @ den
    vals = np.divide(bnum, bden, out=np.full(n_boot, np.nan), where=bden > 0)
    vals = vals[~np.isnan(vals)]
    a = (1 - level) / 2
    point = num.sum() / den.sum() if den.sum() else float("nan")
    return float(point), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Wilson score interval; used instead of the bootstrap when k == 0 or k == n,
    where the percentile bootstrap collapses to a zero-width interval."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))
