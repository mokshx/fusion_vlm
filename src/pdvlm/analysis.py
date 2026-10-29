"""Metrics for one configuration's predictions (rows of a run CSV).

Operating points:
  default     the generated answer ("yes" = alarm)
  calibrated  p_yes >= t, with t chosen on the dev split as the largest value
              that keeps at least TARGET_RECALL of dev person_down samples

All test-set rates come with 95% CIs (cluster bootstrap over source sequences;
Wilson interval when the rate is 0 or 1).
"""

import numpy as np
import pandas as pd

from pdvlm.stats import bootstrap_ci, per_cluster_counts, wilson_ci

TARGET_RECALL = 0.95
HARD_SUBTYPES = ["crouching", "kneeling", "bending", "sitting_chair", "sitting_furniture", "lying_furniture"]


def auroc(df: pd.DataFrame) -> float:
    pos = df[df.label == "person_down"].p_yes.to_numpy()
    neg = df[df.label != "person_down"].p_yes.to_numpy()
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def calibrate(dev: pd.DataFrame, target: float = TARGET_RECALL) -> float:
    pos = np.sort(dev[dev.label == "person_down"].p_yes.to_numpy())
    k = int(np.floor((1 - target) * len(pos)))  # positives allowed below the threshold
    return float(pos[k])


def with_ci(counts: pd.DataFrame, metric: str, k: int, n: int) -> tuple[float, float, float]:
    if n and k in (0, n) and metric in ("recall", "far_hard", "far_normal"):
        lo, hi = wilson_ci(k, n)
        return k / n, lo, hi
    return bootstrap_ci(counts, metric)


def operating_point(test: pd.DataFrame, alarm: pd.Series) -> dict:
    d = test.assign(answer=np.where(alarm, "yes", "no"))
    c = per_cluster_counts(d)
    pos, hard, norm = d.label == "person_down", d.label == "hard_negative", d.label == "normal"
    yes = d.answer == "yes"
    out = {
        "recall": with_ci(c, "recall", int((pos & yes).sum()), int(pos.sum())),
        "far_hard": with_ci(c, "far_hard", int((hard & yes).sum()), int(hard.sum())),
        "far_normal": with_ci(c, "far_normal", int((norm & yes).sum()), int(norm.sum())),
        "f1": bootstrap_ci(c, "f1"),
        "precision": bootstrap_ci(c, "precision"),
    }
    out["far_subtype"] = {s: float(yes[d.subtype == s].mean()) if (d.subtype == s).any() else float("nan")
                          for s in HARD_SUBTYPES}
    out["recall_viewpoint"] = {v: float(yes[pos & (d.viewpoint == v)].mean())
                               for v in sorted(d[pos].viewpoint.unique())}
    out["far_viewpoint"] = {v: float(yes[~pos & (d.viewpoint == v)].mean())
                            for v in sorted(d[~pos].viewpoint.unique())}
    return out


def evaluate(cfg: pd.DataFrame) -> dict:
    """All metrics for one configuration (test and dev rows of one model/prompt/res/frames)."""
    test, dev = cfg[cfg.split == "test"], cfg[cfg.split == "dev"]
    t = calibrate(dev)
    return {
        "n_test": len(test),
        "auroc": auroc(test),
        "auroc_dev": auroc(dev),
        "threshold": t,
        "default": operating_point(test, test.answer == "yes"),
        "calibrated": operating_point(test, test.p_yes >= t),
        "strict_parse": float((cfg.parse_mode == "strict").mean()),
        "unparsed": int((cfg.answer == "unparsed").sum()),
        "lat_median_ms": float(test.e2e_ms.median()),
        "lat_p95_ms": float(test.e2e_ms.quantile(0.95)),
        "prompt_tokens": float(test.prompt_tokens.median()),
    }
