"""Summarise run CSVs: format compliance, accuracy, per-class / per-subtype
false alarms, and latency (median / p95), one row per (model, prompt).

Usage: uv run python scripts/summarize.py results/step0/*.csv
"""

import json
import sys
from pathlib import Path

import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (model, prompt, res, nf), g in df.groupby(["model", "prompt", "resolution", "n_frames"], sort=False):
        pos = g.label == "person_down"
        pred = g.answer == "yes"
        tp, fp = (pos & pred).sum(), (~pos & pred).sum()
        fn = (pos & ~pred).sum()
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        row = {
            "model": model, "prompt": prompt, "res": res, "frames": nf, "n": len(g),
            "strict_parse": (g.parse_mode == "strict").mean(),
            "unparsed": (g.answer == "unparsed").sum(),
            "acc": ((pred == pos)).mean(),
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "FAR_hard_neg": pred[g.label == "hard_negative"].mean(),
            "FAR_normal": pred[g.label == "normal"].mean(),
            "e2e_med_ms": g.e2e_ms.median(), "e2e_p95_ms": g.e2e_ms.quantile(0.95),
            "ttft_med_ms": g.ttft_ms.median() if "ttft_ms" in g and g.ttft_ms.notna().any() else None,
            "prompt_tok_med": g.prompt_tokens.median(),
        }
        out.append(row)
    return pd.DataFrame(out)


def per_subtype(df: pd.DataFrame) -> pd.DataFrame:
    neg = df[df.label != "person_down"].copy()
    neg["false_alarm"] = neg.answer == "yes"
    return neg.pivot_table(index=["model", "prompt"], columns="subtype", values="false_alarm",
                           aggfunc=lambda s: f"{s.sum()}/{len(s)}")


def memory(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for j in sorted(p.parent.glob("run-*.json")):
            d = json.loads(j.read_text())
            b = d["backend"]
            rows.append({"model": d["model"], "ngl": b.get("n_gpu_layers"), "threads": b.get("threads"),
                         "model_mb": b.get("model_size_mb"), "mmproj_mb": b.get("mmproj_size_mb"),
                         **d["memory"], "power": d["env"].get("power_source"),
                         "load_avg": d["env"].get("load_avg_1m")})
    return pd.DataFrame(rows).drop_duplicates()


if __name__ == "__main__":
    paths = [Path(p) for p in sys.argv[1:]]
    df = pd.concat([pd.read_csv(p).assign(source=p.stem) for p in paths])
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.precision", 3)
    for src, g in df.groupby("source", sort=False):
        print(f"\n=== {src}")
        print(summarize(g).to_string(index=False))
        print("\nfalse alarms by negative subtype:")
        print(per_subtype(g).to_string())
    print("\n=== memory / run config")
    print(memory(paths).to_string(index=False))
