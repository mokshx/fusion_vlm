"""Align powermetrics samples with per-request timestamps and compute power and
energy per decision for each configuration.

powermetrics (cpu_power, gpu_power samplers) reports estimated package power of
the CPU, GPU and Neural Engine, not wall power. Each sample covers the preceding
interval; we weight samples by their overlap with each request window. Idle power
is the mean over the idle windows before and after the runs and is subtracted to
get the power attributable to inference.

  uv run python scripts/power_analysis.py
Writes results/power/summary.csv, paper/generated/res_power.tex,
paper/generated/res_power_numbers.tex
"""

import plistlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "results/power"
GEN = ROOT / "paper/generated"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_grid import MODELS, QUANT_ORDER, RES_ORDER, TAG  # noqa: E402


def samples() -> pd.DataFrame:
    """Power samples on the epoch timeline.

    powermetrics stamps each sample with a UTC wall-clock time truncated to whole
    seconds, but reports each sample's exact duration (elapsed_ns). We place
    samples by their cumulative duration and anchor that timeline in absolute time
    with the median offset to the timestamps (+0.5 s for the truncation)."""
    raw = (P / "powermetrics.plist").read_bytes()
    rows, keys_seen = [], set()
    for chunk in raw.split(b"\x00"):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            d = plistlib.loads(chunk)
        except Exception:
            continue  # last sample may be truncated when powermetrics is stopped
        proc = d.get("processor", {})
        keys_seen |= {k for k in proc if "power" in k}
        ts = d.get("timestamp")
        rows.append({"dt": d.get("elapsed_ns", 0) / 1e9,
                     "ts": ts.replace(tzinfo=timezone.utc).timestamp() if isinstance(ts, datetime) else np.nan,
                     "cpu_mw": proc.get("cpu_power", np.nan), "gpu_mw": proc.get("gpu_power", np.nan),
                     "ane_mw": proc.get("ane_power", np.nan), "combined_mw": proc.get("combined_power", np.nan)})
    df = pd.DataFrame(rows)
    if df.combined_mw.isna().all():
        df["combined_mw"] = df[["cpu_mw", "gpu_mw", "ane_mw"]].sum(axis=1, min_count=1)
    cum = df.dt.cumsum()
    resid = df.ts - cum
    offset = float(np.nanmedian(resid)) + 0.5
    df["t_end"] = cum + offset
    df["t_start"] = df.t_end - df.dt
    drift = float(np.nanmedian(resid.iloc[-500:]) - np.nanmedian(resid.iloc[:500]))
    print(f"{len(df)} power samples over {cum.iloc[-1] / 60:.1f} min; keys: {sorted(keys_seen)}; "
          f"clock drift first->last {drift * 1000:.0f} ms")
    return df


def fmt_j(v: float) -> str:
    """Energy with sensible precision: 0.87, 3.22, 13.7, 102."""
    return f"{v:.0f}" if v >= 100 else f"{v:.1f}" if v >= 10 else f"{v:.2f}"


def mean_power(s: pd.DataFrame, windows: list[tuple[float, float]]) -> tuple[float, float]:
    """(time-weighted mean combined power in W, energy in J) over the given windows."""
    energy, total = 0.0, 0.0
    for a, b in windows:
        ov = (np.minimum(s.t_end, b) - np.maximum(s.t_start, a)).clip(lower=0)
        energy += float((ov * s.combined_mw).sum()) / 1000
        total += float(ov.sum())
    return (energy / total if total else np.nan), energy


def main() -> None:
    s = samples()
    m = pd.read_csv(P / "markers.csv")
    idle_w = []
    for label in m.label.unique():
        a = m[(m.label == label) & (m.event == "idle_start")].epoch.iloc[0]
        b = m[(m.label == label) & (m.event == "idle_end")].epoch.iloc[0]
        idle_w.append(mean_power(s, [(a, b)])[0])
    idle = float(np.nanmean(idle_w))
    rows = []
    for dev in ["gpu", "cpu"]:
        path = P / f"{dev}.csv"
        if not path.exists():
            continue
        d = pd.read_csv(path)
        d["resolution"] = d.resolution.astype(str)
        for (spec, res, nf), g in d.groupby(["model", "resolution", "n_frames"]):
            win = list(zip(g.t_start, g.t_end))
            watts, joules = mean_power(s, win)
            busy = sum(b - a for a, b in win)
            rows.append({"device": dev, "spec": spec, "resolution": res, "n_frames": nf, "n": len(g),
                         "power_w": watts, "above_idle_w": watts - idle,
                         "energy_j_per_decision": joules / len(g),
                         "energy_above_idle_j": (watts - idle) * busy / len(g),
                         "median_ms": g.e2e_ms.median()})
    out = pd.DataFrame(rows)
    out["m"] = out.spec.str.split(":").str[0].map(list(MODELS).index)
    out["q"] = out.spec.str.split(":").str[1].map(QUANT_ORDER)
    out["r"] = out.resolution.map(RES_ORDER)
    out = out.sort_values(["device", "m", "q", "r"], ascending=[False, True, True, True])
    out.to_csv(P / "summary.csv", index=False)
    print(f"idle {idle:.2f} W (before/after: {', '.join(f'{w:.2f}' for w in idle_w)})")
    print(out.drop(columns=["m", "q", "r"]).round(3).to_string(index=False))

    lines = [r"\begin{tabular}{llllrrr}", r"\toprule",
             r"Device & Model & Weights & Res. & Power (W) & Above idle (W) & Energy (J/dec.) \\", r"\midrule"]
    prev = None
    qt = {"F16": "Half", "BF16": "Half", "Q8_0": "Qeight", "Q4_K_M": "Qfour"}
    rt = {"native": "Nat", "512": "Fivetwelve", "256": "Twofiftysix"}
    ft = {1: "One", 4: "Four", 8: "Eight"}
    macros = {"PIdleW": f"{idle:.1f}"}
    for _, r in out.iterrows():
        if prev and prev != r.device:
            lines.append(r"\midrule")
        name = MODELS[r.spec.split(":")[0]][0]
        q = r.spec.split(":")[1]
        lines.append(f"{'GPU' if r.device == 'gpu' else 'CPU only'} & {name} & {q.replace('_', chr(92) + '_')} & "
                     f"{r.resolution} & {r.power_w:.1f} & {r.above_idle_w:.1f} & {fmt_j(r.energy_above_idle_j)} \\\\")
        prev = r.device
        k = "P" + TAG[r.spec.split(":")[0]] + qt[q] + rt[r.resolution] + ft[int(r.n_frames)] + r.device.capitalize()
        macros[k + "W"] = f"{r.above_idle_w:.1f}"
        macros[k + "J"] = fmt_j(r.energy_above_idle_j)
    lines += [r"\bottomrule", r"\end{tabular}"]
    g, c = out[out.device == "gpu"], out[out.device == "cpu"]
    macros["PGpuWMin"], macros["PGpuWMax"] = f"{g.above_idle_w.min():.0f}", f"{g.above_idle_w.max():.0f}"
    macros["PCpuWMin"], macros["PCpuWMax"] = f"{c.above_idle_w.min():.0f}", f"{c.above_idle_w.max():.0f}"
    both = g.merge(c, on=["spec", "resolution", "n_frames"], suffixes=("_g", "_c"))
    ratio = both.energy_above_idle_j_c / both.energy_above_idle_j_g
    macros["PCpuGpuRatioMin"], macros["PCpuGpuRatioMax"] = f"{ratio.min():.0f}", f"{ratio.max():.0f}"
    macros["PIdleW"] = f"{idle:.2f}"
    (GEN / "res_power.tex").write_text("\n".join(lines) + "\n")
    (GEN / "res_power_numbers.tex").write_text("% Generated by scripts/power_analysis.py -- do not edit.\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))


if __name__ == "__main__":
    main()
