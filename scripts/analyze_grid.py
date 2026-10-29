"""Turn results/grid/*.csv into the paper's results tables, numbers and figures.

Works on partial results (missing stages are skipped). Every number in the
results section comes from here.

  uv run python scripts/analyze_grid.py
Writes results/grid/summary.csv, paper/generated/res_*.tex, paper/generated/res_numbers.tex,
paper/figures/pareto_{latency,memory}.pdf
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pdvlm.analysis import HARD_SUBTYPES, evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GRID = ROOT / "results/grid"
GEN = ROOT / "paper/generated"
FIG = ROOT / "paper/figures"

MODELS = {  # spec prefix -> (display name, family, total params in B)
    "lfm-450m": ("LFM2.5-VL-450M", "LFM", 0.45),
    "lfm-1.6b": ("LFM2.5-VL-1.6B", "LFM", 1.6),
    "lfm-3b": ("LFM2.5-VL-3B", "LFM", 3.0),
    "qwen3.5-2b": ("Qwen3.5-2B", "Qwen", 2.27),
    "qwen3.5-4b": ("Qwen3.5-4B", "Qwen", 4.66),
    "yolo11n-pose-rule": ("YOLO11n-pose + rule", "Pose", 0.003),
    "yolo11s-pose-rule": ("YOLO11s-pose + rule", "Pose", 0.010),
}
QUANT_ORDER = {"F16": 0, "BF16": 0, "Q8_0": 1, "Q4_K_M": 2, "pose": 3}
RES_ORDER = {"native": 0, "512": 1, "256": 2}
# Periods when the Mac ran on battery (pmset log); timings overlapping them are re-measured.
BATTERY = [("2026-10-26 19:44:32", "2026-10-26 20:25:05")]

# Validated categorical slots (dataviz reference palette, light surface; all-pairs PASS).
FAMILY_COLOR = {"LFM": "#2a78d6", "Qwen": "#eb6834"}
QUANT_MARKER = {"16-bit": "s", "Q8_0": "o", "Q4_K_M": "^"}
INK, INK2, GRID_C, RECESSIVE = "#0b0b0b", "#52514e", "#e7e6e2", "#bdbcb6"


def load() -> pd.DataFrame:
    frames = [pd.read_csv(p).assign(stage=p.stem) for p in sorted(GRID.glob("stage*.csv"))]
    base = ROOT / "results/baseline/rows.csv"
    if base.exists():
        frames.append(pd.read_csv(base).assign(stage="baseline"))
    df = pd.concat(frames, ignore_index=True)
    df["resolution"] = df.resolution.astype(str)
    # Labels come from the current label files (they may have been corrected by the
    # author's by-eye check after the runs); frames removed there are dropped here.
    lab = pd.concat([pd.read_csv(ROOT / f"data/benchmark/v1/labels_{s}.csv") for s in ("test", "dev")])
    lab = lab.set_index("sample_id")[["split", "label", "subtype", "viewpoint"]]
    df = df[df.sample_id.isin(lab.index)].copy()
    for c in lab.columns:
        df[c] = df.sample_id.map(lab[c])
    return df


def run_info() -> dict[str, dict]:
    return {json.loads(p.read_text())["run_id"]: json.loads(p.read_text()) for p in GRID.glob("run-*.json")}


def config_windows() -> list[tuple[str, int, str, datetime, datetime]]:
    """(spec, n_frames, resolution, start, end) for each config from grid.log.
    The log has times only; the date advances whenever the clock goes backwards."""
    day = datetime.fromisoformat("2026-10-26")
    events, prev = [], None  # (time, spec or None for 'grid finished', n, res)
    for line in (GRID / "grid.log").read_text().splitlines():
        m = re.match(r"(\d\d):(\d\d):(\d\d) ", line)
        if not m:
            continue
        t = day.replace(hour=int(m.group(1)), minute=int(m.group(2)), second=int(m.group(3)))
        if prev and t < prev:
            day = day.replace(day=day.day + 1)
            t = t.replace(day=day.day)
        prev = t
        c = re.search(r"--models (\S+).*--n-frames (\d+)", line)
        r = re.search(r"--resolution (\d+)", line)
        events.append((t, c.group(1) if c else None, int(c.group(2)) if c else 0,
                       (r.group(1) if r else "native") if c else ""))
    out = []
    for i, (t0, spec, n, res) in enumerate(events):
        if spec is None:
            continue
        t1 = events[i + 1][0] if i + 1 < len(events) else datetime.now()
        out.append((spec, n, res, t0, t1))
    return out


def on_battery(spec: str, n: int, res: str, windows) -> bool:
    for s, nn, rr, t0, t1 in windows:
        if (s, nn, rr) == (spec, n, res):
            for b0, b1 in BATTERY:
                if t0 < datetime.fromisoformat(b1) and t1 > datetime.fromisoformat(b0):
                    return True
    return False


def apply_retime(s: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Replace latencies of battery-affected configs with re-measured ones (same test frames)."""
    path = GRID / "retime.csv"
    if not path.exists():
        return s
    rt = pd.read_csv(path)
    rt["resolution"] = rt.resolution.astype(str)
    for (spec, prompt, res, nf), g in rt.groupby(["model", "prompt", "resolution", "n_frames"]):
        m = (s.spec == spec) & (s.prompt == prompt) & (s.resolution == res) & (s.n_frames == nf)
        if not m.any():
            continue
        orig = df[(df.model == spec) & (df.prompt == prompt) & (df.resolution == res) & (df.n_frames == nf)]
        same = g.merge(orig, on="sample_id", suffixes=("", "_o"))
        s.loc[m, "lat_median_ms"] = g.e2e_ms.median()
        s.loc[m, "lat_p95_ms"] = g.e2e_ms.quantile(0.95)
        s.loc[m, "streams_1hz"] = int(1000 // g.e2e_ms.quantile(0.95))
        s.loc[m, "battery"] = False
        s.loc[m, "retimed"] = True
        s.loc[m, "retime_same_answers"] = f"{int((same.answer == same.answer_o).sum())}/{len(same)}"
    return s


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    runs, windows, rows = run_info(), config_windows(), []
    n_full = df.drop_duplicates("sample_id").shape[0]
    for (spec, prompt, res, nf), g in df.groupby(["model", "prompt", "resolution", "n_frames"]):
        if g.sample_id.nunique() < n_full:  # still running
            print(f"skipping incomplete {spec} {prompt} res={res} frames={nf}: {g.sample_id.nunique()}/{n_full}")
            continue
        name, family, params = MODELS[spec.split(":")[0]]
        quant = spec.split(":")[1]
        ev = evaluate(g)
        info = [runs[r] for r in g.run_id.unique() if r in runs]
        rss = max((i["memory"].get("peak_rss_mb", math.nan) for i in info), default=math.nan)
        disk = info[0]["backend"]["model_size_mb"] + info[0]["backend"]["mmproj_size_mb"] if info else math.nan
        row = {"spec": spec, "model": name, "family": family, "params_b": params, "quant": quant,
               "quant_bits": "16-bit" if quant in ("F16", "BF16") else quant,
               "is_baseline": family == "Pose",
               "prompt": prompt, "resolution": res, "n_frames": nf,
               "battery": on_battery(spec, nf, res, windows),
               "auroc": ev["auroc"], "auroc_dev": ev["auroc_dev"], "threshold": ev["threshold"],
               "lat_median_ms": ev["lat_median_ms"], "lat_p95_ms": ev["lat_p95_ms"],
               "prompt_tokens": ev["prompt_tokens"], "peak_rss_gib": rss / 1024, "disk_mib": disk,
               "streams_1hz": int(1000 // ev["lat_p95_ms"]), "unparsed": ev["unparsed"], "n_test": ev["n_test"]}
        for op in ["default", "calibrated"]:
            for k in ["recall", "far_hard", "far_normal", "f1", "precision"]:
                v, lo, hi = ev[op][k]
                row[f"{op}_{k}"], row[f"{op}_{k}_lo"], row[f"{op}_{k}_hi"] = v, lo, hi
            for s, v in ev[op]["far_subtype"].items():
                row[f"{op}_far_{s}"] = v
            for vp, v in ev[op]["recall_viewpoint"].items():
                row[f"{op}_recall_{vp}"] = v
        if family == "Pose":  # rule already tuned on dev: its decision is the calibrated operating point
            for c in [c for c in row if c.startswith("default_")]:
                row["calibrated_" + c.removeprefix("default_")] = row[c]
            bt = json.loads((ROOT / "results/baseline/timing.json").read_text()).get(
                spec.split(":")[0].removesuffix("-rule") + ".pt", {})
            row["peak_rss_gib"] = bt.get("rss_gib", math.nan)
        rows.append(row)
    s = pd.DataFrame(rows)
    s["m_order"] = s.spec.str.split(":").str[0].map(list(MODELS).index)
    s["q_order"] = s.quant.map(QUANT_ORDER)
    s["r_order"] = s.resolution.map(RES_ORDER)
    return s.sort_values(["m_order", "prompt", "n_frames", "q_order", "r_order"]).reset_index(drop=True)


# ---------------------------------------------------------------- tables
def pct(v: float) -> str:
    return "--" if pd.isna(v) else f"{100 * v:.0f}"


def ci(row, key: str) -> str:
    return f"{pct(row[key])} \\scriptsize[{pct(row[key + '_lo'])}, {pct(row[key + '_hi'])}]"


def table_prompts(s: pd.DataFrame) -> str:
    t = s[(s.quant == "Q8_0") & (s.resolution == "native") & (s.n_frames == 1)]
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Model & Prompt & Recall & FAR$_\text{hard}$ & FAR$_\text{normal}$ & F1 & AUROC \\", r"\midrule"]
    prev = None
    for _, r in t.iterrows():
        if prev and prev != r.model:
            lines.append(r"\midrule")
        lines.append(f"{r.model if prev != r.model else ''} & {r.prompt} & {pct(r.default_recall)} & "
                     f"{pct(r.default_far_hard)} & {pct(r.default_far_normal)} & {r.default_f1:.2f} & {r.auroc:.3f} \\\\")
        prev = r.model
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_main(s: pd.DataFrame) -> str:
    t = s[(s.quant == "Q8_0") & (s.resolution == "native") & (s.n_frames == 1) & (s.prompt == "A")]
    lines = [r"\begin{tabular}{lrrrrrrr}", r"\toprule",
             r" & \multicolumn{3}{c}{Default answer} & \multicolumn{3}{c}{Calibrated threshold} & \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Model & Recall & FAR$_\text{hard}$ & F1 & Recall & FAR$_\text{hard}$ & F1 & AUROC \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.model} & {ci(r, 'default_recall')} & {ci(r, 'default_far_hard')} & {r.default_f1:.2f} & "
                     f"{ci(r, 'calibrated_recall')} & {ci(r, 'calibrated_far_hard')} & {r.calibrated_f1:.2f} & "
                     f"{r.auroc:.3f} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_grid(s: pd.DataFrame) -> str:
    t = s[(s.n_frames == 1) & (s.prompt == "A")]
    lines = [r"\begin{tabular}{lllrrrrrrr}", r"\toprule",
             r"Model & Weights & Res. & Recall & FAR$_\text{hard}$ & F1 & AUROC & p95 (ms) & RSS (GiB) & Img.\ tok. \\",
             r"\midrule"]
    prev = None
    for _, r in t.iterrows():
        if prev and prev != r.model:
            lines.append(r"\midrule")
        dag = r"$^\ddagger$" if r.battery else ""
        lines.append(f"{r.model if prev != r.model else ''} & {r.quant.replace('_', chr(92) + '_')} & {r.resolution} & "
                     f"{pct(r.calibrated_recall)} & {pct(r.calibrated_far_hard)} & {r.calibrated_f1:.2f} & {r.auroc:.3f} & "
                     f"{r.lat_p95_ms:.0f}{dag} & {r.peak_rss_gib:.2f} & {r.prompt_tokens:.0f} \\\\")
        prev = r.model
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_frames(s: pd.DataFrame) -> str | None:
    t = s[(s.quant == "Q8_0") & (s.resolution == "native") & (s.prompt == "A")]
    if t.n_frames.nunique() < 2:
        return None
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Model & Frames & Recall & FAR$_\text{hard}$ & F1 & AUROC & p95 (ms) \\", r"\midrule"]
    prev = None
    for _, r in t.sort_values(["m_order", "n_frames"]).iterrows():
        if prev and prev != r.model:
            lines.append(r"\midrule")
        lines.append(f"{r.model if prev != r.model else ''} & {r.n_frames} & {pct(r.calibrated_recall)} & "
                     f"{pct(r.calibrated_far_hard)} & {r.calibrated_f1:.2f} & {r.auroc:.3f} & {r.lat_p95_ms:.0f} \\\\")
        prev = r.model
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_subtypes(s: pd.DataFrame, test: pd.DataFrame) -> str:
    t = s[(s.quant == "Q8_0") & (s.resolution == "native") & (s.n_frames == 1) & (s.prompt == "A")]
    counts = test.drop_duplicates("sample_id").subtype.value_counts()
    names = {"crouching": "Crouch", "kneeling": "Kneel", "bending": "Bend", "sitting_chair": "Sit (chair)",
             "sitting_furniture": "Sit (sofa/bed)", "lying_furniture": "Lie (sofa/bed)"}
    head = " & ".join(f"{names[k]} ({counts.get(k, 0)})" for k in HARD_SUBTYPES)
    lines = [r"\begin{tabular}{l" + "r" * len(HARD_SUBTYPES) + "}", r"\toprule", f"Model & {head} \\\\", r"\midrule"]
    for op, label in [("default", "default"), ("calibrated", "calibrated")]:
        lines.append(rf"\multicolumn{{{len(HARD_SUBTYPES) + 1}}}{{l}}{{\textit{{{label} operating point}}}} \\")
        for _, r in t.iterrows():
            lines.append(f"{r.model} & " + " & ".join(pct(r[f'{op}_far_{k}']) for k in HARD_SUBTYPES) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_viewpoint(s: pd.DataFrame, test: pd.DataFrame) -> str:
    t = s[(s.quant == "Q8_0") & (s.resolution == "native") & (s.n_frames == 1) & (s.prompt == "A")]
    vps = [v for v in ["side", "elevated", "overhead"] if f"calibrated_recall_{v}" in t]
    pos = test[test.label == "person_down"].drop_duplicates("sample_id").viewpoint.value_counts()
    lines = [r"\begin{tabular}{l" + "r" * len(vps) + "}", r"\toprule",
             "Model & " + " & ".join(f"{v} ({pos.get(v, 0)})" for v in vps) + r" \\", r"\midrule"]
    for _, r in t.iterrows():
        lines.append(f"{r.model} & " + " & ".join(pct(r[f'calibrated_recall_{v}']) for v in vps) + r" \\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_baseline(s: pd.DataFrame) -> str | None:
    t = s[s.is_baseline]
    if t.empty:
        return None
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Detector & Frames & Recall & FAR$_\text{hard}$ & FAR$_\text{normal}$ & F1 & ms/frame \\", r"\midrule"]
    for _, r in t.sort_values(["m_order", "n_frames"]).iterrows():
        lines.append(f"{r.model} & {r.n_frames} & {ci(r, 'default_recall')} & {ci(r, 'default_far_hard')} & "
                     f"{pct(r.default_far_normal)} & {r.default_f1:.2f} & {r.lat_median_ms / r.n_frames:.0f} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


def table_efficiency(s: pd.DataFrame) -> str | None:
    """GPU (from the grid) and CPU-only (results/grid/cpu.csv) cost of the Pareto-optimal configs."""
    path = GRID / "cpu.csv"
    if not path.exists():
        return None
    cpu = pd.read_csv(path)
    cpu["resolution"] = cpu.resolution.astype(str)
    runs = run_info()
    lines = [r"\begin{tabular}{lllrrrrrr}", r"\toprule",
             r" & & & \multicolumn{3}{c}{Integrated GPU} & \multicolumn{3}{c}{CPU only (10 threads)} \\",
             r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
             r"Model & Weights & Res. & p95 (ms) & RSS (GiB) & Streams & p95 (ms) & RSS (GiB) & Streams \\", r"\midrule"]
    groups = sorted(cpu.groupby(["model", "resolution", "n_frames"]),
                    key=lambda kv: (list(MODELS).index(kv[0][0].split(":")[0]), QUANT_ORDER[kv[0][0].split(":")[1]],
                                    RES_ORDER[kv[0][1]]))
    for (spec, res, nf), g in groups:
        m = s[(s.spec == spec) & (s.resolution == res) & (s.n_frames == nf) & (s.prompt == "A")]
        if m.empty:
            continue
        r = m.iloc[0]
        rss = max((runs[i]["memory"].get("peak_rss_mb", math.nan) for i in g.run_id.unique() if i in runs),
                  default=math.nan) / 1024
        p95 = g.e2e_ms.quantile(0.95)
        lines.append(f"{r.model} & {r.quant.replace('_', chr(92) + '_')} & {res} & {r.lat_p95_ms:.0f} & {r.peak_rss_gib:.2f} & "
                     f"{r.streams_1hz} & {p95:.0f} & {rss:.2f} & {int(1000 // p95)} \\\\")
    return "\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n"


# ---------------------------------------------------------------- figures
def pareto(s: pd.DataFrame, x: str) -> pd.Series:
    """True for configs not dominated (lower x and higher calibrated F1)."""
    keep = []
    for i, r in s.iterrows():
        dominated = ((s[x] <= r[x]) & (s.calibrated_f1 >= r.calibrated_f1)
                     & ((s[x] < r[x]) | (s.calibrated_f1 > r.calibrated_f1))).any()
        keep.append(not dominated)
    return pd.Series(keep, index=s.index)


def short(r) -> str:
    q = {"16-bit": "16b", "Q8_0": "Q8", "Q4_K_M": "Q4"}[r.quant_bits]
    res = "" if r.resolution == "native" else f"/{r.resolution}"
    fr = "" if r.n_frames == 1 else f"/{r.n_frames}f"
    return f"{r.model.replace('LFM2.5-VL-', 'LFM-').replace('Qwen3.5-', 'Qwen-')} {q}{res}{fr}"


def figure(s: pd.DataFrame, x: str, xlabel: str, path: Path, logx: bool) -> list[str]:
    base = s[s.is_baseline & s[x].notna()]
    t = s[s.prompt == "A"].copy()
    t["front"] = pareto(t, x)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2,
                         "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(3.5, 2.6), dpi=300)
    ax.grid(True, color=GRID_C, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    rest = t[~t.front]
    ax.scatter(rest[x], rest.calibrated_f1, s=12, color=RECESSIVE, linewidths=0, zorder=2, label="other configurations")
    f = t[t.front].sort_values(x)
    ax.plot(f[x], f.calibrated_f1, color=INK2, linewidth=1, zorder=3)
    for fam in ["LFM", "Qwen"]:
        for qb, mk in QUANT_MARKER.items():
            p = f[(f.family == fam) & (f.quant_bits == qb)]
            if len(p):
                ax.scatter(p[x], p.calibrated_f1, s=30, marker=mk, color=FAMILY_COLOR[fam],
                           edgecolors="#fcfcfb", linewidths=1, zorder=4)
    # Selective direct labels: frontier points get a number (key in the caption);
    # offsets alternate below/above so neighbouring labels do not collide.
    for k, (_, r) in enumerate(f.iterrows(), start=1):
        dy = -9 if k % 2 else 4
        ax.annotate(str(k), (r[x], r.calibrated_f1), xytext=(3, dy), textcoords="offset points",
                    fontsize=6, color=INK, fontweight="bold")
    if len(base):  # non-VLM baseline: black cross, labelled
        ax.scatter(base[x], base.calibrated_f1, s=28, marker="x", color=INK, linewidths=1, zorder=4)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("F1, calibrated threshold")
    # legend: family = colour, weights = shape (identity never colour-alone)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=fam) for fam, c in FAMILY_COLOR.items()]
    handles += [Line2D([], [], marker=m, linestyle="", color=INK2, label=q) for q, m in QUANT_MARKER.items()]
    handles += [Line2D([], [], marker="o", linestyle="", color=RECESSIVE, label="dominated")]
    if len(base):
        handles += [Line2D([], [], marker="x", linestyle="", color=INK, label="pose + rule")]
    ax.legend(handles=handles, fontsize=5.5, frameon=False, loc="lower right", ncol=2)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"))
    plt.close(fig)
    key = "; ".join(f"{k}: {short(r)}" for k, (_, r) in enumerate(f.iterrows(), start=1))
    macro = "Key" + "".join(w.capitalize() for w in path.stem.split("_"))  # e.g. \KeyParetoLatency
    (GEN / f"fig_{path.stem}_key.tex").write_text(
        f"\\newcommand{{\\{macro}}}{{{key.replace('_', chr(92) + '_')}}}\n")
    return [short(r) for _, r in f.iterrows()]


TAG = {"lfm-450m": "LfmS", "lfm-1.6b": "LfmM", "lfm-3b": "LfmL", "qwen3.5-2b": "QwenS", "qwen3.5-4b": "QwenL",
       "yolo11n-pose-rule": "PoseN", "yolo11s-pose-rule": "PoseS"}


def numbers(s: pd.DataFrame, df: pd.DataFrame) -> str:
    """LaTeX macros for numbers quoted in the results text (\\R<Model><Metric>)."""
    n = {"RNConfigs": str(len(s[~s.is_baseline])), "RNDecisions": f"{len(df[df.stage != 'baseline']):,}",
         "RNUnparsed": str(int((df.answer == "unparsed").sum()))}
    base = s[(s.quant.isin(["Q8_0", "pose"])) & (s.resolution == "native") & (s.n_frames == 1)
             & (s.prompt.isin(["A", "rule"]))]
    for _, r in base.iterrows():
        k = "R" + TAG[r.spec.split(":")[0]]
        n[k + "DefRecall"], n[k + "DefFarHard"] = pct(r.default_recall), pct(r.default_far_hard)
        n[k + "DefFarNormal"], n[k + "DefFone"] = pct(r.default_far_normal), f"{r.default_f1:.2f}"
        n[k + "CalRecall"], n[k + "CalFarHard"] = pct(r.calibrated_recall), pct(r.calibrated_far_hard)
        n[k + "CalFone"], n[k + "Auroc"] = f"{r.calibrated_f1:.2f}", f"{r.auroc:.3f}"
        n[k + "Ptf"], n[k + "Med"] = f"{r.lat_p95_ms:.0f}", f"{r.lat_median_ms:.0f}"
        n[k + "Rss"] = "--" if pd.isna(r.peak_rss_gib) else f"{r.peak_rss_gib:.2f}"
        n[k + "Tok"] = f"{r.prompt_tokens:.0f}"
        for sub, w in [("crouching", "Crouch"), ("kneeling", "Kneel"), ("bending", "Bend"),
                       ("sitting_chair", "SitChair"), ("sitting_furniture", "SitSofa"), ("lying_furniture", "LieSofa")]:
            n[k + "CalFar" + w] = pct(r[f"calibrated_far_{sub}"])
            n[k + "DefFar" + w] = pct(r[f"default_far_{sub}"])
        for vp, w in [("side", "Side"), ("elevated", "Elevated"), ("overhead", "Overhead")]:
            if f"calibrated_recall_{vp}" in r:
                n[k + "CalRecall" + w] = pct(r[f"calibrated_recall_{vp}"])
    # One macro family per configuration: \\C<model><weights><res><frames><metric>.
    qt = {"F16": "Half", "BF16": "Half", "Q8_0": "Qeight", "Q4_K_M": "Qfour", "pose": "Pose"}
    rt = {"native": "Nat", "512": "Fivetwelve", "256": "Twofiftysix"}
    ft = {1: "One", 4: "Four", 8: "Eight"}
    for _, r in s[s.prompt.isin(["A", "rule"])].iterrows():
        k = "C" + TAG[r.spec.split(":")[0]] + qt[r.quant] + rt[r.resolution] + ft[int(r.n_frames)]
        n[k + "Fone"], n[k + "Recall"] = f"{r.calibrated_f1:.2f}", pct(r.calibrated_recall)
        n[k + "FarHard"], n[k + "Auroc"] = pct(r.calibrated_far_hard), f"{r.auroc:.3f}"
        n[k + "DefFone"], n[k + "DefRecall"] = f"{r.default_f1:.2f}", pct(r.default_recall)
        n[k + "Ptf"], n[k + "Streams"] = f"{r.lat_p95_ms:.0f}", str(int(r.streams_1hz))
        n[k + "Rss"] = "--" if pd.isna(r.peak_rss_gib) else f"{r.peak_rss_gib:.2f}"
        n[k + "Tok"] = f"{r.prompt_tokens:.0f}"
    va = s[(s.prompt == "A")]
    best = va.loc[va.calibrated_f1.idxmax()]
    n["RBestFone"], n["RBestCfg"] = f"{best.calibrated_f1:.2f}", short(best)
    knee = va[(va.spec == "lfm-1.6b:Q4_K_M") & (va.resolution == "256") & (va.n_frames == 1)]
    if len(knee):
        n["RKneeGap"] = f"{best.calibrated_f1 - knee.iloc[0].calibrated_f1:.2f}"
    cpu_path = GRID / "cpu.csv"
    if cpu_path.exists():
        cpu = pd.read_csv(cpu_path)
        cpu["resolution"] = cpu.resolution.astype(str)
        for (spec, res, nf), g in cpu.groupby(["model", "resolution", "n_frames"]):
            k = "C" + TAG[spec.split(":")[0]] + qt[spec.split(":")[1]] + rt[res] + ft[int(nf)]
            p95 = g.e2e_ms.quantile(0.95)
            n[k + "CpuPtf"], n[k + "CpuStreams"] = f"{p95:.0f}", str(int(1000 // p95))
    rt_path = GRID / "retime.csv"
    if "retime_same_answers" in s:
        same = s.retime_same_answers.dropna().str.split("/", expand=True).astype(int)
        n["RRetimeConfigs"], n["RRetimeSame"] = str(len(same)), f"{same[0].sum():,}"
        n["RRetimeTotal"] = f"{same[1].sum():,}"
    feat = ROOT / "results/baseline/features_yolo11n-pose.csv"
    if feat.exists():
        f = pd.read_csv(feat)
        det = set(f[f.person >= 0].frame_path)
        lab = pd.concat([pd.read_csv(ROOT / f"data/benchmark/v1/labels_{x}.csv") for x in ("test", "dev")])
        lab["det"] = lab.image_path.isin(det)
        down = lab[(lab.label == "person_down") & (lab.viewpoint != "overhead")]
        norm = lab[lab.label == "normal"]
        n["RPoseDetDown"], n["RPoseDetNormal"] = pct(down.det.mean()), pct(norm.det.mean())
        tim = json.loads((ROOT / "results/baseline/timing.json").read_text())
        n["RPoseNMs"] = f"{tim['yolo11n-pose.pt']['median_ms']:.0f}"
        n["RPoseSMs"] = f"{tim['yolo11s-pose.pt']['median_ms']:.0f}"
    return "% Generated by scripts/analyze_grid.py -- do not edit.\n" + "".join(
        f"\\newcommand{{\\{k}}}{{{v.replace('%', '')}}}\n" for k, v in sorted(n.items()))


def main() -> None:
    df = load()
    s = apply_retime(summarize(df), df)
    s.to_csv(GRID / "summary.csv", index=False)
    test = df[df.split == "test"]
    GEN.mkdir(parents=True, exist_ok=True)
    (GEN / "res_prompts.tex").write_text(table_prompts(s))
    (GEN / "res_main.tex").write_text(table_main(s))
    (GEN / "res_grid.tex").write_text(table_grid(s))
    (GEN / "res_subtypes.tex").write_text(table_subtypes(s, test))
    (GEN / "res_viewpoint.tex").write_text(table_viewpoint(s, test))
    (GEN / "res_numbers.tex").write_text(numbers(s, df))
    te = table_efficiency(s)
    if te:
        (GEN / "res_efficiency.tex").write_text(te)
    tb = table_baseline(s)
    if tb:
        (GEN / "res_baseline.tex").write_text(tb)
    fr = table_frames(s)
    if fr:
        (GEN / "res_frames.tex").write_text(fr)
    front_lat = figure(s, "lat_p95_ms", "p95 latency per decision (ms, Metal GPU)", FIG / "pareto_latency.pdf", True)
    front_mem = figure(s, "peak_rss_gib", "peak memory (GiB, RSS)", FIG / "pareto_memory.pdf", False)
    print(f"{len(s)} configurations summarised ({df.stage.unique().tolist()})")
    print("latency frontier:", front_lat)
    print("memory frontier:", front_mem)
    print("battery-affected timings:", s[s.battery].spec.tolist())
    if "retimed" in s:
        print("re-timed:", s[s.retimed.fillna(False).astype(bool)][["spec", "resolution", "retime_same_answers"]].to_string(index=False))
    front = s[(s.prompt == "A")]
    lat_front = front[pareto(front, "lat_p95_ms")]
    mem_front = front[pareto(front, "peak_rss_gib")]
    union = pd.concat([lat_front, mem_front]).drop_duplicates(["spec", "resolution", "n_frames"])
    (GRID / "frontier.json").write_text(json.dumps(
        [{"spec": r.spec, "resolution": r.resolution, "n_frames": int(r.n_frames)} for _, r in union.iterrows()], indent=2))


if __name__ == "__main__":
    main()
