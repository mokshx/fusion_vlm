"""Generate LaTeX tables and number macros for the paper from result CSVs.

Numbers quoted in the paper should come from here, never be typed by hand, so a
re-run of the experiments updates the manuscript.

Usage: uv run python scripts/make_paper_tables.py
Writes: paper/generated/{pilot_accuracy.tex, pilot_efficiency.tex, numbers.tex}
"""

import json
from pathlib import Path

import pandas as pd

from pdvlm.stats import bootstrap_ci, per_cluster_counts, wilson_ci

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results/step0"
OUT = ROOT / "paper/generated"

MODEL_NAMES = {"lfm-450m:Q8_0": "LFM2.5-VL-450M", "lfm-3b:Q8_0": "LFM2.5-VL-3B",
               "qwen3.5-2b:Q8_0": "Qwen3.5-2B"}
TAG = {"lfm-450m:Q8_0": "Small", "lfm-3b:Q8_0": "Large", "qwen3.5-2b:Q8_0": "Qwen"}


def read_all(paths: list[Path]) -> pd.DataFrame:
    return pd.concat([pd.read_csv(p) for p in paths if p.exists()], ignore_index=True)
PROMPT_ORDER = ["A", "B", "B2"]


def auroc(g: pd.DataFrame) -> float:
    pos = g[g.label == "person_down"].p_yes.to_numpy()
    neg = g[g.label != "person_down"].p_yes.to_numpy()
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def frac(k: int, n: int) -> str:
    return f"{k}/{n}"


def accuracy_rows(df: pd.DataFrame) -> list[dict]:
    rows = []
    for (model, prompt), g in df.groupby(["model", "prompt"]):
        pos, yes = g.label == "person_down", g.answer == "yes"
        tp, fp = int((pos & yes).sum()), int((~pos & yes).sum())
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / int(pos.sum())
        hn, nm = g.label == "hard_negative", g.label == "normal"
        rows.append({
            "model": model, "prompt": prompt,
            "recall": frac(tp, int(pos.sum())),
            "fa_hard": frac(int((hn & yes).sum()), int(hn.sum())),
            "fa_normal": frac(int((nm & yes).sum()), int(nm.sum())),
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "auroc": auroc(g),
            "strict": (g.parse_mode == "strict").mean(),
        })
    order = {p: i for i, p in enumerate(PROMPT_ORDER)}
    return sorted(rows, key=lambda r: (list(MODEL_NAMES).index(r["model"]), order[r["prompt"]]))


def pilot_accuracy(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model (Q8\_0) & Prompt & Recall & FA$_\text{hard}$ & FA$_\text{normal}$ & F1 & AUROC \\",
        r"\midrule",
    ]
    prev = None
    for r in accuracy_rows(df):
        if prev and prev != r["model"]:
            lines.append(r"\midrule")
        name = MODEL_NAMES[r["model"]] if prev != r["model"] else ""
        lines.append(f"{name} & {r['prompt']} & {r['recall']} & {r['fa_hard']} & {r['fa_normal']} "
                     f"& {r['f1']:.2f} & {r['auroc']:.2f} \\\\")
        prev = r["model"]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def run_memory(csv_paths: list[Path]) -> dict[str, dict]:
    """model -> backend/memory info from the run-*.json sidecars next to the CSVs."""
    out = {}
    for csv_path in [p for p in csv_paths if p.exists()]:
        run_ids = set(pd.read_csv(csv_path).run_id)
        for j in csv_path.parent.glob("run-*.json"):
            d = json.loads(j.read_text())
            if d["run_id"] in run_ids:
                out[d["model"]] = d
    return out


def pilot_efficiency(cpu_csvs: list[Path], gpu_csvs: list[Path]) -> tuple[str, dict]:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Model (Q8\_0) & Backend & Median & p95 & Peak RSS & Streams \\",
        r" & & (ms) & (ms) & (GiB) & @1\,Hz \\",
        r"\midrule",
    ]
    nums = {}
    for csvs, backend in [(cpu_csvs, "CPU, 10 thr."), (gpu_csvs, "Metal GPU")]:
        df = read_all(csvs)
        df = df[df.prompt == "A"]
        mem = run_memory(csvs)
        for model in [m for m in MODEL_NAMES if m in set(df.model)]:
            g = df[df.model == model]
            med, p95 = g.e2e_ms.median(), g.e2e_ms.quantile(0.95)
            rss = mem[model]["memory"]["peak_rss_mb"] / 1024
            streams = int(1000 // p95)
            lines.append(f"{MODEL_NAMES[model]} & {backend} & {med:.0f} & {p95:.0f} & {rss:.2f} & {streams} \\\\")
            key = TAG[model] + ("Cpu" if "CPU" in backend else "Gpu")
            nums[f"Pilot{key}MedianMs"] = f"{med:.0f}"
            nums[f"Pilot{key}RssGiB"] = f"{rss:.1f}"
            nums[f"Pilot{key}FootprintGiB"] = f"{mem[model]['memory']['peak_footprint_mb'] / 1024:.1f}"
            nums[f"Pilot{key}Streams"] = str(streams)
            prefill = (g.prompt_ms / g.e2e_ms).median() * 100
            nums[f"Pilot{key}PrefillPct"] = f"{prefill:.0f}"
            nums[f"Pilot{key}DecodeMs"] = f"{g.decode_ms.median():.0f}"
        lines.append(r"\midrule" if backend.startswith("CPU") else "")
    lines = [l for l in lines if l] + [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", nums


def memory_ablation(nums: dict) -> str | None:
    """Table from results/memory_ablation/: one row per model x backend x flags."""
    d = ROOT / "results/memory_ablation"
    csvs = sorted(d.glob("*.csv"))
    if not csvs:
        return None
    rows = []
    for c in csvs:
        df = pd.read_csv(c)
        side = json.loads((d / f"run-{df.run_id.iloc[0]}.json").read_text())
        b, m = side["backend"], side["memory"]
        rows.append({
            "model": side["model"], "gpu": b["n_gpu_layers"] > 0,
            "flags": b.get("extra_args", "") or "default",
            "load_s": b["load_s"], "median": df.e2e_ms.median(),
            "rss": m["peak_rss_mb"] / 1024, "fp": m["peak_footprint_mb"] / 1024,
            "sys": m["sys_mem_delta_mb"] / 1024, "answers": df.set_index("sample_id").answer,
        })
    flag_order = ["default", "--no-mmap", "--no-repack", "--no-mmap --no-repack"]
    rows.sort(key=lambda r: (list(MODEL_NAMES).index(r["model"]), r["gpu"], flag_order.index(r["flags"])))
    # System-level delta is recorded in the run JSONs but not tabulated: on macOS
    # (memory compression, background apps) it was not reproducible.
    lines = [r"\begin{tabular}{lllrrrrc}", r"\toprule",
             r"Model & Backend & Flags & Load & Median & RSS & Footpr. & Same \\",
             r" & & & (s) & (ms) & (GiB) & (GiB) & answers \\", r"\midrule"]
    default = {(r["model"], r["gpu"]): r for r in rows if r["flags"] == "default"}
    prev = None
    for r in rows:
        key = (r["model"], r["gpu"])
        if prev and prev != key:
            lines.append(r"\midrule")
        ref = default[key]["answers"]
        same = int((r["answers"] == ref.reindex(r["answers"].index)).sum())
        flags = r["flags"].replace("--", "-{}-")
        lines.append(f"{MODEL_NAMES[r['model']] if prev != key else ''} & {'Metal' if r['gpu'] else 'CPU'} & "
                     f"\\texttt{{{flags}}} & {r['load_s']:.1f} & {r['median']:.0f} & {r['rss']:.2f} & "
                     f"{r['fp']:.2f} & {same}/{len(r['answers'])} \\\\")
        prev = key
        tag = TAG[r["model"]] + ("Gpu" if r["gpu"] else "Cpu") + \
            {"default": "Default", "--no-mmap": "NoMmap", "--no-repack": "NoRepack",
             "--no-mmap --no-repack": "NoBoth"}[r["flags"]]
        nums[f"Abl{tag}RssGiB"] = f"{r['rss']:.2f}"
        nums[f"Abl{tag}FpGiB"] = f"{r['fp']:.2f}"
        nums[f"Abl{tag}SysGiB"] = f"{r['sys']:.2f}"
        nums[f"Abl{tag}MedianMs"] = f"{r['median']:.0f}"
        nums[f"Abl{tag}LoadS"] = f"{r['load_s']:.1f}"
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cpu_csvs = [RES / "run2/mac_m4pro_cpu.csv", RES / "run4/qwen_cpu.csv"]
    gpu_csvs = [RES / "run2/mac_m4pro_metal.csv", RES / "run3/main_B2_metal.csv", RES / "run4/qwen_metal.csv"]
    floor_csvs = [RES / "run3/floor_metal.csv", RES / "run4/qwen_floor_metal.csv"]
    main_set = read_all(gpu_csvs)
    (OUT / "pilot_accuracy.tex").write_text(pilot_accuracy(main_set))

    eff, nums = pilot_efficiency(cpu_csvs, gpu_csvs)
    (OUT / "pilot_efficiency.tex").write_text(eff)

    # Agreement checks quoted in the text.
    cpu, gpu = read_all(cpu_csvs), main_set
    m = cpu.merge(gpu, on=["model", "prompt", "sample_id"], suffixes=("_c", "_g"))
    nums["PilotCpuGpuAgree"] = frac(int((m.answer_c == m.answer_g).sum()), len(m))
    hf = pd.read_csv(RES / "mac_m4pro_hf_mps.csv")
    q8 = cpu[cpu.model == "lfm-450m:Q8_0"]
    m = q8.merge(hf, on=["prompt", "sample_id"], suffixes=("_q", "_h"))
    nums["PilotQeightBfAgree"] = frac(int((m.answer_q == m.answer_h).sum()), len(m))
    allruns = pd.concat([main_set, read_all(floor_csvs), cpu, hf])
    nums["PilotNAnswers"] = str(len(allruns))
    nums["PilotNUnparsed"] = str(int((allruns.answer == "unparsed").sum()))
    nums["PilotNLenient"] = str(int((allruns.parse_mode == "lenient").sum()))
    # Standing/walking frames filmed in the same rooms as the deliberate floor-lying frames.
    same_room = main_set[main_set.sample_id.str.match(r"adl-(3[0-5]|3[89]|40)-") & (main_set.label == "normal")]
    a = same_room[same_room.prompt == "A"]
    nums["PilotSameRoomNormalRejectA"] = frac(int((a.answer == "no").sum()), len(a))
    floor = read_all(floor_csvs)
    nums["PilotFloorHits"] = frac(int((floor.answer == "yes").sum()), len(floor))

    rows = {(r["model"], r["prompt"]): r for r in accuracy_rows(main_set)}
    for model, tag in [(m, TAG[m]) for m in MODEL_NAMES if m in set(main_set.model)]:
        for p in PROMPT_ORDER:
            r = rows[(model, p)]
            k = f"Pilot{tag}{p.replace('2', 'two')}"
            nums[f"{k}FaHard"] = r["fa_hard"]
            nums[f"{k}FaNormal"] = r["fa_normal"]
            nums[f"{k}Fone"] = f"{r['f1']:.2f}"
            nums[f"{k}Auroc"] = f"{r['auroc']:.2f}"

    # Pilot CIs quoted in the text (one frame per sequence -> ordinary bootstrap).
    def pct(lo, hi):
        return f"{100 * lo:.0f}--{100 * hi:.0f}\\%"
    g3a = main_set[(main_set.model == "lfm-3b:Q8_0") & (main_set.prompt == "A")]
    _, lo, hi = bootstrap_ci(per_cluster_counts(g3a), "far_hard")
    nums["PilotLargeAFaHardCI"] = pct(lo, hi)
    lo, hi = wilson_ci(10, 10)
    nums["PilotRecallWilsonCI"] = pct(lo, hi)

    env = json.loads(next((RES / "run2").glob("run-*.json")).read_text())["env"]
    nums["PilotCpu"] = env["cpu"]
    nums["PilotRamGB"] = f"{env['ram_total_gb']:.0f}"
    nums["PilotLlamaBuild"] = env["llama_cpp_version"].split()[1]
    nums["PilotOs"] = env["os"].split("-")[1]

    abl = memory_ablation(nums)
    if abl:
        (OUT / "memory_ablation.tex").write_text(abl)

    (OUT / "numbers.tex").write_text(
        "% Generated by scripts/make_paper_tables.py -- do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(nums.items())))
    print(f"wrote {OUT}/: pilot_accuracy.tex, pilot_efficiency.tex, numbers.tex ({len(nums)} macros)")


if __name__ == "__main__":
    main()
