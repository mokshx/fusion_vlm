"""Download the source data for the benchmark. Every subcommand is resumable:
files already on disk are skipped, partial files are restarted.

  urfall    4 fps frame bank (one frame every 0.25 s) from all 70 sequences,
            cam0 for all, cam1 for falls, via HTTP range requests into the
            official per-sequence zips; plus per-sequence sync CSVs.
  le2i      annotation files and videos of the annotated scenes (Home_01/02,
            Coffee_room_01/02) from the official dat@UBFC zip, up to a size
            budget; waits for the server if it is down.
  omnifall  OmniFall re-annotation label files (Hugging Face).

MCF (Montreal) is plain per-scenario zips: see scripts/download_all.sh.

Usage: uv run python scripts/download_sources.py {urfall,le2i,omnifall} [--workers N]
"""

import argparse
import csv
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from pdvlm.remote_zip import extract_member, open_remote_zip

RAW = Path("data/raw")
FRAME_STEP = 7.5  # frames at 30 fps -> one frame every 0.25 s


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ---------------------------------------------------------------- UR Fall
UR_BASE = "http://fenix.ur.edu.pl/~mkepski/ds/data"


def urfall(workers: int) -> None:
    out = RAW / "urfall"
    out.mkdir(parents=True, exist_ok=True)
    for name in ("urfall-cam0-falls.csv", "urfall-cam0-adls.csv"):  # official per-frame posture labels
        if not (out / name).exists():
            r = requests.get(f"{UR_BASE}/{name}", timeout=120)
            r.raise_for_status()
            (out / name).write_bytes(r.content)
    stems = [f"fall-{i:02d}-cam{c}-rgb" for i in range(1, 31) for c in (0, 1)]
    stems += [f"adl-{i:02d}-cam0-rgb" for i in range(1, 41)]
    manifest = out / "frames" / "manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    new = not manifest.exists()
    mf = open(manifest, "a", newline="")
    mw = csv.writer(mf, lineterminator="\n")
    if new:
        mw.writerow(["stem", "file", "zip_index", "n_frames"])

    def one(stem: str) -> None:
        seq = stem.rsplit("-cam", 1)[0]
        sync = out / "sync" / f"{seq}-data.csv"
        if not sync.exists():
            sync.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(f"{UR_BASE}/{seq}-data.csv", timeout=120)
            r.raise_for_status()
            sync.write_bytes(r.content)
        done_flag = out / "frames" / stem / ".done"
        if done_flag.exists():
            return
        z, f = open_remote_zip(f"{UR_BASE}/{stem}.zip")
        pngs = sorted(n for n in z.namelist() if n.endswith(".png"))
        idx = sorted({round(k * FRAME_STEP) for k in range(int(len(pngs) / FRAME_STEP) + 1)
                      if round(k * FRAME_STEP) < len(pngs)})
        for i in idx:
            dest = out / "frames" / stem / Path(pngs[i]).name
            if not dest.exists():
                extract_member(z, pngs[i], dest)
                with lock:
                    mw.writerow([stem, dest.name, i, len(pngs)])
                    mf.flush()
        done_flag.touch()
        log(f"urfall {stem}: {len(idx)} frames, {f.bytes_fetched / 1e6:.1f} MB")

    with ThreadPoolExecutor(workers) as ex:
        for fut in [ex.submit(one, s) for s in stems]:
            try:
                fut.result()
            except Exception as e:  # keep going; a re-run picks up what failed
                log(f"urfall ERROR {e!r}")
    mf.close()
    log("urfall finished")


# ---------------------------------------------------------------- Le2i
LE2I_ENDPOINT = "https://search-data.ubfc.fr/dl_data.php?file=101"
LE2I_SCENES = ["Home_01", "Home_02", "Coffee_room_01", "Coffee_room_02"]


def le2i_url(max_wait_h: float = 24) -> str:
    """Resolve the short-lived pre-signed S3 URL; wait while the portal is down."""
    t_end = time.time() + max_wait_h * 3600
    while True:
        try:
            r = requests.get(LE2I_ENDPOINT, allow_redirects=False, timeout=60)
            loc = r.headers.get("Location")
            if r.status_code in (301, 302, 303, 307, 308) and loc:
                return loc
            log(f"le2i portal returned {r.status_code}; retrying in 10 min")
        except requests.RequestException as e:
            log(f"le2i portal error {e!r}; retrying in 10 min")
        if time.time() > t_end:
            raise TimeoutError("le2i portal unavailable")
        time.sleep(600)


def le2i(workers: int, budget_gb: float) -> None:
    out = RAW / "le2i"
    z, f = open_remote_zip(le2i_url)
    infos = [i for i in z.infolist() if not i.is_dir()]
    in_scope = [i for i in infos if any(s in i.filename for s in LE2I_SCENES)]
    small = [i for i in in_scope if not i.filename.lower().endswith((".avi", ".mp4"))]
    small += [i for i in infos if "/" not in i.filename.strip("/")]  # top-level readme/licence
    videos = {s: sorted((i for i in in_scope if i.filename.lower().endswith((".avi", ".mp4"))
                         and s in i.filename), key=lambda i: i.filename) for s in LE2I_SCENES}
    log(f"le2i: {len(infos)} members; {len(small)} small files; "
        + ", ".join(f"{s}: {len(v)} videos" for s, v in videos.items()))
    for i in small:
        dest = out / i.filename
        if not dest.exists():
            extract_member(z, i.filename, dest)
    # Round-robin across scenes so a partial budget still covers all four.
    order, k = [], 0
    while any(k < len(v) for v in videos.values()):
        order += [v[k] for v in videos.values() if k < len(v)]
        k += 1
    budget, chosen = budget_gb * 1e9, []
    for i in order:
        if sum(c.compress_size for c in chosen) + i.compress_size > budget:
            break
        chosen.append(i)
    log(f"le2i: fetching {len(chosen)} videos, {sum(c.compress_size for c in chosen) / 1e9:.2f} GB")

    def one(info) -> None:
        dest = out / info.filename
        if dest.exists():
            return
        zz, ff = open_remote_zip(le2i_url)  # own connection per worker
        extract_member(zz, info.filename, dest)
        log(f"le2i {info.filename} ({info.compress_size / 1e6:.0f} MB)")

    with ThreadPoolExecutor(workers) as ex:
        for fut in [ex.submit(one, i) for i in chosen]:
            try:
                fut.result()
            except Exception as e:
                log(f"le2i ERROR {e!r}")
    log("le2i finished")


# ---------------------------------------------------------------- OmniFall
def omnifall() -> None:
    out = RAW / "omnifall"
    out.mkdir(parents=True, exist_ok=True)
    tree = requests.get("https://huggingface.co/api/datasets/simplexsigil2/omnifall/tree/main/labels",
                        timeout=60).json()
    files = [t["path"] for t in tree if t["path"].endswith(".csv")] + ["README.md", "LABELS.md"]
    for p in files:
        dest = out / p
        if dest.exists():
            continue
        r = requests.get(f"https://huggingface.co/datasets/simplexsigil2/omnifall/resolve/main/{p}", timeout=300)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)
        log(f"omnifall {p} ({len(r.content) / 1e3:.0f} kB)")
    log("omnifall finished")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["urfall", "le2i", "omnifall"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-gb", type=float, default=3.0, help="le2i video budget")
    a = ap.parse_args()
    if a.source == "urfall":
        urfall(a.workers)
    elif a.source == "le2i":
        le2i(a.workers, a.budget_gb)
    else:
        omnifall()


if __name__ == "__main__":
    main()
