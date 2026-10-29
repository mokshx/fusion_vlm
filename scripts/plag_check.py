"""Free, local text-overlap check of the paper against the sources it cites.

Not a substitute for iThenticate/Turnitin: it only compares against the
bibliography (arXiv full text, open-access PDFs / abstracts via OpenAlex,
linked web pages) plus the Hugging Face model cards of the evaluated models.

    uv run --with pypdf python -I scripts/plag_check.py fetch  --corpus /tmp/pdvlm_plag/corpus
    uv run --with pypdf python -I scripts/plag_check.py check  --corpus /tmp/pdvlm_plag/corpus
    uv run --extra hf --with sentence-transformers python -I scripts/plag_check.py semantic --corpus /tmp/pdvlm_plag/corpus
"""

from __future__ import annotations

import argparse
import html
import io
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
UA = {"User-Agent": "pdvlm-overlap-check/0.1 (mailto:chadhamoksh@gmail.com)"}
HF_MODELS = [
    "LiquidAI/LFM2.5-VL-450M",
    "LiquidAI/LFM2.5-VL-1.6B",
    "LiquidAI/LFM2.5-VL-3B",
    "Qwen/Qwen3.5-2B",
    "Qwen/Qwen3.5-4B",
]


# ---------------------------------------------------------------- fetch


def parse_bib(path: Path) -> list[dict]:
    entries = []
    for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", path.read_text(), re.S):
        fields = dict(
            (k.lower(), v.strip())
            for k, v in re.findall(r"(\w+)\s*=\s*\{(.*?)\}\s*,?\s*(?:\n|$)", m.group(2), re.S)
        )
        fields["key"] = m.group(1).strip()
        entries.append(fields)
    return entries


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    return "\n".join((p.extract_text() or "") for p in reader.pages)


def html_text(raw: str) -> str:
    raw = re.sub(r"(?is)<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
    return html.unescape(re.sub(r"<[^>]+>", " ", raw))


def get(url: str, timeout: int = 30, deadline: int = 90) -> requests.Response | None:
    """GET with a hard wall-clock deadline (some servers trickle bytes forever)."""
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True, stream=True)
        if not r.ok:
            return None
        start, chunks = time.monotonic(), []
        for chunk in r.iter_content(1 << 16):
            chunks.append(chunk)
            if time.monotonic() - start > deadline:
                return None
        r._content = b"".join(chunks)
        return r
    except requests.RequestException:
        return None


def to_text(r: requests.Response) -> str:
    ctype = r.headers.get("content-type", "")
    if "pdf" in ctype or r.content[:4] == b"%PDF":
        return pdf_text(r.content)
    return html_text(r.text)


def openalex(doi: str) -> tuple[str, list[str]]:
    r = get(f"https://api.openalex.org/works/doi:{doi}")
    if r is None:
        return "", []
    w = r.json()
    abstract = ""
    if inv := w.get("abstract_inverted_index"):
        pos = sorted((i, word) for word, idxs in inv.items() for i in idxs)
        abstract = " ".join(word for _, word in pos)
    pdfs = [
        loc["pdf_url"]
        for loc in [w.get("best_oa_location") or {}] + (w.get("locations") or [])
        if loc and loc.get("pdf_url")
    ]
    return abstract, list(dict.fromkeys(pdfs))


def fetch_entry(e: dict) -> tuple[str, str]:
    """Return (text, provenance) for a bib entry, best source first."""
    if eprint := e.get("eprint"):
        # arxiv.org/pdf trickles; the HTML render and the export mirror are fast
        for url in (f"https://arxiv.org/html/{eprint}", f"https://export.arxiv.org/pdf/{eprint}"):
            r = get(url)
            if r is not None and len(t := to_text(r).split()) > 500:
                return " ".join(t), f"arXiv:{eprint} full text"
    parts, prov = [], []
    if doi := e.get("doi"):
        abstract, pdfs = openalex(doi)
        for url in pdfs[:2]:
            r = get(url)
            if r is not None:
                try:
                    t = to_text(r)
                except Exception:
                    continue
                if len(t.split()) > 500:
                    return t, f"OA full text {url}"
        if abstract:
            parts.append(abstract)
            prov.append("OpenAlex abstract")
    if url := e.get("url"):
        gh = re.match(r"https://github.com/([^/]+/[^/]+)", url)
        if gh:
            r = get(f"https://raw.githubusercontent.com/{gh.group(1)}/HEAD/README.md")
        else:
            r = get(url)
        if r is not None:
            try:
                parts.append(to_text(r))
                prov.append(f"page {url}")
            except Exception:
                pass
    return "\n".join(parts), ", ".join(prov) or "nothing found"


def cmd_fetch(args) -> None:
    out = Path(args.corpus)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for e in parse_bib(PAPER / "references.bib"):
        dst = out / f"{e['key']}.txt"
        if dst.exists() and not args.force:
            manifest[e["key"]] = "cached"
            continue
        text, prov = fetch_entry(e)
        dst.write_text(text)
        manifest[e["key"]] = f"{prov} ({len(text.split())} words)"
        print(f"{e['key']:28s} {manifest[e['key']]}", flush=True)
        time.sleep(1)  # be polite to arXiv / OpenAlex
    for repo in HF_MODELS:
        key = "hf_" + repo.replace("/", "_")
        r = get(f"https://huggingface.co/{repo}/raw/main/README.md")
        text = r.text if r is not None else ""
        (out / f"{key}.txt").write_text(text)
        manifest[key] = f"HF model card ({len(text.split())} words)"
        print(f"{key:28s} {manifest[key]}", flush=True)
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=1))


# ---------------------------------------------------------------- check


def detex(src: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", src)
    s = re.sub(r"\\begin\{(table|tabular|equation|align)\*?\}.*?\\end\{\1\*?\}", " ", s, flags=re.S)
    s = re.sub(r"\$[^$]*\$", " MATH ", s)
    s = re.sub(r"\\(cite[pt]?|citet|citep|ref|cref|Cref|label|input|include|url|href)\*?(\[[^]]*\])*\{[^}]*\}", " ", s)
    s = re.sub(r"\\[a-zA-Z@]+\*?(\[[^]]*\])?", " ", s)  # strip command names, keep {args}
    s = re.sub(r"[{}~]", " ", s)
    return s


def tokenize(text: str) -> list[str]:
    text = re.sub(r"-\s*\n\s*", "", text)  # de-hyphenate PDF line breaks
    return re.findall(r"[a-z0-9]+", text.lower())


def sentences(raw: str) -> list[str]:
    flat = re.sub(r"\s+", " ", raw)
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", flat) if len(s.split()) >= 6]


def cmd_check(args) -> None:
    n = args.n
    corpus = Path(args.corpus)
    index: dict[tuple, set[str]] = defaultdict(set)
    src_tokens: dict[str, list[str]] = {}
    for f in sorted(corpus.glob("*.txt")):
        toks = tokenize(f.read_text(errors="ignore"))
        src_tokens[f.stem] = toks
        for i in range(len(toks) - n + 1):
            index[tuple(toks[i : i + n])].add(f.stem)

    report = []
    total_words = flagged_words = 0
    for tex in sorted((PAPER / "sections").glob("*.tex")):
        raw = detex(tex.read_text())
        for sent in sentences(raw):
            toks = tokenize(sent)
            total_words += len(toks)
            hit = [False] * len(toks)
            srcs: dict[str, int] = defaultdict(int)
            for i in range(len(toks) - n + 1):
                g = tuple(toks[i : i + n])
                if g in index:
                    for j in range(i, i + n):
                        hit[j] = True
                    for s in index[g]:
                        srcs[s] += 1
            # longest contiguous matched run
            best = run = 0
            for h in hit:
                run = run + 1 if h else 0
                best = max(best, run)
            if best >= args.min_run:
                flagged_words += sum(hit)
                report.append(
                    dict(
                        file=tex.name,
                        longest_run=best,
                        frac=round(sum(hit) / len(toks), 2),
                        sources=sorted(srcs, key=srcs.get, reverse=True)[:3],
                        sentence=sent,
                    )
                )

    report.sort(key=lambda r: (-r["longest_run"], r["file"]))
    lines = [
        f"# Overlap report (n={n}, min run={args.min_run} words)",
        "",
        f"Sources indexed: {len(src_tokens)}; paper words checked: {total_words}; "
        f"words in flagged runs: {flagged_words} ({100 * flagged_words / max(total_words, 1):.1f}%)",
        "",
    ]
    for r in report:
        lines += [
            f"## {r['file']} — run {r['longest_run']} words, {int(r['frac'] * 100)}% of sentence — {', '.join(r['sources'])}",
            "",
            f"> {r['sentence']}",
            "",
        ]
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines[:4]))
    print(f"{len(report)} flagged sentences -> {args.out}")


def cmd_semantic(args) -> None:
    """Paraphrase check: nearest source sentence by embedding cosine similarity."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    src_sents, src_keys = [], []
    for f in sorted(Path(args.corpus).glob("*.txt")):
        flat = re.sub(r"-\s*\n\s*", "", f.read_text(errors="ignore"))
        for s in sentences(flat):
            if 8 <= len(s.split()) <= 80:
                src_sents.append(s)
                src_keys.append(f.stem)
    paper = [
        (tex.name, s)
        for tex in sorted((PAPER / "sections").glob("*.tex"))
        for s in sentences(detex(tex.read_text()))
    ]
    enc = dict(batch_size=256, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    src_emb = model.encode(src_sents, **enc)
    pap_emb = model.encode([s for _, s in paper], **enc)
    sims = pap_emb @ src_emb.T
    best, idx = sims.max(dim=1)

    hits = sorted(
        ((float(b), paper[i], src_keys[j], src_sents[j]) for i, (b, j) in enumerate(zip(best, idx.tolist()))),
        reverse=True,
    )
    lines = [f"# Paraphrase report ({args.model}, cosine >= {args.threshold})", ""]
    lines.append(f"Paper sentences: {len(paper)}; source sentences: {len(src_sents)}")
    lines.append(f"Flagged: {sum(h[0] >= args.threshold for h in hits)}")
    lines.append("")
    for sim, (fname, sent), key, src in hits:
        if sim < args.threshold:
            break
        lines += [f"## {fname} — cos {sim:.2f} — {key}", "", f"> ours: {sent}", ">", f"> src:  {src}", ""]
    Path(args.out).write_text("\n".join(lines))
    print("\n".join(lines[2:4]))
    print("top-10 similarities:", ", ".join(f"{h[0]:.2f}" for h in hits[:10]))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--corpus", required=True)
    f.add_argument("--force", action="store_true")
    c = sub.add_parser("check")
    c.add_argument("--corpus", required=True)
    c.add_argument("--n", type=int, default=6)
    c.add_argument("--min-run", type=int, default=8)
    c.add_argument("--out", default=str(ROOT / "results" / "overlap_report.md"))
    s = sub.add_parser("semantic")
    s.add_argument("--corpus", required=True)
    s.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    s.add_argument("--threshold", type=float, default=0.80)
    s.add_argument("--out", default=str(ROOT / "results" / "paraphrase_report.md"))
    args = ap.parse_args()
    {"fetch": cmd_fetch, "check": cmd_check, "semantic": cmd_semantic}[args.cmd](args)


if __name__ == "__main__":
    main()
