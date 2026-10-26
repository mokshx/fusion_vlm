# Paper: How Small Can You Go?

First draft, IEEEtran journal class (arXiv version; ports to the IEEE Access template).

## Build
```bash
uv run python scripts/make_paper_tables.py   # from repo root: regenerate tables + number macros
cd paper && tectonic main.tex                # or: latexmk -pdf main.tex
```

## Conventions
- **Never type a result by hand.** Numbers come from `generated/numbers.tex` and the
  `generated/*.tex` tables, produced from `results/` by `scripts/make_paper_tables.py`.
- Red `\pending{...}`: depends on experiments not yet run. Blue `\authornote{...}`: author must supply.
  Set `\draftfalse` in `main.tex` before submission; any leftover marker then stops the build.
- Every reference in `references.bib` was checked against a primary source; see
  `references_notes.md`. Do not add a reference without checking it.
- Frames shown in figures: blur faces; caption UR Fall / Le2i frames as CC BY-NC-SA.

## Venue checklist
| | arXiv | IEEE Access | MDPI Sensors |
|---|---|---|---|
| Template | any (this one) | IEEE Access LaTeX template | MDPI LaTeX template |
| Abstract | — | one paragraph, 150–250 words | ≤ 200 words (verify on mdpi.com) |
| Keywords | — | 3–10 | 3–10 |
| Length | — | < 20 pages recommended | (verify) |
| Required | endorsement for first submission (cs.CV) | ORCID, author biographies, AI-use disclosure in Acknowledgment, APC USD 2,160 | Author contributions, funding, data availability, conflicts of interest statements |

## Status
- [x] Full structure, introduction, benchmark, methods, protocol, discussion, limitations, appendix
- [x] Pilot results (Section V-A) from `results/step0/`
- [ ] Related work: VLM-for-fall-detection paragraph and gap statement (needs verified papers)
- [ ] All `\pending` results (RQ1–RQ4), figures, hardware table
- [ ] Author notes: affiliation, conflicts of interest, funding, AI disclosure, repo URL
