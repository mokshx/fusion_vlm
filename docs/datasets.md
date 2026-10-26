# Source datasets: access, licences, labels

Checked 2026-10-20 against the official pages. Re-check before submission.

| | UP-Fall | MCF (Montréal) | UR Fall | Le2i |
|---|---|---|---|---|
| Official source | https://sites.google.com/up.edu.mx/har-up/ (per-trial Google Drive links) | https://www.iro.umontreal.ca/~labimage/Dataset/ | https://fenix.ur.edu.pl/~mkepski/ds/uf.html | dat@UBFC, doi:10.25666/DATAUBFC-2024-04-09 (`FallDataset.zip`) |
| Licence | **None stated** ("publicly available" + citation request) | **None stated** (citation request) | **CC BY-NC-SA 4.0**, "intended for non-commercial academic use" | **CC BY-NC-SA 3.0**, audience "Research" |
| Format | PNG frames 640×480 ~18 fps, 2 cams (lateral, frontal; both 1.82 m, neither overhead) | MPEG4, 8 wall/ceiling-mounted wide-angle cams, all elevated/oblique | PNG 640×480 30 fps; cam0 side (parallel to floor), cam1 ceiling (overhead, falls only) | AVI 320×240 25 fps, 1 cam |
| Labels | Activity per trial + per-timestamp tags in DataSet CSV | PDF tables (tech report pp. 17–22) + **Nov 2025 erratum** | per-frame cam0 CSV: −1 not lying / 0 falling / 1 lying on ground | per-video txt (fall start/end) for Home & Coffee room only |
| Hard negatives | 8 sitting, 9 picking up object, 11 laying | codes 4 crouching, 7 sitting, 8 lying on sofa | not named; inferred from depth features + visual check | via OmniFall re-annotations |

## Required citations
- UP-Fall: Martínez-Villaseñor et al., "UP-Fall Detection Dataset: A Multimodal Approach", *Sensors* 19(9):1988, 2019.
- MCF: Auvinet et al., "Multiple cameras fall dataset", Tech. Rep. 1350, DIRO, Univ. de Montréal, 2010.
- UR Fall: Kwolek & Kepski, "Human fall detection on embedded platform using depth maps and wireless accelerometer", *Comput. Methods Programs Biomed.* 117(3):489–501, 2014.
- Le2i: Charfi et al., *J. Electron. Imaging* 22(4):041106, 2013; dataset DOI above.
- OmniFall labels (if used): HF `simplexsigil2/omnifall`, CC BY-NC-SA 4.0.

## Issues to resolve before publication
1. **NonCommercial terms vs. company affiliation.** UR Fall, Le2i and OmniFall are NC. If the
   paper is published with a company affiliation or feeds a product, ask for written permission
   (UR Fall: mkepski@ur.edu.pl; Le2i: dat@UBFC contact for Dubois/Miteran).
2. **ShareAlike.** Figures with UR Fall / Le2i frames must be captioned as CC BY-NC-SA and
   excluded from the article's CC BY licence; released label files derived from them must be BY-NC-SA.
3. **No licence (UP-Fall, MCF)** means all rights reserved: evaluate and cite, but release only
   frame indices + scripts, and ask the authors before showing frames in figures.
4. **Faces.** Blur faces in paper figures.
5. **Disclose label gaps.** Le2i Office/Lecture room unannotated; UR Fall labels are cam0 only;
   MCF labels need the erratum; MCF scenario 23 has "falling" codes despite being described fall-free.

## Access notes (practical)
- UR Fall server is slow and throttles parallel connections. Archives support HTTP range
  requests, so `scripts/urfall_fetch_candidates.py` pulls individual PNGs from the remote zips.
- UP-Fall Google Drive links returned "Quota exceeded" for anonymous downloads on 2026-10-20;
  retry later or download through an authenticated Drive account.
- Unofficial Kaggle/HF mirrors exist for all four; several relabel as CC0/Apache/CC-BY, which the
  originals do not permit. Use official sources only.
