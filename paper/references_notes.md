# Reference verification notes

All entries in `references.bib` were checked on 2026-10-26. Format: `key`: what it supports in the paper. Verified at: URL.

## A. Datasets
- `kwolek2014urfall`: UR Fall Detection dataset (source dataset). Verified at: https://doi.org/10.1016/j.cmpb.2014.09.005 (Crossref metadata)
- `martinez2019upfall`: UP-Fall Detection dataset (source dataset). Verified at: https://doi.org/10.3390/s19091988 (Crossref)
- `auvinet2010mcfd`: Multiple Cameras Fall Dataset, Montreal (source dataset). The dataset page asks for exactly this citation ("Technical report 1350, DIRO - Université de Montréal, July 2010"). Verified at: https://web.archive.org/web/2024/http://www.iro.umontreal.ca/~labimage/Dataset/ (the live page is behind an anti-bot wall)
- `charfi2013le2i`: Le2i fall dataset, original paper. Verified at: https://doi.org/10.1117/1.JEI.22.4.041106 (Crossref; article number 041106)
- `dubois2014le2idata`: Le2i dataset as deposited on dat@UBFC (creators Dubois and Miteran, publicationYear 2014 in DataCite). Verified at: https://api.datacite.org/dois/10.25666/DATAUBFC-2024-04-09 and https://search-data.ubfc.fr/FR-13002091000019-2024-04-09_Fall-Detection-Dataset.html
- `schneider2025omnifall`: OmniFall unified benchmark and label taxonomy. Also related work: it reports zero-shot MLLMs confusing "fallen" with "lying". Verified at: https://arxiv.org/abs/2505.19889 (v3 has 9 authors; the HF card's BibTeX lists only 6, so follow arXiv) and https://huggingface.co/datasets/simplexsigil2/omnifall

## B. Surveys and reviews
- `mubashir2013survey`: classic survey of fall-detection approaches (wearable, ambient, vision). Verified at: https://doi.org/10.1016/j.neucom.2011.09.037
- `igual2013challenges`: challenges and trends in fall-detection systems (e.g. lack of real-world data, staged falls). Verified at: https://doi.org/10.1186/1475-925X-12-66
- `gutierrez2021review`: review of vision-based fall-detection systems. Verified at: https://doi.org/10.3390/s21030947
- `alam2022review`: review of deep-learning vision-based fall detection. Verified at: https://doi.org/10.1016/j.compbiomed.2022.105626

## C. Pose estimation and pose-based fall detection
- `cao2021openpose`: OpenPose, bottom-up 2D pose estimation. Verified at: https://doi.org/10.1109/TPAMI.2019.2929257 (TPAMI 43(1), 2021; online 2019)
- `maji2022yolopose`: YOLO-Pose, the single-stage keypoint detector family behind the baseline. Verified at: https://doi.org/10.1109/CVPRW56347.2022.00297 (IEEE pages 2636-2645; the CVF open-access PDF numbers them 2637-2646)
- `jocher2024yolo11`: Ultralytics YOLO11 software citation, for a yolo11*-pose baseline. Verified at: https://docs.ultralytics.com/models/yolo11/ (BibTeX from the docs source, github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo11.md)
- `jocher2026yolo26`: Ultralytics YOLO26 paper, the current preferred citation in the repo's CITATION.cff, for a yolo26*-pose baseline. Verified at: https://github.com/ultralytics/ultralytics/blob/main/CITATION.cff and https://arxiv.org/abs/2606.03748. Use only the entry that matches the model you actually run.
- `chen2020openposefall`: rule-based fall detection from OpenPose keypoints (hip-centre descent speed, body-centreline angle, width/height ratio). Precedent for the keypoint and torso-angle rule baseline. Verified at: https://doi.org/10.3390/sym12050744
- `ramirez2021skeleton`: fall detection and activity recognition from skeleton (pose) features, evaluated on UP-Fall. Verified at: https://doi.org/10.1109/ACCESS.2021.3061626
- `pereira2024industrialfall`: YOLOv8 fall detection aimed at industrial setups (detector-based, not pose rules). Verified at: https://arxiv.org/abs/2408.04605

## D. Vision-language models
- `radford2021clip`: CLIP. Verified at: https://proceedings.mlr.press/v139/radford21a.html
- `liu2023llava`: LLaVA / visual instruction tuning. Verified at: https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html
- `li2023blip2`: BLIP-2. Verified at: https://proceedings.mlr.press/v202/li23q.html
- `tschannen2025siglip2`: SigLIP 2 vision encoder (used in small VLMs such as LFM2-VL). Verified at: https://arxiv.org/abs/2502.14786
- `beyer2024paligemma`: PaliGemma, a small (3B) transfer VLM. Verified at: https://arxiv.org/abs/2407.07726
- `marafioti2025smolvlm`: SmolVLM, a small efficient VLM. Verified at: https://arxiv.org/abs/2504.05299
- `chu2023mobilevlm`: MobileVLM, a VLM designed for mobile devices. Verified at: https://arxiv.org/abs/2312.16886
- `bai2025qwen25vl`: Qwen2.5-VL technical report (Qwen VLM lineage). Verified at: https://arxiv.org/abs/2502.13923
- `qwen2026qwen35`: Qwen3.5 models under test (2B, 4B). The official citation is the blog. There is no Qwen3.5 technical report; the only Qwen3.5-titled tech report on arXiv is Qwen3.5-Omni (2604.15804), which is a different model. Big models came out 2026-02-24 and the 0.8B/2B/4B/9B models on 2026-03-02. Verified at: https://huggingface.co/Qwen/Qwen3.5-2B (Citation section) and https://github.com/QwenLM/Qwen3.5 (README "News" and citation; the repo now redirects to QwenLM/Qwen3.8)
- `amini2025lfm2`: LFM2 technical report, covering the architecture and LFM2-VL. This is the paper cited by the LFM2.5-VL-450M and 1.6B model cards. Title confirmed as "LFM2 Technical Report" with 33 authors (first: Alexander Amini). Verified at: https://arxiv.org/abs/2511.23404 and https://huggingface.co/LiquidAI/LFM2.5-VL-450M
- `liquidai2025lfm2vl`: LFM2-VL release blog (2025-08-12). Verified at: https://www.liquid.ai/blog/lfm2-vl-efficient-vision-language-models
- `liquidai2026lfm25`: LFM2.5 release blog (2026-01-05), which announces LFM2.5-VL-1.6B. Verified at: https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai
- `liquidai2026lfm25vl450m`: LFM2.5-VL-450M release blog (2026-04-08). Verified at: https://www.liquid.ai/blog/lfm2-5-vl-450m
- `liquidai2026lfm25vl3b`: LFM2.5-VL-3B release blog (2026-08-12). The HF model card gives this as the citation. Verified at: https://www.liquid.ai/blog/lfm2-5-vl-3b and https://huggingface.co/LiquidAI/LFM2.5-VL-3B

## E. Hallucination, yes-bias and prompt sensitivity
- `li2023pope`: POPE yes/no object-probing benchmark. It reports that LVLMs over-answer "yes", which motivates the analysis of yes-rate and false positives. Verified at: https://doi.org/10.18653/v1/2023.emnlp-main.20
- `guan2024hallusionbench`: HallusionBench, a yes/no-style diagnostic with control-group question pairs that analyses response tendencies and language priors. Verified at: https://doi.org/10.1109/CVPR52733.2024.01363
- `zong2024permutations`: prompt sensitivity. Simple permutations of answer options change (V)LM answers. Verified at: https://proceedings.mlr.press/v235/zong24b.html
- `chan2026yesbias`: explicitly names "yes-bias" as a common VLM hallucination, where models answer "yes" indiscriminately, and links it to system-prompt attention. Verified at: https://arxiv.org/abs/2601.12430

## F. VLMs and (M)LLMs for fall detection and workplace safety
- `junca2026thermal`: zero-shot Gemini Flash screening of low-resolution thermal clips for falls. One prompt sentence materially changed sensitivity, and identical requests disagreed on 4% of windows. Directly relevant to the prompt-sensitivity findings. Verified at: https://arxiv.org/abs/2610.29913
- `shahabian2026vlmfall`: zero-shot VLM prompting (contrastive prompt bank) and few-shot linear probe for fall detection on an assistive robot, compared against skeleton baselines and a rule-based heuristic. Accepted at ICSR+Art 2026 (LNCS), in press. Verified at: https://researchprofiles.herts.ac.uk/en/publications/visionlanguage-model-for-fall-detection-in-socially-assistive-rob/
- `schneider2025omnifall` (also in A): zero-shot MLLMs are compared with fine-tuned models on in-the-wild falls; the zero-shot models confuse fallen with lying.
- `yao2026surveillance`: zero-shot MLLMs on video anomaly detection show a conservative bias toward "normal" (high precision, recall collapse), and class-specific prompts shift the decision boundary. A useful contrast to yes-bias. Verified at: https://arxiv.org/abs/2603.04727
- `adil2026svlm`: six small (<4B) VLMs (Qwen3-VL 2B/4B, InternVL3, SmolVLM, Gemma-3 4B) evaluated zero-shot for construction hazard identification, with YOLO11n detection-guided prompts. Closest to "small VLM for site safety". Verified at: https://arxiv.org/abs/2604.05210
- `chen2026constructionsite10k`: ConstructionSite 10k benchmark; large VLMs evaluated zero-shot and few-shot as construction safety inspectors. Verified at: https://doi.org/10.1017/dce.2026.10044
- `bui2026construction`: GPT-4o, Florence-2 and LLaVA-1.5 recognising construction worker actions and emotions from site images. Verified at: https://arxiv.org/abs/2601.10835
- `yarrabothula2026steelbench`: benchmark of nine VLMs on real steel-plant CCTV (activity and safety-rule reasoning). The best model reaches 42.6% action accuracy against an 84.6% human score. Verified at: https://arxiv.org/abs/2607.05264
- `chen2024clip2safety`: Clip2Safety, a VLM pipeline for PPE compliance across workplace scenarios. Verified at: https://arxiv.org/abs/2408.07146
- `wu2025monitorvlm`: MonitorVLM, VLM safety-violation detection from mining surveillance video. Verified at: https://arxiv.org/abs/2510.03666
- `mithsara2025slmfall`: construction fall-detection system with small language models (TinyLlama, Phi-3) for emergency communication. The LMs handle communication, not visual detection. Verified at: https://doi.org/10.22260/ISARC2025/0077 and https://www.iaarc.org/publications/fulltext/077_Development_of_a_Fall_Detection_and_Safety_Communication_System_Using_Small_Language_Models.pdf

## G. Quantisation and on-device inference
- `dettmers2022llmint8`: 8-bit inference (LLM.int8()). Note that the NeurIPS proceedings print the title as "GPT3.int8(): 8-bit Matrix Multiplication for Transformers at Scale", while arXiv:2208.07339 uses "LLM.int8()". The bib uses the proceedings title. Swap it if you cite the arXiv version. Verified at: https://proceedings.neurips.cc/paper_files/paper/2022/hash/c3ba4962c05c49636d4c6206a97e9c8a-Abstract-Conference.html
- `frantar2023gptq`: GPTQ post-training weight quantisation. Venue (ICLR 2023) taken from the authors' arXiv comment; OpenReview (forum tcbBPnfwxS) could not be fetched because of a bot check. Verified at: https://arxiv.org/abs/2210.17323
- `lin2024awq`: AWQ. The proceedings title includes "On-Device", unlike the arXiv title. Verified at: https://proceedings.mlsys.org/paper_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html
- `gerganov2023llamacpp`: llama.cpp / GGUF runtime used for all VLM runs. Add the exact release tag or commit you used; the latest release seen on 2026-10-26 was v0.7.0. Verified at: https://github.com/ggml-org/llama.cpp (GitHub API: created 2023-03-10; top contributor ggerganov = Georgi Gerganov)
- `laskaridis2024melt`: systematic on-device LLM benchmarking (Android, iOS, NVIDIA Jetson), covering latency, energy and the accuracy cost of quantisation. Verified at: https://doi.org/10.1145/3636534.3690668
- `taherin2026hydra`: phase-aware (prefill/decode) characterisation of LLM inference on Jetson AGX Xavier, Orin and Thor with llama.cpp and HF Transformers, across quantisation levels. Verified at: https://arxiv.org/abs/2608.25053

## H. Benchmarking methodology
- `reddi2020mlperf`: MLPerf Inference latency and throughput methodology. Verified at: https://doi.org/10.1109/ISCA45697.2020.00045
- `guo2017calibration`: calibration and ECE. Verified at: https://proceedings.mlr.press/v70/guo17a.html
- `efron1993bootstrap`: bootstrap confidence intervals. Verified at: https://doi.org/10.1007/978-1-4899-4541-9 (Crossref: Springer US record dated 1993 for the Chapman & Hall edition; a 1994 CRC re-issue has DOI 10.1201/9780429246593)

## I. Workplace fall statistics
- `bls2026cfoi`: US BLS CFOI 2024 (news release USDL-26-0230, 19 Feb 2026). Quotable text: "Fatal falls, slips, and trips decreased 4.6 percent to 844 in 2024 from 885 in 2023." Total fatal work injuries in 2024 were 5,070, so falls, slips and trips are 844/5,070 = 16.6% (computed). Transportation incidents were the most frequent fatal event at 38.2%, so for the US say "a leading cause", not "the leading cause". Among construction and extraction workers, fatal falls, slips and trips numbered 370 in 2024. Verified at: https://www.bls.gov/news.release/cfoi.nr0.htm
- `hse2026fatal`: UK HSE, 2025/26 (provisional, RIDDOR). 126 workers were killed in work-related accidents; 31 of them died from falls from a height, "the most common kind of fatal accident ... accounting for around a quarter of fatal injuries to workers in 2025/26". Verified at: https://www.hse.gov.uk/statistics/fatals-overview.htm (updated 2026-06-30)

## J. Lone worker / man-down
- `hse2020loneworkers`: HSE INDG73(rev4), published 03/20, ISBN 978-0-7176-6729-1. It recommends "devices designed to raise the alarm in an emergency which can be operated manually or automatically". This is the official UK basis for automatic lone-worker / man-down alarms. Verified at: https://www.hse.gov.uk/pubns/indg73.pdf

## K. Hardware
- `nvidia_orin_nano_super`: Jetson Orin Nano (Super) Developer Kit. The page lists 67 INT8 TOPS (up from 40), 8 GB 128-bit LPDDR5, 102 GB/s (up from 68), 7-25 W, and an Ampere GPU with 1024 CUDA cores and 32 tensor cores. Verified at: https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/
- `nvidia_orin_nano_userguide`: official developer-kit user guide (setup and JetPack). Verified at: https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/quick_start.html

## Could not verify
- **BS 8484:2022** ("Provision of lone worker services. Code of practice"). BSI's own product page only showed the draft-for-comment record (21/30435721 DC, 3 Nov 2021), and the published-standard page is client-rendered ("Product not found"). The 2022 edition, its supersession of BS 8484:2016 and its effective date (31 Mar 2022) are confirmed only by secondary sources (NSI circular NSI 016 22, vendor blogs). It was left out of the bib. Add it only after checking the BSI shop page or a library copy for the publication date and ISBN.
- **Qwen3.5 technical report**: none exists on arXiv (only Qwen3.5-Omni, 2604.15804, a different model). The blog citation is used instead. The qwen.ai blog page is rendered in JavaScript, so its text could not be read; the title and date come from the official HF model cards and the QwenLM GitHub README.
- **A separate LFM2-VL or LFM2.5-VL technical report**: none found. LFM2-VL is covered inside the LFM2 Technical Report (arXiv:2511.23404). The LFM2.5-VL-3B card cites only the blog, and its HF tags reference arXiv:2305.03393, which is unrelated to the model and was not used.
- **GPTQ OpenReview page**: blocked by a bot check, so the ICLR 2023 venue comes from the authors' arXiv comment.
- **chan2026yesbias as ACL Findings 2026**: the proceedings version is not yet indexed in Crossref, so it is cited as arXiv.
- **shahabian2026vlmfall**: accepted and in press (LNCS). There is no DOI or pages yet, and the institutional page does not name the VLM backbone. Update before submission.
- **UK HSE non-fatal figure** (slips, trips or falls on the same level as the most common non-fatal injury): the HSE non-fatal overview pages returned 404, so it is not quoted.
- **BLS archived PDF** (https://www.bls.gov/news.release/archives/cfoi_02192026.pdf): HTTP 403 to automated fetch. The HTML release at cfoi.nr0.htm was verified instead. Once BLS publishes the 2025 CFOI (around Dec 2026 to Feb 2027), the cfoi.nr0.htm URL will point to newer data, so keep the release number USDL-26-0230.
- **Jetson Orin Nano product-page date**: the page carries no publication date, so the entry has no year and uses an access date.
- **Eurostat ESAW figure**: not searched in depth. The BLS and HSE figures were judged sufficient.
- **A peer-reviewed zero-shot small-VLM study for falls in an industrial setting**: none found. Most fall-specific VLM/MLLM work is in eldercare or homes (Junca 2026, Shahabian Alashti 2026, OmniFall), and the industrial VLM work targets PPE or hazards rather than person-down. This gap supports the paper's novelty claim.
