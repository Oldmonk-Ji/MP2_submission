# DASE7506 MP2 — Submission Package

> **2026-10-27 update: the pre-test selected frozen candidate is round232.**
> Official validation **762/798**, held **1479/1582**, authorized official test
> **393/467 (84.1542%)**. Assets **16,511,713 bytes (15.7468 MiB)**;
> fresh four-thread Xeon 8352V prediction plus scoring **27.05 s**.
> Use experiments/round232/frozen/method/entrypoint.py and
> experiments/round232/bundle with the unchanged official predict.py.
> Certification: experiments/round232/official_test_20261027/result.json.
> All 85 frozen hashes remained unchanged. Test was not used for tuning or
> selection. The report PDF and the historical package details below still
> describe round214; update the report before website submission.

**Method:** Verification-Aware Comparative Ranking (frozen round214, arm `strict_gap_t3`)
**Official scores:** validation **760/798 (95.24%)**, test **391/467 (83.73%)**
**Student ID:** *(to be filled)* · **Deadline:** 2026-10-31 23:59 UTC+8

This repository contains the code, checkpoints, official evaluation JSONs, and the report for the MP2 submission. The frozen method reads the question, the four candidate texts, and their fixed representations, and returns exactly one candidate index per question. It uses no IDs, no lookup tables, no cached predictions, no external data or models.

## Repository layout

```
research/                          # all training/selection/freeze code
  round214_selector.py             # frozen prediction interface (predict(bank, arrays, checkpoint))
  round214_freeze.py               # rebuilds the frozen bundle + validation cache (reproducibility)
  feature_validation.py            # official-style validation harness (verifier v5)
  round22_common.py                # shared data loading, 3-fold split, 12 excluded questions
  round3_features.py, round4_gates.py, round4_train.py
                                   # base expert training + parse-and-repair gate
  round68_selector.py, round79_selector.py, round80_train.py, round80_selector.py,
  round82_relations.py, round103_features.py, round103_train.py,
  round136_features.py, round209_trace.py, round210_features.py, round210_fit.py
                                   # comparative channels + strict verification trace
  ...                               # additional ablation/experiment scripts (see plan.json per round)
report/report.pdf (report.md)      # final report (9 pages)
experiments/round214/bundle/       # inference checkpoint (15,824,973 bytes = 15.09 MiB)
experiments/round214/frozen/       # official evaluation artifacts (verifier v5)
experiments/final_evaluation_20261021/  # historical baseline (test 386/467)
DASE7506-MP2-main/                 # official distribution (data, scorer, guide) — unchanged
```

## Installation

```text
# Windows / Python 3.11
python -m venv .venv
.\.venv\Scripts\pip install numpy scipy scikit-learn lightgbm threadpoolctl
```

Dependencies: `numpy`, `scipy`, `scikit-learn` (liblinear LogisticRegression), `lightgbm` (4.x), `threadpoolctl`. The official data and scorer (protocol v5) are used unchanged from `DASE7506-MP2-main/`.

## Reproducing the frozen result

1. Rebuild the frozen bundle and validation prediction cache:

```text
.\.venv\Scripts\python.exe research\round214_freeze.py
```

This assembles `experiments/round214/bundle` from the trained assets (8 base experts, fusion model, four comparative channel weights, strict weight, configuration), re-runs the selector on validation, and asserts the result equals the frozen 760/798.

2. Run the frozen evaluation (validation + test, official scorer):

```text
.\.venv\Scripts\python.exe research\feature_validation.py ^
  --selection experiments/round214/final_feature_selection.json ^
  --method research/round214_selector.py ^
  --output experiments/round214/frozen --with-test
```

Expected output: validation 760/798; test 391/467; 137/137 harness checks passed; fresh test prediction + scoring ≈ 25.5 s on an i7-11800H with 4 threads (limit 30 s on the reference Xeon 8457C).

The official evaluation JSONs are committed at `experiments/round214/frozen/` (`test_score.json`, `validation_score.json`, `summary.json`, `completion_manifest.json`, `interface_checks.json`).

## Training the components from scratch

All components train on CPU with 4 threads (training seeds and per-stage costs are recorded in each round's `plan.json` / `training_complete.json`):

| Component | Entry script | Train data | Time (CPU) | Seed |
|:----------|:-------------|:-----------|-----------:|-----:|
| 8 base experts (MLPs + TF-IDF) | `research/round4_train.py` + round25 recipe | 4,743 questions (12 excluded) | minutes per model | 20261014, 25025, 25026 |
| Question-context projection (PCA-64) | round79 pipeline | train | 39 s | 79079 |
| Quantity-flow contrastive weight | `research/round80_train.py` | train candidate pairs | 207 s | 80080 |
| Relation classifier (6 ops) | `research/round82_relations.py` | train | 199 s | — |
| Local structure weight | `research/round103_train.py` | train | 91 s | — |
| Strict-trace contrastive weight | `research/round210_fit.py` | train candidate pairs | 159 s | 80080 |
| LambdaRank fusion (200 trees × 7 leaves) | fusion stages in rounds 79–104 | train folds 0–1 | ~6 s | 68068 |
| Gap-rule threshold τ = 3.0 | `research/round213_scan.py`, `round214_compose.py` | train folds (out-of-fold scores) | — | — |

Training protocol: fold 2 of the shared 3-fold split (1,582 questions) is never used for training the fusion or strict scorer and serves as internal held-out sanity (1477/1582). The 12 excluded defective/ambiguous question rows (0-based): 34, 56, 1442, 1445, 1854, 2174, 2351, 2912, 4296, 4331, 4382, 4644.

## Checkpoint asset manifest (SHA-256)

Bundle total: **15,824,973 bytes (15.09 MiB) ≤ 16 MiB**. Files under `experiments/round214/bundle/`:

| File | Bytes | SHA-256 (first 32 hex) |
|:-----|------:|:-----------------------|
| baseline/member0/model.npz | 1,632,888 | 4d89be7794d1ee1481f1a0bf81bedd4e |
| baseline/member1/model.npz | 1,680,728 | 8ac034bf797c6acefcc294fbc818f137 |
| baseline/member2/model.npz | 2,730,088 | 83f6f04a260623fb7aa2db91e3bcaa84 |
| baseline/member3/model.npz | 400,504 | 13b7523109ba7f6ca0bf3fe2457e879f |
| baseline/member3/vocabulary.json | 1,259,237 | 2575d8a98c210a36db6b4f536f12ebe9 |
| baseline/member4/model.npz | 1,267,544 | 88c0b018599870cc64dcd985d9ad49c4 |
| baseline/member5/model.npz | 1,267,544 | 37f0df66959b0ea01d8a6514210509de |
| baseline/member6/model.npz | 2,058,600 | 872525e4f23403d7c4cb6549bd9a28a8 |
| baseline/member7/model.npz | 2,058,600 | 90826629f983994fa00f98fe8f5ee2fb |
| meta.txt (LambdaRank) | 214,246 | 5ec100c58e1e47b29873df0e60830555 |
| flow.npz | 65,802 | d1c9bde59cd5f5360d048a8d3edff456 |
| relation.npz | 393,746 | ddeb9e84bb8ab1ad8ec1a6be17b3b044 |
| space.npz | 267,236 | 0cd9fd0f5ed16e27489f04b802df1c5b |
| strict.npz | 262,410 | de01bfdb612c1ae4fc4c277d35b716e0 |
| structure.npz | 262,410 | e1763f5db88874f4aa61fe37938d5c1e |
| config.json + member configs | < 1 KB | see completion_manifest.json |

Full hashes for every file (including all config.json files) are in `experiments/round214/frozen/completion_manifest.json`.

## Resource use

- **Inference assets:** 15.09 MiB uncompressed (limit 16 MiB).
- **Runtime** (local i7-11800H, 4 threads, fresh process incl. startup/imports/loading): test prediction 25.07 s + scoring 0.46 s = 25.5 s (limit 30 s on reference Xeon 8457C); validation prediction 36.9 s.
- **Training/search cost:** all components train in seconds to minutes on CPU (table above). Development search included many ablation rounds whose scripts and JSON records are preserved under `research/` and `experiments/` with per-round `plan.json` (hypothesis + hashes) and results.

## AI assistance and reuse disclosure

Substantive AI coding assistance (Claude Code) was used for feature design, training loops, ablations, and the report; all code was validated against the official harness before freezing. Reused work: operation-relation supervision adapted from Roy & Roth (2015, D15-1202); GSM8K-derived data and the official scorer are the course distribution (MIT). No other external data, models, or APIs.

## Report

`report/report.pdf` — final report, 9 pages: method, ablations, error analysis, resources, reproducibility, disclosure.
