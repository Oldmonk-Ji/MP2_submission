# New frozen development candidate: round235

**Validation 763/798 (95.6140%); held 1479/1582; 15.7568 MiB. Official TEST 394/467 (84.3683%).**

The quadratic expert now conditions numerical verification features on the shared semantic context (32 whitened context coordinates crossed with 82 standardized numeric/grounding features). This adds 10,534 checkpoint bytes relative to round232. All projections and weights fit supplied TRAIN only; TRAIN/VAL development selected the new candidate.

Guarded entrypoint: experiments/round235/frozen/method/entrypoint.py. Checkpoint: experiments/round235/bundle/. Remote Xeon 8352V reproduced 763/798; a 467-question validation timing proxy took 24.97 seconds in fresh four-thread prediction/scoring processes. Existing official test scores below belong to their named older frozen candidates.

---

Authorized official TEST prediction plus scoring took **26.78 seconds** on four Xeon 8352V CPU cores. All 85 frozen hashes remained unchanged; the candidate was selected on TRAIN/VAL before this evaluation. See experiments/round235/official_test_20261027/result.json. The report PDF remains the historical round214 report and requires updating before submission.


## Authorized frozen round232 test evaluation

Official TEST: **393/467**; fresh four-thread prediction plus scoring: **27.05 seconds** on Xeon Platinum 8352V. This is not timing certification on the course reference Xeon 8457C. The candidate was selected on TRAIN/VAL before testing. All 85 frozen hashes remained unchanged; no test-based tuning or method selection was performed. See experiments/round232/official_test_20261027/result.json. The original validation-only frozen certificate is retained unchanged.

# HKU DASE7506 MP2 — current research and frozen results

## New frozen candidate: round232

**Validation 762/798; internal held 1479/1582; 15.7468 MiB; 146 interface
samples passed. Official TEST: 393/467 (84.1542%).**

This candidate adds a TRAIN-only quadratic expert over whitened candidate
residuals and shared semantic context to the previous pipeline. Raw score
weight 0.25 was selected using validation development, subject to the held
threshold. It requires no new pretrained model or external data.

Candidate checkpoint: experiments/round232/bundle/.
Guarded prediction entrypoint: experiments/round232/frozen/method/entrypoint.py.
Use the reproduction commands below with round214 replaced by round232;
expected validation is 762. All original scored artifacts are preserved.

Remote Xeon 8352V reproduction: validation 762/798; 467 validation questions
took 26.99 seconds in separate fresh prediction/scoring processes, four
threads. This is a timing proxy, not a new test evaluation.

The existing report PDF documents round214 and must be updated if round232
becomes the final submission. No website submission is implied by this repo.
See deployment/README.md for isolated remote reproduction and timing.

---

# HKU DASE7506 MP2 — frozen round214

Select one of four supplied GSM8K candidate solutions using comparative
ranking and strict numerical verification.

| Recorded result | Correct / total |
| --- | --- |
| Validation | 760 / 798 (95.2381%) |
| Test, from the existing frozen evaluation | 391 / 467 (83.7259%) |
| Internal held training fold | 1477 / 1582 |

The recorded evaluation passed 137 interface/protocol checks. The checkpoint
contains 15,824,973 bytes (15.0919 MiB). The historical local fresh test
prediction plus scoring time was 25.52 seconds on an i7-11800H, four threads.
This does not certify timing on the course reference machine.

## Reproduce frozen inference

Use Python 3.11 or 3.12 and install requirements.txt in a separate environment.
Place the **provided course data** in DASE7506-MP2-main/code/data/, preserving
its original candidates/ subdirectory and manifest. The public source is
[the official course distribution](https://github.com/xudongwu-0/DASE7506-MP2).
This repository carries the unchanged official Python evaluator and data
license/notice. No pretrained weights or external datasets are required.

From the repository root on Linux:

    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    python DASE7506-MP2-main/code/predict.py --method experiments/round214/frozen/method/entrypoint.py --checkpoint experiments/round214/bundle --split validation --output remote-work/validation_predictions.json
    python DASE7506-MP2-main/code/score_predictions.py --predictions remote-work/validation_predictions.json --split validation --output remote-work/validation_score.json

Expected validation count: **760/798**. The frozen entrypoint prevents the
prediction process from opening reference-answer files.

## Contents and provenance

- experiments/round214/bundle/: canonical inference checkpoint.
- experiments/round214/frozen/: original evaluation records, source snapshot,
  hashes, and an identical archival checkpoint copy. Only one checkpoint
  copy is loaded for inference; the duplicate is retained for hash auditing.
- report/report.pdf and report/report.md: existing English report.
- research/: development and training source archive.
- experiments/round224/: subsequent unsuccessful train-held experiment;
  it does not change the frozen submitted candidate.

The training scripts are preserved as a staged research archive. Rebuilding
all stages requires their intermediate artifacts or rerunning the upstream
stages; this snapshot does not claim a single-command from-scratch rebuild.
Frozen inference is complete and does not require retraining.

Student ID is still a placeholder. Website submission has not been performed
by publishing this repository. SUBMISSION_README.md is the original handoff
document; use the commands above for the standalone frozen checkpoint.

The test result above is the already recorded result. Ongoing development
uses training and validation only. A new method must pass validation >760
and internal held >=1477 before a separately authorized test evaluation.
The independent remote deployment received official TEST files only for the separately authorized round232 evaluation.

AI assistance: the original pipeline/report disclosed Claude Code; subsequent
research, packaging and deployment assistance used OpenAI Codex.

## Subsequent TRAIN/VAL continuation

Rounds236-240 evaluated 30 predefined configurations across answer-group evidence, nonlinear semantic regions, question-only context, sparse lexical conditioning and reduced contextual features. No candidate exceeded validation763 while preserving held1479. Round235 remains selected; no additional TEST evaluation was performed. Full outcomes are in experiments/round240/continuation_summary.json.

## Independent raw-text neural research

Rounds241-246 train new word/CNN/cross-attention networks and a two-layer Transformer from random initialization, including TRAIN-only masked-language and number-binding pretraining. Exported research checkpoints are retained separately from the selected bundle. The best complementary TRAIN-held result is1481/1582, but validation remains763/798, so round235 remains selected and no further TEST evaluation was performed. See deployment/RAW_TEXT_MODELS.md and experiments/round246/continuation_summary.json.
