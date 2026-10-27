# DASE7506 MP2 - Final frozen submission

**Final method:** round257, verification-aware comparative ranking with a conditional quadratic expert and a small scratch-trained text expert.

| Item | Final record |
| --- | --- |
| Official VAL | **763/798 (95.6140%)** |
| Internal TRAIN-held development fold | **1481/1582** |
| Official TEST | **394/467 (84.3683%)** |
| Required inference assets | **16,686,526 bytes (15.9135 MiB)** |
| Fresh four-thread TEST prediction + scoring | **27.59135 s** on Xeon Platinum 8352V |
| Interface | **146 samples passed** |
| Freeze audit | **89 file hashes unchanged** before/after TEST |
| Student | **纪弘泽 (3036801840)** |
| Deadline | **2026-10-31 23:59 UTC+8** |

Optimization is complete. The final candidate was selected before its one authorized final TEST run, using VAL first and TRAIN-held second. VAL tied round235 while held improved by two. The user explicitly requested this final test and stopping afterwards. No post-TEST parameter or method reselection was performed. Old certified models remain archival records.

## Files to submit

- Source and guarded entrypoint: experiments/round257/frozen/method/entrypoint.py.
- Canonical required checkpoint: experiments/round257/bundle/.
- English final-method report: report/report.pdf (four A4 pages); editable source report/report.md.
- Immutable VAL/interface certificate: experiments/round257/frozen/.
- Authorized final TEST certificate: experiments/round257/official_test_20261027/result.json and completion_manifest.json.
- Public repository: https://github.com/Oldmonk-Ji/MP2_submission.

The report and this document identify the student as 纪弘泽 (3036801840). Publishing the repository has not submitted an MP2 website entry. The measured server is not the course reference Xeon8457C; no reference-machine timing certificate is claimed.

## Reproduce inference

Install requirements.txt in an isolated inference environment. Place the provided course files in DASE7506-MP2-main/code/data/ with the original candidates/ directory and manifest. No external weights or data are needed. Inference does not require Torch.

From the repository root with that environment activated:

```text
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
python DASE7506-MP2-main/code/predict.py \
  --method experiments/round257/frozen/method/entrypoint.py \
  --checkpoint experiments/round257/bundle \
  --split validation --output remote-work/validation_predictions.json
python DASE7506-MP2-main/code/score_predictions.py \
  --predictions remote-work/validation_predictions.json \
  --split validation --output remote-work/validation_score.json
python deployment/verify_release.py --round 257 --git
```

Expected validation count is763. Prediction reads only supplied inputs and checkpoint assets; attempts to read reference answers are blocked. The model returns one candidate index per question, using the unchanged official numeric parser/scorer. It does not use question-ID answer lookups or cached predictions.

The canonical checkpoint and its archival frozen copy are identical; only one copy is loaded for inference. Historical research checkpoints are not required inference assets. The recorded final TEST should not be rerun for routine reproduction.

## Method, training and provenance

The fixed score combines the comparative ranking pipeline, a context-conditioned quadratic expert at weight0.25, and a raw-centered text-network score at weight0.01. It then applies the unchanged parse-and-repair policy (margin4) and conservative strict-gap rule (gap3, top2, gate unchanged). Final experts fit all4743 eligible TRAIN questions under the standing twelve-row TRAIN exclusion; VAL/TEST have no exclusions.

The text network is trained from random word embeddings on provided TRAIN only, seed241241,60 epochs, with a2048-word vocabulary, local convolution, question/candidate attention and group comparison. Its weights useFP16 storage and NumPy float32 inference. No external corpus, pretrained model or inference API is used. Only the course-supplied fixed representations are used by the other experts.

Inference source and exact asset hashes are archived with the frozen method. Research source and complete negative results are retained for audit. Rebuilding the complete historical pipeline requires its intermediate artifacts or rerunning upstream stages; this package guarantees complete frozen inference, not a one-command retraining of every historical stage. PyTorch2.8 is a training-only dependency in requirements-training.txt.

Substantive AI assistance: Claude Code and OpenAI Codex, disclosed in the English report. The data/scorer license and notice are preserved. No other repository or robot/ASPIRE process was modified by the isolated MP2 deployment.