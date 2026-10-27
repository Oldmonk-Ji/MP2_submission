# Final frozen candidate: round257

**VAL763/798; held1481/1582; 15.9135 MiB;146 interface samples passed.**

Official TEST: **394/467 (84.3683%)**; fresh four-thread Xeon8352V prediction plus scoring **27.59 s**.

Selected before TEST by TRAIN/VAL: round235 plus the scratch raw-text expert at centered-logit weight0.01, before the unchanged parse/repair and strict rules. The user authorized one final test despite validation tying round235 and requested that optimization stop afterwards. No post-test parameter or method reselection is permitted.

Entrypoint: experiments/round257/frozen/method/entrypoint.py; checkpoint: experiments/round257/bundle/. The final English report is report/report.pdf (four A4 pages), and FINAL_RESULT.json records the completed freeze. Student ID remains to be filled; the MP2 website submission has not been performed. Older method scores are retained only in their named archives.

---

## Reproduce frozen inference

Install requirements.txt in an isolated Python environment. Place the provided course files under DASE7506-MP2-main/code/data/ with the original candidates/ folder and manifest. No additional data/models/API are needed; Torch is training-only.

From the repository root with the inference environment activated:

    export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
    python DASE7506-MP2-main/code/predict.py --method experiments/round257/frozen/method/entrypoint.py --checkpoint experiments/round257/bundle --split validation --output remote-work/validation_predictions.json
    python DASE7506-MP2-main/code/score_predictions.py --predictions remote-work/validation_predictions.json --split validation --output remote-work/validation_score.json
    python deployment/verify_release.py --round 257 --git

Expected validation:763/798. The unchanged official scorer determines numeric correctness. Prediction reference reads are blocked. Only the canonical checkpoint is required; duplicate frozen and research checkpoints are archival and not loaded for inference.

## Final submission files

- [English report PDF](report/report.pdf) and [editable source](report/report.md).
- [Submission instructions](SUBMISSION_README.md).
- [Final result record](FINAL_RESULT.json).
- [Official TEST certificate](experiments/round257/official_test_20261027/result.json).
- [VAL/interface freeze](experiments/round257/frozen/summary.json).

Student ID remains a placeholder. Fill it in before website submission; the MP2 website entry has not been submitted by publishing this repository. Deadline:2026-10-31 23:59 UTC+8. Timing is measured on Xeon8352V and is not reference-Xeon8457C certification.

## Provenance and research archive

The pre-TEST snapshot is a41d8ab / round257-val763-held1481. The final method was chosen on TRAIN/VAL before its single authorized last TEST run. Model optimization is stopped. The exact source closure and89 frozen hashes are preserved; TEST was not used to tune or reselect the final method. No routine TEST rerun is needed.

Research source, cross-fitted text checkpoints, full negative-result records and historical certified methods remain under research/ and experiments/. See the Chinese handoff and deployment/RAW_TEXT_MODELS.md. Training-only large intermediates can require regeneration; complete frozen inference does not require retraining.

AI assistance by Claude Code and OpenAI Codex is disclosed in the report. The official data/scorer license and notice are preserved.
