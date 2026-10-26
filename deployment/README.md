# Isolated CPU deployment

The public repository is https://github.com/Oldmonk-Ji/MP2_submission.
The initial frozen release preserves all 90 hashes in the original completion
manifest. Its canonical checkpoint is 15,824,973 bytes.

## Remote reproduction on 2026-10-27

- CPU: Intel Xeon Platinum 8352V.
- MP2 processes: four CPU cores (affinity 60–63), nice level 10, no GPU.
- Validation: 760/798 using the unchanged official prediction/scoring tools.
- Full 798-question validation: 40.72 seconds prediction plus 0.19 seconds scoring.
- First 467 validation questions: 27.79 seconds prediction plus 0.12 seconds
  scoring in separate fresh processes; total 27.92 seconds.

The 467-question measurement is a **validation timing proxy**, not another test
evaluation or a certification on the course's Xeon 8457C.
See experiments/deployment_20261027/result.json for versions and exact timings.

## New frozen candidate round232

- Validation independently reproduced: **762/798**.
- Full validation prediction/scoring: 42.07 + 0.22 seconds.
- 467-question validation proxy: **26.99 seconds** total.
- Assets: **16,511,713 bytes (15.7468 MiB)**.
- Local standard interface suite: **146 samples, all passed**.
- Remote frozen artifact hashes: **85/85 verified**, canonical and archived
  checkpoint copies identical.
- No TEST evaluation has been performed for this candidate.

Details: experiments/deployment_20261027/round232.json.
Reproduce the guarded validation and proxy timing with:

    nice -n 10 taskset -c 60-63 ../.venv/bin/python deployment/validate_remote.py --round 232 --expected 762

## Directory isolation

The deployment uses its own dase7506-mp2 directory with:

- repo/: project sources, frozen checkpoint, private train/validation data.
- .venv/: isolated Python 3.11 inference and classical-model environment.
- .train-venv/: isolated Python 3.12 environment reusing the already installed
  PyTorch library; all new package installations go into this environment.

Existing robot/ASPIRE repositories and processes are not modified or stopped.
The MP2 training scripts explicitly run on CPU.
The server could not reach GitHub directly, so the identical repository commit
was transferred as a Git bundle and verified after cloning.

## Data boundary

Only the provided TRAIN and validation inputs/references, the frozen model,
and TRAIN/validation research caches were transferred. No test input or test
reference file was included. Existing aggregate frozen test records remain
in the public release as provenance.

The original research data-access audit hook continues to reject test paths.
No credentials are stored in this repository.
