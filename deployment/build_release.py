"""Build an allowlisted public snapshot and a private train/validation payload."""
from pathlib import Path
import hashlib
import json
import shutil
import tarfile
import sys

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "release" / "MP2_submission"
PRIVATE = ROOT / ".deploy-private"


def copy(path):
    source = ROOT / path
    target = DEST / path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    PRIVATE.mkdir(exist_ok=True)
    for folder in ("research", "starter"):
        for p in (ROOT / folder).glob("*.py"):
            copy(p.relative_to(ROOT))
    for name in ("SUBMISSION_README.md", "report/report.md", "report/report.pdf",
                 "交接文档_2026-10-27.md"):
        copy(name)
    for p in (ROOT / "DASE7506-MP2-main/code").glob("*.py"):
        copy(p.relative_to(ROOT))
    for p in (ROOT / "DASE7506-MP2-main/guide").glob("*.md"):
        copy(p.relative_to(ROOT))
    for name in ("LICENSE.txt", "NOTICE.txt", "manifest.json"):
        copy(Path("DASE7506-MP2-main/code/data") / name)
    for p in (ROOT / "experiments/round214").rglob("*"):
        if p.is_file() and "__pycache__" not in p.parts:
            copy(p.relative_to(ROOT))
    if (ROOT / "experiments/round232/frozen/completion_manifest.json").exists():
        for p in (ROOT / "experiments/round232").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                copy(p.relative_to(ROOT))
    if (ROOT / "experiments/round235/frozen/completion_manifest.json").exists():
        for p in (ROOT / "experiments/round235").rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                copy(p.relative_to(ROOT))
    for r in ("round22", "round77", "round210", "round216", "round218",
              "round219", "round220", "round221", "round222", "round223", "round224",
              "round225", "round226", "round227", "round228", "round229", "round230",
              "round231", "round233", "round234", "deployment_20261027"):
        for p in (ROOT / "experiments" / r).glob("*.json"):
            if p.stat().st_size < 1_000_000:
                copy(p.relative_to(ROOT))
    for p in (ROOT / "deployment").iterdir():
        if p.is_file() and p.suffix in (".py", ".sh", ".md"):
            copy(p.relative_to(ROOT))
    (DEST / ".gitattributes").write_text("* -text\n", encoding="utf-8")
    (DEST / ".gitignore").write_text(
        ".venv/\n__pycache__/\n*.pyc\n.deploy-private/\nremote-work/\n"
        "DASE7506-MP2-main/code/data/gsm8k_*.jsonl\n"
        "DASE7506-MP2-main/code/data/candidates/\n"
        "experiments/round216/cache.npz\n"
        "experiments/round77/nested/\n", encoding="utf-8")
    (DEST / "requirements.txt").write_text(
        "numpy>=1.26\nscipy>=1.11\nscikit-learn>=1.4\n"
        "lightgbm>=4.0,<5\nthreadpoolctl>=3.1\n", encoding="utf-8")
    (DEST / "README.md").write_text("""# HKU DASE7506 MP2 — frozen round214

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
The remote deployment deliberately omits every test input/reference file.

AI assistance: the original pipeline/report disclosed Claude Code; subsequent
research, packaging and deployment assistance used OpenAI Codex.
""", encoding="utf-8")
    if (ROOT / "experiments/round232/frozen/completion_manifest.json").exists():
        readme = DEST / "README.md"
        readme.write_text("""# HKU DASE7506 MP2 — current research and frozen results

## New frozen candidate: round232

**Validation 762/798; internal held 1479/1582; 15.7468 MiB; 146 interface
samples passed. Its TEST score has not been measured.**

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

""" + readme.read_text(encoding="utf-8"), encoding="utf-8")
    test_record = ROOT / 'experiments/round232/official_test_20261027/result.json'
    if test_record.exists():
        result = json.loads(test_record.read_text(encoding='utf-8'))
        readme = DEST / 'README.md'
        contents = readme.read_text(encoding='utf-8')
        contents = contents.replace('Its TEST score has not been measured.',
            f"Official TEST: {result['score']['correct']}/467 ({100*result['score']['correct']/467:.4f}%).")
        contents = contents.replace('The remote deployment deliberately omits every test input/reference file.',
            'The independent remote deployment received official TEST files only for the separately authorized round232 evaluation.')
        certificate = (f"\n## Authorized frozen round232 test evaluation\n\n"
            f"Official TEST: **{result['score']['correct']}/467**; fresh four-thread prediction "
            f"plus scoring: **{result['total_seconds']:.2f} seconds** on Xeon Platinum 8352V. "
            "This is not timing certification on the course reference Xeon 8457C. "
            "The candidate was selected on TRAIN/VAL before testing. All 85 frozen hashes "
            "remained unchanged; no test-based tuning or method selection was performed. "
            "See experiments/round232/official_test_20261027/result.json. "
            "The original validation-only frozen certificate is retained unchanged.\n\n")
        readme.write_text(certificate + contents, encoding='utf-8')
    candidate_summary = ROOT / 'experiments/round235/frozen/summary.json'
    if candidate_summary.exists():
        summary = json.loads(candidate_summary.read_text(encoding='utf-8'))
        readme = DEST / 'README.md'
        prefix = ("# New frozen development candidate: round235\n\n"
            f"**Validation {summary['validation_correct']}/798 (95.6140%); held 1479/1582; "
            f"{summary['asset_MiB']:.4f} MiB. TEST has not been evaluated for this candidate.**\n\n"
            "The quadratic expert now conditions numerical verification features on the "
            "shared semantic context (32 whitened context coordinates crossed with 82 "
            "standardized numeric/grounding features). This adds 10,534 checkpoint bytes "
            "relative to round232. All projections and weights fit supplied TRAIN only; "
            "TRAIN/VAL development selected the new candidate.\n\n"
            "Guarded entrypoint: experiments/round235/frozen/method/entrypoint.py. "
            "Checkpoint: experiments/round235/bundle/. Remote Xeon 8352V reproduced "
            "763/798; a 467-question validation timing proxy took 24.97 seconds in "
            "fresh four-thread prediction/scoring processes. Existing official test scores "
            "below belong to their named older frozen candidates.\n\n---\n\n")
        readme.write_text(prefix + readme.read_text(encoding='utf-8'), encoding='utf-8')
    manifest = {}
    for p in DEST.rglob("*"):
        if p.is_file() and ".git" not in p.parts and p.name != "release_manifest.json":
            rel = p.relative_to(DEST).as_posix()
            manifest[rel] = {"bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    (DEST / "release_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    payload = PRIVATE / "mp2_train_validation_payload.tar.gz"
    if "--public-only" in sys.argv:
        print(json.dumps({"public_files":len(manifest),
                          "public_bytes":sum(v["bytes"] for v in manifest.values()),
                          "public_dir":str(DEST)}))
        return
    with tarfile.open(payload, "w:gz", compresslevel=1) as tar:
        data = ROOT / "DASE7506-MP2-main/code/data"
        for split in ("train", "validation"):
            for name in (f"gsm8k_{split}.jsonl",
                         f"candidates/gsm8k_{split}.jsonl",
                         f"candidates/gsm8k_{split}.npz"):
                p = data / name
                tar.add(p, arcname=p.relative_to(ROOT).as_posix())
        for name in ("experiments/round216/cache.npz",
                     "experiments/round77/nested/components.npz"):
            tar.add(ROOT / name, arcname=name)
    print(json.dumps({"public_files": len(manifest), "public_bytes": sum(
        v["bytes"] for v in manifest.values()), "public_dir": str(DEST),
        "private_payload_bytes": payload.stat().st_size, "test_files": 0}))


if __name__ == "__main__":
    main()
