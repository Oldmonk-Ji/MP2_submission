"""Fresh-process frozen validation and a clearly labeled 467-row timing proxy."""
from pathlib import Path
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, choices=(214,232), default=214)
parser.add_argument("--expected", type=int, default=760)
args = parser.parse_args()
stage = f"round{args.round}"
env = dict(os.environ)
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    env[key] = "4"
env["CUDA_VISIBLE_DEVICES"] = ""
OUT = ROOT / "remote-work" / stage
OUT.mkdir(exist_ok=True, parents=True)
tasks = [
    ("validation_prediction", ["DASE7506-MP2-main/code/predict.py", "--method",
        f"experiments/{stage}/frozen/method/entrypoint.py", "--checkpoint",
        f"experiments/{stage}/bundle", "--split", "validation", "--output",
        f"remote-work/{stage}/validation_predictions.json"]),
    ("validation_scoring", ["DASE7506-MP2-main/code/score_predictions.py", "--predictions",
        f"remote-work/{stage}/validation_predictions.json", "--split", "validation", "--output",
        f"remote-work/{stage}/validation_score.json"]),
    ("proxy_prediction", ["deployment/benchmark_worker.py", "predict", "--round", str(args.round)]),
    ("proxy_scoring", ["deployment/benchmark_worker.py", "score", "--round", str(args.round)]),
]
result = dict(machine=platform.platform(), python=sys.version,
              threads=4, affinity=sorted(os.sched_getaffinity(0)),
              packages={m: importlib.metadata.version(m) for m in
                        ("numpy", "scipy", "scikit-learn", "lightgbm", "threadpoolctl")},
              official_test_repeated=False, reference_machine_certified=False)
for name, command in tasks:
    tick = time.perf_counter()
    p = subprocess.run([sys.executable, *command], cwd=ROOT, env=env,
                        text=True, capture_output=True)
    result[name + "_seconds"] = time.perf_counter() - tick
    (OUT / (name + ".stdout")).write_text(p.stdout)
    (OUT / (name + ".stderr")).write_text(p.stderr)
    print(name, result[name + "_seconds"], p.returncode, p.stdout.strip(), flush=True)
    if p.returncode:
        print(p.stderr, flush=True)
        raise SystemExit(p.returncode)
result["validation"] = json.loads((OUT / "validation_score.json").read_text())
assert result["validation"]["correct"] == args.expected
result["proxy_total_seconds"] = result["proxy_prediction_seconds"] + result["proxy_scoring_seconds"]
(OUT / "deployment_validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
