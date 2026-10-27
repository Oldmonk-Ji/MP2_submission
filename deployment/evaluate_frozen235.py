"""One authorized official test evaluation; never inspect per-row test outcomes."""
from pathlib import Path
import hashlib
import json
import os
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'experiments/round235/official_test_20261027'
env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    env[key] = '4'

def verify():
    return json.loads(subprocess.check_output([sys.executable, 'deployment/verify_release.py', '--round', '235'], cwd=ROOT, env=env))

before = verify()
OUT.mkdir(parents=True, exist_ok=False)
(OUT / 'authorization.json').write_text(json.dumps(dict(user_authorized=True, user_message='授权', scope='One official test evaluation of already frozen round235; no test-based tuning or selection'), indent=2))
result = dict(before=before, threads=4, affinity=sorted(os.sched_getaffinity(0)), machine=platform.platform(), test_used_for_tuning_or_selection=False, reference_machine_certified=False)
commands = [
    ('prediction', ['DASE7506-MP2-main/code/predict.py', '--method', 'experiments/round235/frozen/method/entrypoint.py', '--checkpoint', 'experiments/round235/bundle', '--split', 'test', '--output', str(OUT / 'test_predictions.json')]),
    ('scoring', ['DASE7506-MP2-main/code/score_predictions.py', '--predictions', str(OUT / 'test_predictions.json'), '--split', 'test', '--output', str(OUT / 'test_score.json')]),
]
for name, command in commands:
    tick = time.perf_counter()
    p = subprocess.run([sys.executable, *command], cwd=ROOT, env=env, capture_output=True, text=True)
    result[name + '_seconds'] = time.perf_counter() - tick
    (OUT / (name + '.stdout')).write_text(p.stdout)
    (OUT / (name + '.stderr')).write_text(p.stderr)
    if p.returncode:
        raise RuntimeError(f'{name} failed; inspect local log before any recovery')
result['score'] = json.loads((OUT / 'test_score.json').read_text())
result['total_seconds'] = result['prediction_seconds'] + result['scoring_seconds']
result['after'] = verify()
assert result['before'] == result['after']
result['frozen_hashes_unchanged'] = True
(OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}
(OUT / 'completion_manifest.json').write_text(json.dumps(dict(hashes=hashes), indent=2) + '\n')
print(json.dumps(result, indent=2), flush=True)
