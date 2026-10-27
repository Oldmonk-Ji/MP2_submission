"""Package round232 delta relative to the already deployed round214."""
from pathlib import Path
import tarfile
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--round',type=int,default=232)
args=parser.parse_args()

ROOT=Path(__file__).resolve().parents[1]
paths=[]
stage=ROOT/f"experiments/round{args.round}"
paths.extend(p for p in stage.glob("*.json"))
paths.extend(p for p in (stage/"frozen").rglob("*")
             if p.is_file() and "bundle" not in p.relative_to(stage/"frozen").parts
             and "__pycache__" not in p.parts)
paths.extend(stage/"bundle"/n for n in ("config.json","quadratic.npz"))
if (stage/'bundle/text').exists():
    paths.extend(p for p in (stage/'bundle/text').rglob('*') if p.is_file())
paths.extend(ROOT/"deployment"/n for n in
             ("validate_remote.py","benchmark_worker.py","verify_release.py"))
target=ROOT/f".deploy-private/mp2_candidate{args.round}.tar.gz"
with tarfile.open(target,"w:gz") as tar:
    for path in paths:
        tar.add(path,arcname=path.relative_to(ROOT).as_posix())
print(dict(files=len(paths),bytes=target.stat().st_size))
