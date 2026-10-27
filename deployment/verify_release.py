"""Verify the copied frozen artifacts without accessing any benchmark data."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--round", type=int, choices=(214,232,235), default=214)
parser.add_argument("--git", action="store_true")
args = parser.parse_args()
stage = f"round{args.round}"
manifest = json.loads((ROOT / f"experiments/{stage}/frozen/completion_manifest.json").read_text())
checked = 0
for name, expected in manifest["hashes"].items():
    path = ROOT / name.replace("\\", "/")
    assert path.is_file(), f"Missing frozen artifact: {name}"
    content = (subprocess.check_output(["git", "show", "HEAD:" + name.replace("\\", "/")],
                                      cwd=ROOT) if "--git" in sys.argv else path.read_bytes())
    assert hashlib.sha256(content).hexdigest() == expected, name
    checked += 1
bundle = ROOT / f"experiments/{stage}/bundle"
total = 0
for path in bundle.rglob("*"):
    if not path.is_file():
        continue
    rel = path.relative_to(bundle)
    twin = ROOT / f"experiments/{stage}/frozen/bundle" / rel
    assert path.read_bytes() == twin.read_bytes(), str(rel)
    total += path.stat().st_size
expected = json.loads((ROOT / f"experiments/{stage}/frozen/summary.json").read_text())["asset_bytes"]
assert total == expected and total <= 16 * 2**20
print(json.dumps(dict(frozen_hashes_verified=checked, asset_bytes=total,
                      canonical_matches_frozen=True, benchmark_data_opened=False)))
