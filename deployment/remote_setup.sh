#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/dase7506-mp2
mkdir -p "$ROOT"
if [[ ! -d "$ROOT/repo/.git" ]]; then
  git clone /root/mp2_repository_20261027.bundle "$ROOT/repo"
  git -C "$ROOT/repo" remote set-url origin https://github.com/Oldmonk-Ji/MP2_submission.git
else
  git -C "$ROOT/repo" status --short
fi
PY=/root/miniconda3/envs/py311/bin/python
[[ -x "$PY" ]] || PY=/root/miniconda3/bin/python
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  "$PY" -m venv --system-site-packages "$ROOT/.venv"
fi
"$ROOT/.venv/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/repo/requirements.txt"
cd "$ROOT/repo"
"$ROOT/.venv/bin/python" deployment/remote_inventory.py
"$ROOT/.venv/bin/python" deployment/verify_release.py
