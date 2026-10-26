#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/dase7506-mp2
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
if [[ ! -x "$ROOT/.train-venv/bin/python" ]]; then
  /root/miniconda3/bin/python -m venv --system-site-packages "$ROOT/.train-venv"
fi
"$ROOT/.train-venv/bin/python" -m pip install --disable-pip-version-check -r "$ROOT/repo/requirements.txt"
"$ROOT/.train-venv/bin/python" "$ROOT/repo/deployment/remote_inventory.py"
