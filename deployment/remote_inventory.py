"""Read-only environment inventory; no credentials or process arguments."""
import importlib.util
import json
import os
import platform
import sys

print(json.dumps({
    "python": sys.executable, "version": sys.version, "platform": platform.platform(),
    "affinity": sorted(os.sched_getaffinity(0)),
    "packages": {m: importlib.util.find_spec(m) is not None for m in
                 ("numpy", "scipy", "sklearn", "lightgbm", "threadpoolctl", "torch")},
}, indent=2))
