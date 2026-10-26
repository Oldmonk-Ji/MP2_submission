"""Input-only inference for the frozen quadratic expert; NumPy only."""
import numpy as np


def scores(e, g, w):
    q = e.mean(1)
    r = e - q[:, None]
    r = np.clip(((r-w["rmean"]) @ w["rpc"].T)/w["rscale"], -6, 6)
    q = np.clip(((q-w["qmean"]) @ w["qpc"].T)/w["qscale"], -6, 6)
    g = np.clip((g-w["gmean"])/w["gscale"], -6, 6)
    cross = (r[:, :, :, None]*q[:, None, None, :]).reshape(len(e), 4, -1)
    a, b = np.triu_indices(32)
    quad = r[:, :, a]*r[:, :, b]
    x = np.concatenate([r/np.sqrt(128), cross/np.sqrt(4096),
                        quad/np.sqrt(len(a)), g/np.sqrt(g.shape[-1]),
                        (g-g.mean(1, keepdims=True))/np.sqrt(g.shape[-1])], 2)
    return np.ascontiguousarray(x, dtype=np.float32) @ w["weight"]
