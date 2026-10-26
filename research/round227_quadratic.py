"""Convex conditional quadratic ranking of supplied representation residuals."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from features import prepare
from round3_features import grounding

SEED = 227227


def inputs(bank, arrays):
    p = prepare(bank, arrays)
    g = np.concatenate([p["base"], p["numeric"], p["checks2"], grounding(bank)], 2)
    return np.asarray(arrays["embeddings"], np.float32), g.astype(np.float32)


def fit_space(e, g, rows):
    q = e.mean(1)
    r = e - q[:, None]
    rp = PCA(n_components=128, svd_solver="randomized", random_state=SEED).fit(r[rows].reshape(-1, 1024))
    qp = PCA(n_components=32, svd_solver="randomized", random_state=SEED).fit(q[rows])
    return dict(rmean=rp.mean_, rpc=rp.components_,
                rscale=np.sqrt(rp.explained_variance_).clip(1e-4),
                qmean=qp.mean_, qpc=qp.components_,
                qscale=np.sqrt(qp.explained_variance_).clip(1e-3),
                gmean=g[rows].mean((0, 1)), gscale=g[rows].std((0, 1)).clip(.05))


def transform(e, g, w):
    q = e.mean(1)
    r = e - q[:, None]
    r = np.clip(((r-w["rmean"]) @ w["rpc"].T)/w["rscale"], -6, 6)
    q = np.clip(((q-w["qmean"]) @ w["qpc"].T)/w["qscale"], -6, 6)
    g = np.clip((g-w["gmean"])/w["gscale"], -6, 6)
    cross = (r[:, :, :, None] * q[:, None, None, :]).reshape(len(e), 4, -1)
    # Full low-dimensional within-candidate quadratic structure, not only
    # coordinatewise embedding products.
    a, b = np.triu_indices(32)
    quad = r[:, :, a] * r[:, :, b]
    x = np.concatenate([r/np.sqrt(128), cross/np.sqrt(4096),
                        quad/np.sqrt(len(a)), g/np.sqrt(g.shape[-1]),
                        (g-g.mean(1, keepdims=True))/np.sqrt(g.shape[-1])], 2)
    return np.ascontiguousarray(x, dtype=np.float32)


def fit(e, g, y, rows):
    w = fit_space(e, g, rows)
    x = transform(e, g, w)
    diffs, weights = [], []
    for i in rows:
        positive, negative = np.flatnonzero(y[i] == 1), np.flatnonzero(y[i] == 0)
        count = len(positive)*len(negative)
        for a in positive:
            for b in negative:
                diffs.append(x[i, a]-x[i, b])
                weights.append(1/count)
    delta = np.asarray(diffs, np.float32)
    train = np.concatenate([delta, -delta])
    target = np.r_[np.ones(len(delta)), np.zeros(len(delta))]
    model = LogisticRegression(C=1, fit_intercept=False, solver="liblinear",
                               max_iter=500, tol=1e-5, random_state=SEED)
    model.fit(train, target, sample_weight=np.tile(np.asarray(weights)/2, 2))
    w["weight"] = model.coef_[0].astype(np.float32)
    return w, x @ w["weight"]


def scores(e, g, w):
    return transform(e, g, w) @ w["weight"]
