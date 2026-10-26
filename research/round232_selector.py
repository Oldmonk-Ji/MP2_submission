"""Frozen round214 pipeline plus a validation-selected quadratic expert."""
from pathlib import Path
import json
import numpy as np
from features import prepare
from round3_features import build as base_build, grounding
from round3_selector import leaf_scores
from round22_selector import dense_forward
from round136_features import build, summarize
from round210_features import build as strict_features
from round82_relations import probabilities
from round80_selector import augment
from round79_selector import project, design as previous
from round4_gates import apply_gate
from round232_quadratic_inference import scores as quadratic_scores


def components(bank, arrays, path):
    """Same base experts as round214, sharing deterministic raw features."""
    path = Path(path)
    cfg = json.loads((path/"config.json").read_text())
    prepared = prepare(bank, arrays)
    ground = grounding(bank)
    cache, outputs = {}, []
    for member in cfg["members"]:
        sub = path/member["directory"]
        mc = json.loads((sub/"config.json").read_text())
        if mc["family"] == "mlp":
            kind = mc["features"]
            if kind not in cache:
                cache[kind] = base_build(bank, arrays, kind, prepared=prepared, ground=ground)
            with np.load(sub/"model.npz") as w:
                score = dense_forward(cache[kind], w)
        else:
            score = leaf_scores(bank, arrays, sub)
        outputs.append(score)
    small = np.concatenate([prepared["base"], prepared["numeric"],
                            prepared["checks2"], ground], 2).astype(np.float32)
    return np.stack(outputs, 2), ground, small


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    import lightgbm as lgb
    p = Path(checkpoint)
    cfg = json.loads((p/"config.json").read_text())
    ss, g, small = components(bank, arrays, p/"baseline")
    root, extra, flow, pairs, ptr, packets, structure = build(bank, cfg["structure_arm"])
    with np.load(p/"space.npz") as w:
        context = project(arrays, w)
    with np.load(p/"flow.npz") as w:
        flow_score = np.asarray(flow @ w["weight"]).reshape(-1, 4)
    x, bs = previous(bank, ss, cfg["weights"], g, root, extra, context, "context_influence")
    x = augment(x, flow_score)
    with np.load(p/"relation.npz") as w:
        relation = summarize(probabilities(pairs, w), ptr, packets)
    x = np.concatenate([x, relation, relation-relation.mean(1, keepdims=True)], 2)
    with np.load(p/"structure.npz") as w:
        structure_score = np.asarray(structure @ w["weight"]).reshape(-1, 4)
    x = augment(x, structure_score)
    model = lgb.Booster(model_file=str(p/"meta.txt"))
    s1 = bs + model.predict(x.reshape(-1,x.shape[-1]), num_threads=4).reshape(-1,4)
    with np.load(p/"quadratic.npz") as w:
        auxiliary = quadratic_scores(np.asarray(arrays["embeddings"], np.float32), small, w)
    s1 = s1 + cfg["quadratic_alpha"]*(auxiliary-auxiliary.mean(1, keepdims=True))
    base = apply_gate(bank, s1, "parse_and_repair", 4)
    with np.load(p/"strict.npz") as w:
        strict = np.asarray(strict_features(bank) @ w["weight"]).reshape(-1,4)
    row = np.arange(len(bank))
    best = strict.argmax(1)
    top2 = np.argsort(-s1, axis=1)[:, :2]
    flip = (best != base) & (strict[row,best]-strict[row,base] > cfg["strict_tau"])
    flip &= (base == s1.argmax(1)) & (top2 == best[:,None]).any(1)
    result = base.copy()
    result[flip] = best[flip]
    return result.astype(int).tolist()
