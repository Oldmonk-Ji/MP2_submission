"""Train-only probe of question-target versus candidate-conclusion alignment."""
import re
import time
from collections import Counter

import lightgbm as lgb
import numpy as np
from threadpoolctl import threadpool_limits

from round22_common import ROOT, dump, load
from round4_gates import apply_gate
from common import parse_answer

OUT = ROOT / "experiments/round224"
PARAMS = dict(objective="lambdarank", lambdarank_truncation_level=1,
              metric="None", num_leaves=7, min_data_in_leaf=100,
              learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
              num_threads=4, deterministic=True, force_col_wise=True)
STOP = set("how many much what which who does do did is are was were will would can "
           "the a an of for to in on at by with from as and or its it their his her "
           "they them there have has had be been this that these those each every all "
           "some number amount answer final find determine calculate get gets got".split())
PAIRS = (("profit", "revenue income sales"),
         ("left remain remaining", "total all"),
         ("cost expense spent", "revenue income earned"),
         ("before original initially", "after final now"),
         ("hour hours", "minute minutes"),
         ("each per", "total all"))


def content(s):
    result = []
    for w in re.findall(r"[a-z]+", s.lower()):
        if len(w) <= 2 or w in STOP:
            continue
        if len(w) > 5 and w.endswith("ing"):
            w = w[:-3]
        elif len(w) > 4 and w.endswith("ed"):
            w = w[:-2]
        elif len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        result.append(w)
    return result


def question_target(q):
    parts = re.split(r"[?!.]", q)
    last = next((part for part in reversed(parts) if part.strip()), q)
    return set(content(last)), set(content(q))


def conclusion(text):
    matches = list(re.finditer(r"(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?", text))
    anchor = matches[-1].start() if matches else len(text)
    near = text[max(0, anchor - 180):min(len(text), anchor + 70)]
    tail = text[max(0, anchor - 360):min(len(text), anchor + 70)]
    return set(content(near)), set(content(tail)), anchor / max(len(text), 1)


def row_features(row):
    target, qwords = question_target(row["question"])
    info = [conclusion(c["text"]) for c in row["candidates"]]
    near = [v[0] for v in info]
    counts = Counter(w for s in near for w in s)
    parsed = [parse_answer(c["text"]) for c in row["candidates"]]
    values = []
    for j in range(4):
        n, t = info[j][:2]
        unique = n - set().union(*(near[k] for k in range(4) if k != j))
        raw = [len(target & n) / max(1, len(target)),
               len(target & t) / max(1, len(target)),
               len(qwords & n) / max(1, len(qwords)),
               len(target & n) / max(1, len(n)),
               len(target & unique) / max(1, len(target)),
               len(unique) / max(1, len(n)),
               sum(1 / counts[w] for w in target & n) / max(1, len(target)),
               info[j][2]]
        for source, opposite in PAIRS:
            a, b = set(content(source)), set(content(opposite))
            raw.extend((float(bool(target & a) and bool(n & a)),
                        float(bool(target & a) and bool(n & b) and not bool(n & a))))
        raw.append(float(parsed[j] is not None))
        values.append(raw)
    a = np.asarray(values, np.float32)
    return np.concatenate([a, a - a.mean(0, keepdims=True)], 1)


def features(bank):
    return np.asarray([row_features(row) for row in bank], np.float32)


def strict_choice(bank, s1, strict):
    base = apply_gate(bank, s1, "parse_and_repair", 4)
    rank = np.argsort(-s1, axis=1)[:, :2]
    best = strict.argmax(1)
    i = np.arange(len(bank))
    flip = (best != base) & (strict[i, best] - strict[i, base] > 3)
    flip &= (base == s1.argmax(1)) & (rank == best[:, None]).any(1)
    result = base.copy()
    result[flip] = best[flip]
    return result


def run():
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    keys = ("nx", "nbs", "x", "bs", "vx", "vbs", "folds",
            "strict_nested", "strict_val", "held_s1", "val_s1")
    with np.load(ROOT / "experiments/round216/cache.npz") as z:
        c = {k: z[k] for k in keys}
    bank, _, y, _ = load("train")
    assert len(bank) == len(c["folds"]) == len(y) == 4743
    fit = c["folds"] != 2
    held = ~fit
    bheld = [bank[i] for i in np.flatnonzero(held)]
    f = features(bank)
    x = np.concatenate([c["nx"], f], 2)
    ds = lgb.Dataset(x[fit].reshape(-1, x.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=c["nbs"][fit].ravel())
    model = lgb.train(PARAMS, ds, num_boost_round=200)
    hs = c["nbs"][held] + model.predict(x[held].reshape(-1, x.shape[-1])).reshape(-1, 4)
    old = strict_choice(bheld, c["held_s1"], c["strict_nested"][held])
    new = strict_choice(bheld, hs, c["strict_nested"][held])
    yh = y[held]
    old_ok = yh[np.arange(len(yh)), old]
    new_ok = yh[np.arange(len(yh)), new]
    rec = dict(hypothesis="answer-window target and conclusion disagreement",
               training_rows=int(fit.sum()), held_rows=int(held.sum()),
               old_held=int(old_ok.sum()), new_held=int(new_ok.sum()),
               changed=int((old != new).sum()),
               fixes=int(((old != new) & (new_ok > old_ok)).sum()),
               harms=int(((old != new) & (new_ok < old_ok)).sum()),
               test_access=False)
    print("HELD", rec, flush=True)
    if rec["new_held"] >= 1477:
        val, _, vy, _ = load("validation")
        vf = features(val)
        xf = np.concatenate([c["x"], f], 2)
        dsf = lgb.Dataset(xf.reshape(-1, xf.shape[-1]), label=y.ravel(),
                          group=np.full(len(bank), 4), init_score=c["bs"].ravel())
        final_model = lgb.train(PARAMS, dsf, num_boost_round=200)
        xv = np.concatenate([c["vx"], vf], 2)
        vs = c["vbs"] + final_model.predict(xv.reshape(-1, xv.shape[-1])).reshape(-1, 4)
        oldv = strict_choice(val, c["val_s1"], c["strict_val"])
        newv = strict_choice(val, vs, c["strict_val"])
        rec.update(old_val=int(vy[np.arange(len(val)), oldv].sum()),
                   new_val=int(vy[np.arange(len(val)), newv].sum()),
                   val_changed=int((oldv != newv).sum()))
        print("VAL", rec, flush=True)
    rec["seconds"] = time.perf_counter() - tick
    dump(OUT / "target_conclusion_result.json", rec)
    return rec


if __name__ == "__main__":
    run()
