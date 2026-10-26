"""Round216: diverse second learners and blends against the frozen fusion.

Members:
  P  pointwise binary GBM (candidate-level correctness probability)
  W  pairwise win-probability GBM over within-row pairs (y=1 vs y=0)
  E  ExtraTrees on the candidate features
Each is trained on fit folds / all rows exactly like the fusion. Member scores
are cached; blend weights are chosen on HELD, then reported on VAL.
"""
from round22_common import ROOT, load, read, dump
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
from sklearn.ensemble import ExtraTreesClassifier
from common import parse_answer
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round216'
P_PARAMS = dict(objective='binary', metric='None', num_leaves=7, min_data_in_leaf=100,
                learning_rate=.03, lambda_l2=50, seed=9109, verbosity=-1, num_threads=4,
                deterministic=True, force_col_wise=True)
W_PARAMS = dict(objective='binary', metric='None', num_leaves=7, min_data_in_leaf=200,
                learning_rate=.05, lambda_l2=50, seed=3113, verbosity=-1, num_threads=4,
                deterministic=True, force_col_wise=True, feature_fraction=0.7)


def zrow(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-6)


def pair_frame(x, bs, y, rows):
    feats, lab = [], []
    for i in rows:
        pos = np.flatnonzero(y[i] == 1)
        neg = np.flatnonzero(y[i] == 0)
        for a in pos:
            for b in neg:
                d = x[i, a] - x[i, b]
                feats.append(np.concatenate([d, [bs[i, a] - bs[i, b]]]))
                lab.append(1.0)
                feats.append(np.concatenate([-d, [bs[i, b] - bs[i, a]]]))
                lab.append(0.0)
    return np.asarray(feats, np.float32), np.asarray(lab, np.float32)


def pair_scores(model, x, bs):
    n = x.shape[0]
    out = np.zeros((n, 4))
    cnt = np.zeros((n, 4))
    idx = np.arange(n)
    for a in range(4):
        for b in range(4):
            if a == b:
                continue
            f = np.concatenate([x[idx, a] - x[idx, b], (bs[idx, a] - bs[idx, b])[:, None]], 1)
            out[idx, a] += model.predict(f, num_threads=4)
            cnt[idx, a] += 1
    return out / np.maximum(cnt, 1)


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    cache = dict(np.load(OUT / 'cache.npz'))
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    fit = cache['folds'] != 2
    held = ~fit
    x, nx, vx = cache['x'], cache['nx'], cache['vx']
    bs, nbs, vbs = cache['bs'], cache['nbs'], cache['vbs']
    held_bank = [r for r, k in zip(b, held) if k]
    scores = {}
    sp = OUT / 'member_scores.npz'
    if sp.exists():
        z = np.load(sp)
        for k in ('P', 'W', 'E'):
            scores[k] = (z[k + '_h'], z[k + '_v'])
        print('members loaded', time.perf_counter() - tick, flush=True)
    else:
        ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel())
        pm_h = lgb.train(P_PARAMS, ds, num_boost_round=200)
        ds = lgb.Dataset(x.reshape(-1, x.shape[-1]), label=y.ravel())
        pm_v = lgb.train(P_PARAMS, ds, num_boost_round=200)
        scores['P'] = (pm_h.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4),
                       pm_v.predict(vx.reshape(-1, vx.shape[-1])).reshape(-1, 4))
        print('P done', time.perf_counter() - tick, flush=True)
        fw, lw = pair_frame(nx, nbs, y, np.flatnonzero(fit))
        wm = lgb.train(W_PARAMS, lgb.Dataset(fw, label=lw), num_boost_round=200)
        fw2, lw2 = pair_frame(x, bs, y, np.arange(len(b)))
        wm2 = lgb.train(W_PARAMS, lgb.Dataset(fw2, label=lw2), num_boost_round=200)
        scores['W'] = (pair_scores(wm, nx[held], nbs[held]), pair_scores(wm2, vx, vbs))
        print('W done', time.perf_counter() - tick, flush=True)
        et_h = ExtraTreesClassifier(n_estimators=400, max_features=0.2, min_samples_leaf=5,
                                    n_jobs=4, random_state=55).fit(
            nx[fit].reshape(-1, nx.shape[-1]), y[fit].ravel())
        et_v = ExtraTreesClassifier(n_estimators=400, max_features=0.2, min_samples_leaf=5,
                                    n_jobs=4, random_state=55).fit(
            x.reshape(-1, x.shape[-1]), y.ravel())
        scores['E'] = (et_h.predict_proba(nx[held].reshape(-1, nx.shape[-1]))[:, 1].reshape(-1, 4),
                       et_v.predict_proba(vx.reshape(-1, vx.shape[-1]))[:, 1].reshape(-1, 4))
        print('E done', time.perf_counter() - tick, flush=True)
        np.savez(sp, **{k + s: arr for k, pair in scores.items() for s, arr in zip(('_h', '_v'), pair)})
    base_h = cache['held_s1']
    base_v = cache['val_s1']
    hardv = np.array([len({p for p in (parse_answer(c['text']) for c in row['candidates'])
                          if p is not None}) >= 2 for row in v])
    records = []

    def rec(s_h, s_v, name):
        hc = apply_gate(held_bank, s_h, 'parse_and_repair', 4)
        vc = apply_gate(v, s_v, 'parse_and_repair', 4)
        r = dict(name=name,
                 held=int((y[held][np.arange(int(held.sum())), hc] == 1).sum()),
                 val=int((vy[np.arange(len(v)), vc] == 1).sum()),
                 val_hard=int((vy[hardv, vc[hardv]] == 1).sum()))
        records.append(r)
        print('REC', r, flush=True)
        return r

    rec(base_h, base_v, 'base')
    for k, (sh, sv) in scores.items():
        rec(sh, sv, k)
    best = None
    for w1 in (0, .1, .2, .3, .4, .5, .7, 1.0):
        for w2 in (0, .1, .2, .3, .4, .5):
            for w3 in (0, .1, .2, .3):
                if w1 == w2 == w3 == 0:
                    continue
                sh, sv = zrow(base_h), zrow(base_v)
                for k, w in (('P', w1), ('W', w2), ('E', w3)):
                    if w:
                        sh = sh + w * zrow(scores[k][0])
                        sv = sv + w * zrow(scores[k][1])
                hc = apply_gate(held_bank, sh, 'parse_and_repair', 4)
                h = int((y[held][np.arange(int(held.sum())), hc] == 1).sum())
                key = (h, -(w1 + w2 + w3))
                if best is None or key > best[0]:
                    best = (key, (w1, w2, w3), sh, sv, h)
    key, ws, sh, sv, h = best
    vc = apply_gate(v, sv, 'parse_and_repair', 4)
    best_rec = dict(name='best_on_held', weights=ws, held=h,
                    val=int((vy[np.arange(len(v)), vc] == 1).sum()),
                    val_hard=int((vy[hardv, vc[hardv]] == 1).sum()))
    records.append(best_rec)
    print('BEST', best_rec, flush=True)
    dump(OUT / 'exp_ensemble.json', dict(records=records, test_opened=False,
                                         seconds=time.perf_counter() - tick))
    print('DONE', time.perf_counter() - tick, flush=True)
