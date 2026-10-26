"""Round216: cache the fusion input tensors so model/feature experiments are fast.

Builds exactly the round210/213/214 feature surface (exact104 + straight-line
augments), the round209 strict scores, the round211 verification masks mapped
to full train order, 3-fold oof stage-1 scores, held/val stage-1 scores, and
answer-cluster structure features.  Nothing is trained here beyond the two
canonical stage-1 models; TEST is never touched.
"""
from round22_common import ROOT, load, read, dump, sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design, project
from round68_selector import components
from round3_features import grounding
from common import parse_answer
from fractions import Fraction as F
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round216'
S1_PARAMS = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
                 min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
                 num_threads=4, deterministic=True, force_col_wise=True)


def cluster_block(bank):
    n = len(bank)
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    feats = np.zeros((n, 4, 8), np.float32)
    packed = {}
    for i, ps in enumerate(parsed):
        vals = [p for p in ps if p is not None]
        groups = {}
        for j, p in enumerate(ps):
            if p is not None:
                groups.setdefault(p, []).append(j)
        ndist = len(groups)
        sizes = [len(v) for v in groups.values()]
        top = max(sizes) if sizes else 0
        packed[i] = (ps, groups)
        for j, p in enumerate(ps):
            if p is None:
                feats[i, j] = [0, 0, 0, 0, 0, 0, 0, 1]
                continue
            size = len(groups[p])
            num = F(p)
            feats[i, j] = [
                size / 4.0,
                float(size == 1),
                float(size == top and top > 1),
                float(ndist),
                float(num.denominator == 1),
                float(num < 0),
                float(np.log1p(abs(float(num)))),
                float(abs(float(num)) > 1e12),
            ]
    return feats, packed


def chain_block(bank, packed):
    """Number-token overlap between each candidate and the row's other chains."""
    import re
    n = len(bank)
    feats = np.zeros((n, 4, 4), np.float32)
    for i, row in enumerate(bank):
        nums = []
        for j, c in enumerate(row['candidates']):
            toks = re.findall(r'-?\d[\d,]*(?:\.\d+)?', c['text'])
            nums.append({t.replace(',', '') for t in toks})
        ps, groups = packed[i]
        for j in range(4):
            if ps[j] is None:
                continue
            same = [k for k in range(4) if k != j and ps[k] is not None and ps[k] == ps[j]]
            other = [k for k in range(4) if k != j]
            union_same = set().union(*[nums[k] for k in same]) if same else set()
            union_other = set().union(*[nums[k] for k in other]) if other else set()
            inter = len(nums[j] & union_same)
            feats[i, j] = [
                len(nums[j] & union_same) / max(len(nums[j]), 1),
                len(nums[j] & union_other) / max(len(nums[j]), 1),
                len(union_same & union_other) / max(len(union_same | union_other), 1),
                float(len(same)),
            ]
    return feats


if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    dump(OUT / 'cache_plan.json', dict(
        method='Cache exact104 fusion inputs, strict scores, verification masks and answer-cluster '
               'features for fast round216 experiments. No TEST access.',
        source_sha256=sha(__file__), test_opened=False))
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    weights = [m['weight'] for m in read(ROOT / 'experiments/round25/selected_bundle/config.json')['members']]
    cache = {}
    for split in ('train', 'validation'):
        with np.load(ROOT / f'experiments/round78/{split}_features.npz') as z:
            cache[split] = (z['root'], z['extra'])
    with np.load(ROOT / 'experiments/round68/oof.npz') as z:
        assert np.array_equal(z['ids'], ids)
        ss = z['components']
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        assert np.array_equal(z['ids'], ids)
        nested = z['components']; folds = z['folds']
    with np.load(ROOT / 'experiments/round79/full_space.npz') as w:
        context = project(a, w); vc = project(va, w)
    with np.load(ROOT / 'experiments/round79/diagnostic_space.npz') as w:
        nc = project(a, w)
    g = grounding(b)
    vs, vg = components(v, va, ROOT / 'experiments/round25/selected_bundle')
    x, bs = design(b, ss, weights, g, *cache['train'], context, 'context_influence')
    nx, nbs = design(b, nested, weights, g, *cache['train'], nc, 'context_influence')
    vx, vbs = design(v, vs, weights, vg, *cache['validation'], vc, 'context_influence')
    with np.load(ROOT / 'experiments/round80/unsigned_scores.npz') as z:
        x = augment(x, z['oof']); nx = augment(nx, z['nested']); vx = augment(vx, z['validation'])
    with np.load(ROOT / 'experiments/round83/relation_scores.npz') as z:
        x = extend(x, z['oof']); nx = extend(nx, z['nested']); vx = extend(vx, z['validation'])
    with np.load(ROOT / 'experiments/round103/local_scores.npz') as z:
        x = augment(x, z['oof']); nx = augment(nx, z['nested']); vx = augment(vx, z['validation'])
    with np.load(ROOT / 'experiments/round210/scores.npz') as z:
        strict_oof, strict_nested, strict_val = z['oof'], z['nested'], z['validation']
    with np.load(ROOT / 'experiments/round211/verified_masks.npz') as z:
        mfit, mheld, mval = z['fit'].astype(bool), z['held'].astype(bool), z['val'].astype(bool)
    fit = folds != 2
    held = ~fit
    verified = np.zeros((len(b), 4), bool)
    verified[fit] = mfit
    verified[held] = mheld
    print('FEATURES done', time.perf_counter() - tick, flush=True)
    with np.load(ROOT / 'experiments/round213/oof_s1.npz') as z:
        assert np.array_equal(z['ids'], ids)
        oof_s1 = z['scores']
    ds = lgb.Dataset(nx[fit].reshape(-1, nx.shape[-1]), label=y[fit].ravel(),
                     group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
    hm = lgb.train(S1_PARAMS, ds, num_boost_round=200)
    held_s1 = nbs[held] + hm.predict(nx[held].reshape(-1, nx.shape[-1])).reshape(-1, 4)
    m104 = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    val_s1 = vbs + m104.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    print('MODELS done', time.perf_counter() - tick, flush=True)
    tfeat, tpacked = cluster_block(b)
    vfeat, vpacked = cluster_block(v)
    tchain = chain_block(b, tpacked)
    vchain = chain_block(v, vpacked)
    np.savez(OUT / 'cache.npz',
             x=x, bs=bs, nx=nx, nbs=nbs, vx=vx, vbs=vbs,
             y=y, vy=vy, folds=folds.astype(np.int8), ids=ids,
             strict_oof=strict_oof, strict_nested=strict_nested, strict_val=strict_val,
             verified=verified, verified_val=mval,
             oof_s1=oof_s1, held_s1=held_s1, val_s1=val_s1,
             tfeat=tfeat, vfeat=vfeat, tchain=tchain, vchain=vchain)
    print('SAVED', (OUT / 'cache.npz').stat().st_size, time.perf_counter() - tick, flush=True)
