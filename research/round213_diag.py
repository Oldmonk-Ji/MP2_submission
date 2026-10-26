"""Per-row analysis of the 24 VAL errors with correct candidate in fusion top-2."""
from round22_common import ROOT, load, read, dump, sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design, project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time

OUT = ROOT / 'experiments/round213'
if __name__ == '__main__':
    threadpool_limits(4)
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    weights = [m['weight'] for m in read(ROOT / 'experiments/round25/selected_bundle/config.json')['members']]
    cache = {}
    for split in ('train', 'validation'):
        with np.load(ROOT / f'experiments/round78/{split}_features.npz') as z:
            cache[split] = (z['root'], z['extra'])
    with np.load(ROOT / 'experiments/round68/oof.npz') as z:
        ss = z['components']
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        nested = z['components']; folds = z['folds']
    with np.load(ROOT / 'experiments/round79/full_space.npz') as w:
        context = project(a, w); vc = project(va, w)
    with np.load(ROOT / 'experiments/round79/diagnostic_space.npz') as w:
        nc = project(a, w)
    g = grounding(b)
    vs, vg = components(v, va, ROOT / 'experiments/round25/selected_bundle')
    x, bs = design(b, ss, weights, g, *cache['train'], context, 'context_influence')
    vx, vbs = design(v, vs, weights, vg, *cache['validation'], vc, 'context_influence')
    with np.load(ROOT / 'experiments/round80/unsigned_scores.npz') as z:
        x = augment(x, z['oof']); vx = augment(vx, z['validation'])
    with np.load(ROOT / 'experiments/round83/relation_scores.npz') as z:
        x = extend(x, z['oof']); vx = extend(vx, z['validation'])
    with np.load(ROOT / 'experiments/round103/local_scores.npz') as z:
        x = augment(x, z['oof']); vx = augment(vx, z['validation'])
    m104 = lgb.Booster(model_file=str(ROOT / 'experiments/round210/bundle_control/meta.txt'))
    val_s1 = vbs + m104.predict(vx.reshape(-1, vx.shape[-1]), num_threads=4).reshape(-1, 4)
    gated = apply_gate(v, val_s1, 'parse_and_repair', 4)
    assert (vy[np.arange(len(v)), gated] == 1).sum() == 759
    with np.load(ROOT / 'experiments/round211/verified_masks.npz') as z:
        mv = z['val']
    with np.load(ROOT / 'experiments/round210/scores.npz') as z:
        strict = z['validation']
    ranks = 4 - 1 - val_s1.argsort(1).argsort(1)
    correct_any = vy == 1
    rows = []
    for i in range(len(v)):
        if vy[i, gated[i]] == 1:
            continue
        top2 = ranks[i] <= 1
        in_top2 = int((top2 & correct_any[i]).any())
        rows.append(dict(vi=i,
                         q=v[i]['question'][:80],
                         chosen=int(gated[i]), gate_override=bool(gated[i] != val_s1[i].argmax()),
                         correct=[int(j) for j in np.flatnonzero(correct_any[i])],
                         scores=val_s1[i].round(3).tolist(),
                         base=vbs[i].round(3).tolist(),
                         verified=[bool(m) for m in mv[i]],
                         strict=strict[i].round(3).tolist(),
                         in_top2=in_top2))
    dump(OUT / 'error_rows.json', rows)
    print('errors', len(rows), 'top2', sum(r['in_top2'] for r in rows), flush=True)
    for r in rows:
        print(r['vi'], 'top2' if r['in_top2'] else '    ', 'gate' if r['gate_override'] else '    ',
              'chosen', r['chosen'], 'correct', r['correct'],
              'score', r['scores'][r['chosen']], '->', r['scores'][r['correct'][0]],
              'vrf', r['verified'], 'strict', r['strict'][r['correct'][0]], flush=True)
