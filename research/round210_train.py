"""Fusion readout: exact104 base plus round209 strict mean score and event channels."""
from round22_common import ROOT, load, read, dump, sha
from round83_selector import extend
from round80_selector import augment
from round79_selector import design, project
from round68_selector import components
from round3_features import grounding
from round4_gates import apply_gate
from round210_events import build as events_build
from round209_trace import FLAGS
from threadpoolctl import threadpool_limits
import numpy as np, lightgbm as lgb, time, shutil

OUT = ROOT / 'experiments/round210'
if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    assert (ROOT / 'experiments/round210/fit_complete.json').exists()
    arms = ['control', 'events', 'events_ord', 'strict', 'strict_events', 'strict_events_ord']
    dump(OUT / 'train_plan.json', dict(
        arms=arms,
        method='Exact104 base plus (a) round209 strict mean score trained by round210_fit, '
               '(b) round210 event channels with and without ordinal binding. '
               'Same fixed 200-tree 7-leaf fusion; strict improvement over 759 required before TEST.',
        flags=FLAGS,
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
        nested = z['components']
        folds = z['folds']
    for fold in range(3):
        for stage in ([f'oof{fold}', f'nested{fold}'] if fold != 2 else ['oof2']):
            assert not set(ids[folds == fold]) & set(read(ROOT / f'experiments/round210/{stage}.json')['training_ids'])
    with np.load(ROOT / 'experiments/round79/full_space.npz') as w:
        context = project(a, w)
        vc = project(va, w)
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
    ev = events_build(b)
    vev = events_build(v)
    evo = events_build(b, dict(FLAGS, ordinal=True))
    vevo = events_build(v, dict(FLAGS, ordinal=True))
    with np.load(ROOT / 'experiments/round210/scores.npz') as z:
        strict = z['oof'], z['nested'], z['validation']
    params = dict(objective='lambdarank', lambdarank_truncation_level=1, metric='None', num_leaves=7,
                  min_data_in_leaf=100, learning_rate=.03, lambda_l2=50, seed=68068, verbosity=-1,
                  num_threads=4, deterministic=True, force_col_wise=True)
    records = []
    fit = folds != 2
    held = ~fit
    for arm in arms:
        ax, an, av = x, nx, vx
        if 'events' in arm:
            e, ve = (evo, vevo) if arm.endswith('_ord') else (ev, vev)
            ax = np.concatenate([ax, e], 2)
            an = np.concatenate([an, e], 2)
            av = np.concatenate([av, ve], 2)
        if 'strict' in arm:
            ax = augment(ax, strict[0]); an = augment(an, strict[1]); av = augment(av, strict[2])
        ds = lgb.Dataset(an[fit].reshape(-1, an.shape[-1]), label=y[fit].ravel(),
                         group=np.full(fit.sum(), 4), init_score=nbs[fit].ravel())
        hm = lgb.train(params, ds, num_boost_round=200)
        hc = apply_gate([r for r, k in zip(b, held) if k],
                        nbs[held] + hm.predict(an[held].reshape(-1, an.shape[-1])).reshape(-1, 4),
                        'parse_and_repair', 4)
        dump(OUT / f'{arm}_held.json', dict(correct=int(y[held][np.arange(held.sum()), hc].sum()),
                                            n=int(held.sum()), nested=True))
        ds = lgb.Dataset(ax.reshape(-1, ax.shape[-1]), label=y.ravel(), group=np.full(len(b), 4),
                         init_score=bs.ravel())
        m = lgb.train(params, ds, num_boost_round=200)
        c = apply_gate(v, vbs + m.predict(av.reshape(-1, av.shape[-1])).reshape(-1, 4), 'parse_and_repair', 4)
        bc = apply_gate(v, vbs, 'parse_and_repair', 4)
        ok = vy[np.arange(len(v)), c]
        bo = vy[np.arange(len(v)), bc]
        r = dict(arm=arm, correct=int(ok.sum()), repairs=int(((ok == 1) & (bo == 0)).sum()),
                 harms=int(((ok == 0) & (bo == 1)).sum()), feature_dimension=int(ax.shape[-1]))
        if arm == 'control':
            assert r['correct'] == 759
            assert c.tolist() == read(ROOT / 'experiments/round104/validation_local_predictions.json')['selected']
        bundle = OUT / f'bundle_{arm}'
        bundle.mkdir(exist_ok=True)
        shutil.copytree(ROOT / 'experiments/round25/selected_bundle', bundle / 'baseline')
        m.save_model(str(bundle / 'meta.txt'))
        for source, name in [('round79/full_space.npz', 'space.npz'), ('round80/unsigned/full.npz', 'flow.npz'),
                             ('round82/full.npz', 'relation.npz'), ('round103/local/full.npz', 'structure.npz')]:
            shutil.copy2(ROOT / 'experiments' / source, bundle / name)
        if 'strict' in arm:
            shutil.copy2(ROOT / 'experiments/round210/full.npz', bundle / 'strict.npz')
        dump(bundle / 'config.json', dict(arm='relations', weights=weights, structure_arm='local',
                                          strict_arm='mean210' if 'strict' in arm else 'control',
                                          events_arm='events_ord' if arm.endswith('_ord') else ('events' if 'events' in arm else 'control'),
                                          event_flags=FLAGS))
        pred = read(ROOT / 'experiments/round63/validation_sft_predictions.json')
        pred['selected'] = c.tolist()
        dump(OUT / f'validation_{arm}_predictions.json', pred)
        r['asset_bytes'] = sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())
        assert r['asset_bytes'] <= 16 * 1024 ** 2
        records.append(r)
        print('VAL', r, flush=True)
    dump(OUT / 'validation_records.json', records)
    new = max(records[1:], key=lambda r: (r['correct'], -r['harms'], -r['asset_bytes']))
    dump(OUT / 'final_feature_selection.json', dict(
        best_new=dict(arm=new['arm'], correct=new['correct'], bundle='experiments/round210/bundle_' + new['arm']),
        selected='round210' if new['correct'] > 759 else 'round104',
        test_target_fixed_before_test=True, test_used_for_selection=False))
    dump(OUT / 'training_complete.json', dict(seconds=time.perf_counter() - tick,
                                              training_ids=ids.tolist(), test_opened=False))
