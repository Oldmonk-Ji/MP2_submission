"""Round222: boosting-rounds-only sweep at fixed tree geometry.

Round216's capacity grid changed num_leaves and min_data_in_leaf together with
rounds, so under-fitting at the production geometry was never isolated.  TEST
is never touched.
"""
from round22_common import ROOT, load, dump
from round216_exp import train_eval, load_cache
from threadpoolctl import threadpool_limits
import numpy as np, time

OUT = ROOT / 'experiments/round222'

if __name__ == '__main__':
    threadpool_limits(4)
    tick = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    cache = load_cache()
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    from common import parse_answer
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in v]
    cache['vhard'] = np.array([len({p for p in ps if p is not None}) >= 2 for ps in parsed])
    recs = []
    for rounds in (50, 100, 150, 200, 300, 400, 600, 800):
        rec, _, _, _ = train_eval({}, cache, b, v, y, vy, rounds=rounds,
                                  tag=f'rounds{rounds}')
        recs.append(rec)
        print('R', rec, flush=True)
    for lr, rounds in ((.05, 300), (.05, 500), (.015, 600)):
        rec, _, _, _ = train_eval(dict(learning_rate=lr), cache, b, v, y, vy,
                                  rounds=rounds, tag=f'lr{lr}r{rounds}')
        recs.append(rec)
        print('R', rec, flush=True)
    dump(OUT / 'rounds.json', dict(records=recs, test_opened=False,
                                   seconds=time.perf_counter() - tick))
    print('DONE', time.perf_counter() - tick, flush=True)
