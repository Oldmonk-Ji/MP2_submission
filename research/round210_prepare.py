"""Build round209 strict-trace local features for all TRAIN/VAL inputs."""
from round22_common import ROOT, load, read, dump, sha
from round210_features import programs, build
from round209_trace import checks, FLAGS
from scipy import sparse
import time

OUT = ROOT / 'experiments/round210'
if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    tick = time.perf_counter()
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    assert checks()
    dump(OUT / 'plan.json', dict(
        checks=checks(),
        method='Same local103 65536-dimensional structure features over round209 strict-trace programs '
               '(floor/ceil markup, single-variable linear solves, taint refinement, fraction-quotient fallback). '
               'This is an additional optional trace branch on top of the exact104 readout; nothing frozen yet.',
        flags=FLAGS,
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'research').glob('round21*.py')},
        test_opened=False))
    for split in ('train', 'validation'):
        b, a, y, ids = load(split)
        pp = []
        for i, row in enumerate(b):
            pp.append(programs(row))
            if i % 500 == 0:
                print('PROGRAMS', split, i, flush=True)
        x = build(b, pp, lambda i: print('FEATURES', split, i, flush=True))
        sparse.save_npz(OUT / f'{split}_features.npz', x)
        dump(OUT / f'{split}_metadata.json', dict(ids=ids.tolist(), rows=x.shape[0], columns=x.shape[1],
                                                  nonzeros=x.nnz, test_opened=False))
        print('DONE', split, x.shape, x.nnz, flush=True)
    dump(OUT / 'preparation_complete.json', dict(seconds=time.perf_counter() - tick, test_opened=False))
