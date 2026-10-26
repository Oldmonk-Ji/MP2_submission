"""Fit-fold audit of the five round209 extraction fixes versus frozen round201.

Measures, per flag configuration, reference-program coverage and the signal
quality of newly gained roots (do they land on numerically correct candidates?).
Fit folds only (folds!=2); TEST stays sealed.
"""
from round22_common import ROOT, load, read, dump, sha
from round201_trace import extract as base_extract
from round209_trace import extract, FLAGS
from round209_trace import checks
from round109_alignment import mathematical_key
from round44_programs import canonical, tree
from round51_programs import value
from round29_programs import quantities
from features import parse_answer
from fractions import Fraction as F
from collections import defaultdict, Counter
import numpy as np, time

OUT = ROOT / 'experiments/round209'
BASE_FLAGS = dict(floor=False, taint=False, symbolic=False, divfall=False, ordinal=False)
CONFIGS = [
    ('base', dict(BASE_FLAGS)),
    ('floor', dict(FLAGS, floor=True, taint=False, symbolic=False, divfall=False)),
    ('taint', dict(FLAGS, floor=False, taint=True, symbolic=False, divfall=False)),
    ('symbolic', dict(FLAGS, floor=False, taint=False, symbolic=True, divfall=False)),
    ('divfall', dict(FLAGS, floor=False, taint=False, symbolic=False, divfall=True)),
    ('all', dict(FLAGS)),
    ('all_ordinal', dict(FLAGS, ordinal=True)),
]


def key(p):
    try:
        return mathematical_key(tree(p))
    except (ValueError, ZeroDivisionError):
        return None


def run(tick):
    dump(OUT / 'plan.json', dict(
        method='Compare frozen round201 strict trace with round209 five-fix trace on TRAIN fit folds. '
               'Measures reference coverage and signal quality of newly gained roots. Design evidence only.',
        source_hashes={str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'research').glob('round209*.py')},
        test_opened=False))
    b, a, y, ids = load('train')
    refs = defaultdict(list)
    ref_vals = {}
    for r in read(ROOT / 'experiments/round115/fit_sources.json'):
        refs[r['train_index']].append(r['program'])
        assert r['fold'] != 2
    with np.load(ROOT / 'experiments/round77/nested/components.npz') as z:
        folds = z['folds']
        assert np.array_equal(ids, z['ids'])
    fit = np.flatnonzero(folds != 2)
    # candidate-level correctness against reference value
    ref_val_by_idx = {}
    for idx, ps in refs.items():
        qs = quantities(b[list(ids).index(idx)]['question'])
        vals = set()
        for p in ps:
            try:
                vals.add(F(value(p, qs)))
            except (ValueError, ZeroDivisionError, OverflowError, IndexError):
                pass
        if len(vals) == 1:
            ref_val_by_idx[idx] = next(iter(vals))
    stats = {}
    for name, flags in CONFIGS:
        st = Counter()
        st['candidates'] = 0
        events = Counter()
        examples = []
        t0 = time.perf_counter()
        for i in fit:
            idx = int(ids[i])
            row = b[i]
            gold = {key(p) for p in refs.get(idx, [])} - {None}
            gold_val = ref_val_by_idx.get(idx)
            for j, c in enumerate(row['candidates']):
                old, _ = base_extract(row['question'], c['text'])
                new, r = extract(row['question'], c['text'], flags)
                events.update(r.get('events', []))
                ok = {key(p) for p in old} - {None}
                nk = {key(p) for p in new} - {None}
                st['candidates'] += 1
                st['old_roots'] += len(old)
                st['new_roots'] += len(new)
                if ok == nk:
                    st['identical'] += 1
                else:
                    st['changed'] += 1
                    if len(new) > len(old):
                        st['gained_roots_candidates'] += 1
                    if len(new) < len(old):
                        st['lost_roots_candidates'] += 1
                if gold:
                    st['reference_parents'] += 1
                    if bool(ok & gold):
                        st['old_covered'] += 1
                    if bool(nk & gold):
                        st['new_covered'] += 1
                        if not bool(ok & gold):
                            st['coverage_gain'] += 1
                    if bool(ok & gold) and not bool(nk & gold):
                        st['coverage_loss'] += 1
                if len(new) > len(old) and gold_val is not None:
                    ans = parse_answer(c['text'])
                    st['gained_roots_total'] += 1
                    if ans is not None and F(ans) == gold_val:
                        st['gained_roots_on_correct'] += 1
                    else:
                        st['gained_roots_on_wrong'] += 1
                        if len(examples) < 12:
                            examples.append(dict(train_index=idx, candidate=j,
                                                 answer=ans, gold=str(gold_val),
                                                 question=row['question'][:200]))
            if st['candidates'] % 2000 == 0:
                print(name, st['candidates'], '%.0fs' % (time.perf_counter() - t0), flush=True)
        stats[name] = dict(stats=st, events=events, seconds=time.perf_counter() - t0)
        dump(OUT / ('audit_%s.json' % name), dict(stats=st, events=events,
                                                  gained_root_examples=examples, test_opened=False))
        print(name, 'covered', st['new_covered'], '/', st['reference_parents'],
              'gain', st['coverage_gain'], 'loss', st['coverage_loss'],
              'roots', st['new_roots'], '(old %d)' % st['old_roots'],
              'gained-on-correct', st['gained_roots_on_correct'], '/', st['gained_roots_total'], flush=True)
    dump(OUT / 'fit_audit.json', dict(per_config={n: dict(stats=s['stats'], events=s['events']) for n, s in stats.items()},
                                      test_opened=False))
    return stats


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True)
    tick = time.perf_counter()
    assert read(ROOT / 'experiments/round199/training_complete.json')['test_opened'] == False
    assert checks()
    run(tick)
