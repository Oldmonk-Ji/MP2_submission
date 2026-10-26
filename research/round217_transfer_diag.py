"""Round217: how much train->val template transfer is available?

Masks numbers in questions, builds a skeleton key, and measures exact
skeleton reuse between TRAIN and VALIDATION; also reports a TF-IDF cosine
top-1 for context. Pure diagnostic: no TEST access, no model changes.
"""
from round22_common import ROOT, load, read, dump
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np, re

OUT = ROOT / 'experiments/round217'
OUT.mkdir(exist_ok=True)
NUM = re.compile(r'\d[\d,]*(?:\.\d+)?')


def skeleton(q):
    s = NUM.sub(' <n> ', q.lower())
    s = re.sub(r'[^a-z<> ]+', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()


def numbers(q):
    return [float(x.replace(',', '')) for x in NUM.findall(q)]


if __name__ == '__main__':
    b, a, y, ids = load('train')
    v, va, vy, _ = load('validation')
    from common import parse_answer
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in v]
    hard = np.array([len({p for p in ps if p is not None}) >= 2 for ps in parsed])
    train_sk = {}
    for i, row in enumerate(b):
        train_sk.setdefault(skeleton(row['question']), []).append(i)
    matches = []
    for i, row in enumerate(v):
        k = skeleton(row['question'])
        hits = train_sk.get(k, [])
        if len(hits) == 1:
            matches.append((i, hits[0], len(numbers(row['question'])), len(numbers(b[hits[0]]['question']))))
        elif len(hits) > 1:
            matches.append((i, hits[0], len(numbers(row['question'])), len(numbers(b[hits[0]]['question']))))
    mi = np.array([m[0] for m in matches], int)
    eq_n = np.array([m[2] == m[3] for m in matches])
    print('exact skeleton', len(matches), 'of', len(v), flush=True)
    print('equal number counts', int(eq_n.sum()), flush=True)
    print('matched hard', int(hard[mi].sum()), 'matched free', int((~hard[mi]).sum()), flush=True)
    # tf-idf cosine top1 (masked questions)
    docs_t = [skeleton(r['question']) for r in b]
    docs_v = [skeleton(r['question']) for r in v]
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=3)
    Xt = vec.fit_transform(docs_t)
    Xv = vec.transform(docs_v)
    S = (Xv @ Xt.T).toarray()
    top = S.argmax(1)
    topv = S.max(1)
    print('cosine top1 quantiles', np.quantile(topv, [0, .25, .5, .75, .9, 1]).round(3), flush=True)
    for thr in (0.8, 0.9, 0.95):
        m = topv >= thr
        print('cos>=', thr, 'rows', int(m.sum()), 'hard', int(hard[m].sum()), flush=True)
    dump(OUT / 'transfer_coverage.json', dict(
        exact_skeleton=int(len(matches)), exact_equal_numbers=int(eq_n.sum()),
        exact_hard=int(hard[mi].sum()), exact_free=int((~hard[mi]).sum()),
        val_n=len(v), val_hard=int(hard.sum()),
        cosine_thresholds={str(t): dict(rows=int((topv >= t).sum()), hard=int(hard[topv >= t].sum()))
                           for t in (0.8, 0.9, 0.95)},
        cosine_quantiles=[round(float(x), 4) for x in np.quantile(topv, [0, .25, .5, .75, .9, 1])],
        test_opened=False))
    np.savez(OUT / 'cosine_top.npz', top=top, topv=topv)
    print('DONE', flush=True)
