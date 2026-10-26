"""Round216: dump the hard-subset errors with candidate evidence."""
from round22_common import ROOT, load, read
from common import parse_answer
import numpy as np, re

OUT = ROOT / 'experiments/round216'


def eq_lines(text, limit=6):
    lines = []
    for ln in text.splitlines():
        s = ln.strip()
        if not s:
            continue
        if re.search(r'=|\+|\-|\*|/|\\times|\\frac|\\div', s) and any(ch.isdigit() for ch in s):
            lines.append(s)
        if len(lines) >= limit:
            break
    return ' | '.join(lines)


if __name__ == '__main__':
    v, va, vy, _ = load('validation')
    n = len(v)
    pred = read(ROOT / 'experiments/round214/validation_strict_gap_t3_predictions.json')
    sel = np.asarray(pred['selected'], int)
    with np.load(ROOT / 'experiments/round211/verified_masks.npz') as z:
        mv = z['val'].astype(bool)
    with np.load(ROOT / 'experiments/round210/scores.npz') as z:
        strict = z['validation']
    parsed = [[parse_answer(c['text']) for c in row['candidates']] for row in v]
    ndist = np.array([len({p for p in ps if p is not None}) for ps in parsed])
    hard = ndist >= 2
    ok_sel = vy[np.arange(n), sel] == 1
    err = np.flatnonzero(hard & ~ok_sel)
    out = []
    for i in err:
        correct = [int(j) for j in np.flatnonzero(vy[i] == 1)]
        out.append(dict(
            vi=int(i), question=v[i]['question'],
            parsed=parsed[i], chosen=int(sel[i]), correct=correct,
            chosen_ans=parsed[i][sel[i]],
            strict=[round(float(x), 2) for x in strict[i]],
            verified=[bool(x) for x in mv[i]],
            cands=[c['text'][-220:].replace('\n', ' ') for c in v[i]['candidates']],
            eqs=[eq_lines(c['text']) for c in v[i]['candidates']],
        ))
    (OUT / 'error_texts.txt').write_text('\n\n'.join(
        f"[{r['vi']}] Q: {r['question']}\n"
        f"  parsed={r['parsed']} chosen={r['chosen']} correct={r['correct']} "
        f"strict={r['strict']} verified={r['verified']}\n" +
        '\n'.join(f"  c{j}: tail: {r['cands'][j]}\n       eqs: {r['eqs'][j]}" for j in range(4))
        for r in out), encoding='utf-8')
    print('written', len(out), 'error rows', flush=True)
