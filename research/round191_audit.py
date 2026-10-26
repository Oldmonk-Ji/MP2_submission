from round22_common import ROOT,load,read,dump
from round29_programs import quantities
from collections import Counter
import numpy as np
if __name__=='__main__':
    out=ROOT/'experiments/round191';out.mkdir();b,a,y,ids=load('train')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds']
    rows=[]
    for i in np.flatnonzero(f!=2):
        q=b[i]['question'];qs=quantities(q);counts=Counter(str(n['value']) for n in qs)
        if any(v>1 for v in counts.values()):
            rows.append(dict(train_index=int(ids[i]),question=q,quantities=[dict(value=str(n['value']),context=q[max(0,n['start']-45):n['end']+40]) for n in qs if counts[str(n['value'])]>1],candidate_lines=[line for c in b[i]['candidates'] for line in c['text'].splitlines() if '=' in line][:12]))
    dump(out/'fit_same_value_review.json',rows);print('FIT same-value parents',len(rows))
    for r in rows[:8]:print(r)
