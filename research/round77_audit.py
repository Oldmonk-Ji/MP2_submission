"""TRAIN-fit-only audit of information lost by unordered program representations."""
from round22_common import ROOT,load,read,dump,sha
from collections import Counter
import numpy as np

if __name__=='__main__':
    b,a,y,ids=load('train');ts=read(ROOT/'experiments/round76/train_tokens.json')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    rows=[];counts=Counter()
    for i in np.flatnonzero(f!=2):
        prog=[s[s.index('sep')+1:-1] for s in ts[i]];pairs=[]
        counts['parents']+=1;counts['mixed_parents']+=int(y[i].min()!=y[i].max())
        counts['unknown_candidates']+=sum(p==['unknown_program'] for p in prog)
        counts['correct_candidates']+=int(y[i].sum())
        counts['unknown_correct_candidates']+=sum(p==['unknown_program'] and y[i,j]>0 for j,p in enumerate(prog))
        for j in range(4):
            for k in range(j):
                if y[i,j]==y[i,k]:continue
                if Counter(prog[j])==Counter(prog[k]):
                    kind='exact_equal' if prog[j]==prog[k] else 'order_only'
                    counts[kind+'_pairs']+=1;pairs.append(dict(left=j,right=k,kind=kind,left_program=prog[j],right_program=prog[k]))
        if pairs:rows.append(dict(train_index=int(ids[i]),question=b[i]['question'],contrasts=pairs))
    dump(ROOT/'experiments/round77/fit_representation_audit.json',dict(counts={k:int(v) for k,v in counts.items()},parents_with_aliases=len(rows),examples=rows,source_hash=sha(__file__),train_fit_only=True,test_opened=False))
    print(dict(counts),flush=True)
