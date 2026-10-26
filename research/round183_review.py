from round22_common import ROOT,load,read,dump
from round101_templates import canonical
from round103_features import matrix
from round107_latent import root_features
from round29_programs import quantities
import numpy as np
if __name__=='__main__':
    b,a,y,ids=load('train');lookup={int(k):i for i,k in enumerate(ids)};out=ROOT/'experiments/round183';packets=read(ROOT/'experiments/round180/train_programs.json');ptr=np.load(out/'train_root_ptr.npy');gold=np.load(out/'fit_gold_roots.npy');rows=[]
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds']
    seen=set()
    for r in read(out/'fit_root_choice_errors.json'):
        idx=r['train_index']
        if idx in seen:continue
        seen.add(idx);i=lookup[idx];j=r['candidate'];roots={}
        for p in packets[i][j]:roots[repr(canonical(p))]=p
        fs,counts=root_features(b[i],packets[i]);off=sum(counts[:j]);x=matrix(fs[off:off+counts[j]])
        with np.load(ROOT/f'experiments/round181/nested{folds[i]}.npz') as z:s=np.asarray(x@z['weight']).ravel()
        lo,hi=ptr[4*i+j:4*i+j+2];rr=list(roots.values());gg=gold[lo:hi];best=int(s.argmax());example=dict(**r,quantities=[str(n['value'])+':'+n['raw'] for n in quantities(b[i]['question'])],chosen=rr[best],reference_equivalent=rr[int(np.flatnonzero(gg)[0])]);rows.append(example)
    dump(out/'fit_choice_review_material.json',rows)
    for r in rows:print(r,flush=True)
