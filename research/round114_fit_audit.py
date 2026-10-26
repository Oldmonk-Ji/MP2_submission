"""Inspect only TRAIN-fit cross-fitted mistakes in reference-root choices."""
from round22_common import ROOT,load,read,dump,sha
from round101_templates import canonical
from round29_programs import quantities
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np
OUT=ROOT/'experiments/round114'
if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();b,a,y,ids=load('train');base=ROOT/'experiments/round107'
    x=sparse.load_npz(base/'train_root_features.npz');ptr=np.load(base/'train_root_ptr.npy')
    gold=np.load(ROOT/'experiments/round109/full_gold_roots.npy');active=np.load(ROOT/'experiments/round109/full_active_candidates.npy')
    packets=read(ROOT/'experiments/round102/train_programs.json')
    reviewed={r['train_index'] for r in read(ROOT/'experiments/round51/reviewed_programs.json')}
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    examples={}
    for fold in (0,1):
        path=ROOT/f'experiments/round110/nested{fold}.npz';train=set(read(path.with_suffix('.json'))['training_ids'])
        with np.load(path) as w:s=np.asarray(x@w['weight']).ravel()
        for c in np.flatnonzero(active&np.repeat(f==fold,4)):
            i,j=divmod(int(c),4);idx=int(ids[i]);assert idx not in train
            lo,hi=ptr[c:c+2];ss=s[lo:hi];gg=gold[lo:hi];best=int(ss.argmax())
            if gg[best]:continue
            programs={}
            for p in packets[i][j]:
                try:programs[repr(canonical(p))]=p
                except (ValueError,ZeroDivisionError):continue
            roots=list(programs.values());assert len(roots)==hi-lo
            margin=float(ss[best]-ss[gg].max());r=dict(train_index=idx,fold=fold,candidate=j,question=b[i]['question'],quantities=quantities(b[i]['question']),predicted_program=roots[best],equivalent_reference_programs=[p for p,g in zip(roots,gg) if g],margin=margin,assistant_reviewed_source=idx in reviewed)
            if idx not in examples or margin>examples[idx]['margin']:examples[idx]=r
    rows=sorted(examples.values(),key=lambda r:(-r['margin'],r['train_index']))
    dump(OUT/'fit_root_errors.json',rows);dump(OUT/'audit_manifest.json',dict(fit_folds=[0,1],n=len(rows),ordering='Descending OOF wrong-root margin; one candidate per parent',source_sha256=sha(__file__),test_opened=False))
    for r in rows[:8]:print({k:v for k,v in r.items() if k not in ('quantities',)},flush=True)
