from round22_common import ROOT,load,read,dump,sha
from round80_train import fit
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round204'
if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();assert (ROOT/'experiments/round203/preparation_complete.json').exists();dump(OUT/'plan.json',dict(method='Same103 local candidate-contrast logisticC1 objective and65536-dimensional features; only supplied program hypotheses now come from frozen strict179 traces. Missing traces remain unsupported; no correctness hard gate.',source_sha256=sha(__file__),test_opened=False))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');x=sparse.load_npz(ROOT/'experiments/round203/train_features.npz');vx=sparse.load_npz(ROOT/'experiments/round203/validation_features.npz')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    oof=np.zeros((len(b),4));nested=np.zeros_like(oof)
    for stage in ('oof0','oof1','oof2','nested0','nested1','full'):
        held=int(stage[-1]) if stage!='full' else None;ix=np.arange(len(b)) if held is None else np.flatnonzero(folds==1-held) if stage.startswith('nested') else np.flatnonzero(folds!=held);w=fit(x,y,ix,ids,OUT/f'{stage}.npz')
        if held is not None:
            te=np.flatnonzero(folds==held);rr=(te[:,None]*4+np.arange(4)).ravel();(nested if stage.startswith('nested') else oof)[te]=np.asarray(x[rr]@w).reshape(-1,4)
        else:val=np.asarray(vx@w).reshape(-1,4)
        print('COMPLETE',stage,flush=True)
    nested[folds==2]=oof[folds==2];np.savez(OUT/'scores.npz',oof=oof,nested=nested,validation=val);te=folds==2;pred=oof[te].argmax(1);report=dict(correct=int(y[te][np.arange(te.sum()),pred].sum()),n=int(te.sum()),candidate_parse_gate_not_applied=True,test_opened=False);dump(OUT/'held_standalone.json',report);print(report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,training_ids=ids.tolist()))
