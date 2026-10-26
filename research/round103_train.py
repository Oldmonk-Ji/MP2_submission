from round22_common import ROOT,load,read,dump,sha
from round103_features import build,checks
from round80_train import fit
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round103'

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();checks();assert read(ROOT/'experiments/round102/cache_complete.json')['test_opened']==False
    dump(OUT/'plan.json',dict(arms=['local','whole'],C=1,dimension=65536,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round103*.py')},fit_audit=read(ROOT/'experiments/round102/fit_audit.json')['counts'],labels='TRAIN original candidate final correctness; weak program compatibility, not process gold',test_opened=False,checks=checks()))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');cache={}
    for split,rows in [('train',b),('validation',v)]:
        packets=read(ROOT/f'experiments/round102/{split}_programs.json');cache[split]={}
        for arm in ('local','whole'):
            cache[split][arm]=build(rows,packets,arm);sparse.save_npz(OUT/f'{split}_{arm}.npz',cache[split][arm]);print('FEATURES',split,arm,cache[split][arm].shape,cache[split][arm].nnz,flush=True)
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    for arm in ('local','whole'):
        xx=cache['train'][arm];oof=np.zeros((len(b),4));ns=np.zeros_like(oof)
        for held in range(3):
            ix=np.flatnonzero(folds!=held);te=np.flatnonzero(folds==held);w=fit(xx,y,ix,ids,OUT/arm/f'oof{held}.npz');rows=(te[:,None]*4+np.arange(4)).ravel();oof[te]=np.asarray(xx[rows]@w).reshape(-1,4)
        ns[folds==2]=oof[folds==2]
        for held in (0,1):
            ix=np.flatnonzero(folds==1-held);te=np.flatnonzero(folds==held);w=fit(xx,y,ix,ids,OUT/arm/f'nested{held}.npz');rows=(te[:,None]*4+np.arange(4)).ravel();ns[te]=np.asarray(xx[rows]@w).reshape(-1,4)
        w=fit(xx,y,np.arange(len(b)),ids,OUT/arm/'full.npz');vs=np.asarray(cache['validation'][arm]@w).reshape(-1,4);np.savez(OUT/f'{arm}_scores.npz',oof=oof,nested=ns,validation=vs);te=folds==2;pred=oof[te].argmax(1);report=dict(correct=int(y[te][np.arange(te.sum()),pred].sum()),n=int(te.sum()),candidate_parse_gate_not_applied=True);dump(OUT/f'{arm}_held_standalone.json',report);print('STANDALONE',arm,report,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
