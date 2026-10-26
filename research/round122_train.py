from round22_common import ROOT,load,read,dump,sha
from round122_features import build,checks
from round80_train import fit
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round122'
if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir()
    dump(OUT/'plan.json',dict(checks=checks(),method='Matched round103 local structure ranker: replace conflated person tokens by stable within-question first-mention entity identifiers. All roots, tags, hash dimension, pair loss, C1 and six fit partitions unchanged.',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round122*.py')},test_opened=False))
    b,a,y,ids=load('train');v,va,vy,_=load('validation')
    x=build(b,read(ROOT/'experiments/round102/train_programs.json'));vx=build(v,read(ROOT/'experiments/round102/validation_programs.json'))
    sparse.save_npz(OUT/'train_features.npz',x);sparse.save_npz(OUT/'validation_features.npz',vx)
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    outputs=dict(oof=np.zeros((len(b),4)),nested=np.zeros((len(b),4)))
    stages=[(f'oof{k}',np.flatnonzero(f!=k),np.flatnonzero(f==k),'oof') for k in range(3)]+[(f'nested{k}',np.flatnonzero(f==1-k),np.flatnonzero(f==k),'nested') for k in (0,1)]+[('full',np.arange(len(b)),None,'validation')]
    for stage,ix,te,kind in stages:
        w=fit(x,y,ix,ids,OUT/f'{stage}.npz');xx=vx if te is None else x[(te[:,None]*4+np.arange(4)).ravel()];s=np.asarray(xx@w).reshape(-1,4)
        if te is None:outputs[kind]=s
        else:outputs[kind][te]=s
        print('COMPLETE',stage,flush=True)
    outputs['nested'][f==2]=outputs['oof'][f==2];np.savez(OUT/'entities_scores.npz',**outputs)
    te=f==2;choice=outputs['oof'][te].argmax(1);r=dict(correct=int(y[te][np.arange(te.sum()),choice].sum()),n=int(te.sum()),parse_gate_applied=False);dump(OUT/'held_standalone.json',r);print(r,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
