from round22_common import ROOT,load,read,dump,sha
from round112_decoupled import aggregate,predict,checks
from round107_latent import subset
from round80_train import fit
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round112'
if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    dump(OUT/'plan.json',dict(hypothesis='Process alignment improved fit-nested root decisions from1608 to2820 of3170 but candidate score did not improve. Separate within-candidate explanation preference from cross-candidate answer ranking.',method='Freeze each corresponding round110 teacher, use softmax root weights to pool root features, then fit original C1 parent-balanced candidate ranker. No extra search.',teacher_fitted_only_on_same_allowed_parents=True,checks=checks(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round112*.py')},test_opened=False))
    b,a,y,ids=load('train');base=ROOT/'experiments/round107';x=sparse.load_npz(base/'train_root_features.npz');ptr=np.load(base/'train_root_ptr.npy');vx=sparse.load_npz(base/'validation_root_features.npz');vp=np.load(base/'validation_root_ptr.npy')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    outputs=dict(oof=np.zeros((len(b),4)),nested=np.zeros((len(b),4)))
    stages=[(f'oof{k}',np.flatnonzero(f!=k),np.flatnonzero(f==k),'oof') for k in range(3)]+[(f'nested{k}',np.flatnonzero(f==1-k),np.flatnonzero(f==k),'nested') for k in (0,1)]+[('full',np.arange(len(b)),None,'validation')]
    for stage,ix,te,kind in stages:
        path=ROOT/f'experiments/round110/{stage}.npz';assert read(path.with_suffix('.json'))['training_ids']==ids[ix].tolist()
        with np.load(path) as z:teacher=z['weight']
        features=aggregate(x,ptr,teacher);ranker=fit(features,y,ix,ids,OUT/f'{stage}.npz')
        assert np.allclose(predict(x,ptr,teacher,ranker).ravel(),features@ranker,atol=2e-5)
        xx,pp=(vx,vp) if te is None else subset(x,ptr,te);s=predict(xx,pp,teacher,ranker)
        if te is None:outputs[kind]=s
        else:outputs[kind][te]=s
        print('COMPLETE',stage,flush=True)
    outputs['nested'][f==2]=outputs['oof'][f==2];np.savez(OUT/'decoupled_scores.npz',**outputs)
    te=f==2;choice=outputs['oof'][te].argmax(1);report=dict(correct=int(y[te][np.arange(te.sum()),choice].sum()),n=int(te.sum()),parse_gate_applied=False)
    dump(OUT/'held_standalone.json',report);print('STANDALONE',report,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
