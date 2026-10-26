"""Outer-fold-clean OOF conditioning for the relation residual diagnostic."""
from round22_common import ROOT,load,read,dump,sha
from round3_features import build
from features import metadata
from round3_train import text_fit
from train_models import fit_mlp
from round68_oof import fit as uniform_fit
from round22_selector import dense_forward
from round14_prepare import family
import numpy as np,torch,time
OUT=ROOT/'experiments/round77'

def prepare(b,a,y,ids):
    path=OUT/'nested';path.mkdir(parents=True)
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    with np.load(ROOT/'experiments/round68/oof.npz') as z:
        assert np.array_equal(z['ids'],ids);ss=z['components'].copy()
    for fold in range(3):
        partition=read(ROOT/f'experiments/round22/oof/fold{fold}/partition.json')
        assert set(ids[f==fold])==set(partition['held_ids'])
    cache={k:build(b,a,k) for k in ('base','ground','answer')};valid=metadata(b)['valid'];tick=time.perf_counter()
    for held in (0,1):
        tr=np.flatnonzero(f==1-held);te=np.flatnonzero(f==held)
        assert not {family(b[i]['question']) for i in tr}&{family(b[i]['question']) for i in np.flatnonzero(f!=1-held)}
        part=path/f'fold{held}';part.mkdir();dump(part/'partition.json',dict(training_ids=ids[tr].tolist(),held_ids=ids[te].tolist(),outer_held_excluded=ids[f==2].tolist()))
        for member in range(8):
            cfg=read(ROOT/f'experiments/round25/selected_bundle/member{member}/config.json');p=part/f'member{member}';p.mkdir()
            if member==3:s,_=text_fit([b[i] for i in tr],y[tr],[b[i] for i in te],cfg['mode'],cfg['C'],p)
            else:
                x=cache[cfg['features']]
                if member<3:
                    fit_mlp(x[tr],y[tr],None,None,valid[tr],cfg,p,fixed_epochs=cfg['fixed_epochs'])
                    with np.load(p/'model.npz') as w:s=dense_forward(x[te],w)
                else:
                    w,s=uniform_fit(x[tr],y[tr],x[te],cfg);np.savez(p/'model.npz',**w)
                dump(p/'config.json',cfg)
            ss[te,:,member]=s;print('NESTED',held,member,'seconds',round(time.perf_counter()-tick,2),flush=True)
    np.savez(path/'components.npz',ids=ids,components=ss,folds=f)
    dump(path/'complete.json',dict(seconds=time.perf_counter()-tick,outer_fold=2,all_conditioning_models_exclude_outer_fold=True,source_hash=sha(__file__)))
    return ss,f
