"""Train relation decisions using only constraints shared by all reference alternatives."""
from round22_common import ROOT,load,read,dump,sha
from round119_consensus import consensus
from round82_train import fit
from round82_relations import pair_features,matrix,probabilities
from collections import defaultdict
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round120'
if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();OUT.mkdir();audit=read(ROOT/'experiments/round119/fit_audit.json');assert audit['source_sha256']==sha(ROOT/'research/round119_consensus.py')
    dump(OUT/'plan.json',dict(method='Same six-way relation classifier and C1 as round82, replacing labels with intersection over every complete reference lineage alternative.',source_sha256=sha(__file__),consensus_source_sha256=audit['source_sha256'],test_opened=False))
    b,a,y,ids=load('train');index={int(idx):i for i,idx in enumerate(ids)};groups=defaultdict(list);folds={}
    for r in read(ROOT/'experiments/round115/full_sources.json'):groups[r['train_index']].append(r['program']);folds[r['train_index']]=r['fold']
    rows=[];fs=[]
    for idx,ps in sorted(groups.items()):
        for (i,j),label in sorted(consensus(ps).items()):
            rows.append(dict(train_index=idx,fold=folds[idx],left=i,right=j,label=label));fs.append(pair_features(b[index[idx]]['question'],i,j))
    assert sorted((r['train_index'],r['left'],r['right'],r['label']) for r in rows if r['fold']!=2)==sorted((r['train_index'],r['left'],r['right'],r['label']) for r in read(ROOT/'experiments/round119/fit_consensus_labels.json'))
    x=matrix(fs);targets=np.asarray([r['label'] for r in rows]);parents=np.asarray([r['train_index'] for r in rows]);f=np.asarray([r['fold'] for r in rows]);sparse.save_npz(OUT/'pair_features.npz',x);dump(OUT/'labels.json',rows)
    oof=np.zeros((len(rows),6));old=np.zeros_like(oof)
    for held in range(3):
        m=fit(x,targets,parents,f!=held,OUT/f'oof{held}.npz');oof[f==held]=m.predict_proba(x[f==held])
        with np.load(ROOT/f'experiments/round82/oof{held}.npz') as w:old[f==held]=probabilities(x[f==held],w)
        print('COMPLETE',held,flush=True)
    for held in (0,1):fit(x,targets,parents,f==1-held,OUT/f'nested{held}.npz')
    fit(x,targets,parents,np.ones(len(rows),bool),OUT/'full.npz');np.savez(OUT/'oof_scores.npz',probability=oof,old_probability=old,labels=targets,parents=parents,folds=f)
    te=f==2;r=dict(n=int(te.sum()),old_correct=int((old[te].argmax(1)==targets[te]).sum()),new_correct=int((oof[te].argmax(1)==targets[te]).sum()),parents=len(set(parents)),pairs=len(rows),test_opened=False)
    dump(OUT/'held_relations.json',r);print(r,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
