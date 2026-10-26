"""Fixed relational-feature ablation; no extra supervision or model search."""
from round22_common import ROOT,load,read,dump,sha
from round94_context import features,matrix,join
from round82_train import fit
from round82_relations import LABELS
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round94'

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    dump(OUT/'plan.json',dict(hypothesis='Target words must interact with operand-local roles, rather than shift operation priors identically for every quantity pair.',source_labels='round82 unchanged',additional_dimensions=8192,C=1,max_iter=700,feature_scale=.25,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/round94_context.py',ROOT/'research/round94_train.py']},test_opened=False))
    b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)};rows=read(ROOT/'experiments/round82/labels.json');x=join(sparse.load_npz(ROOT/'experiments/round82/pair_features.npz'),matrix([features(b[index[r['train_index']]]['question'],r['left'],r['right']) for r in rows]));targets=np.asarray([r['label'] for r in rows]);parents=np.asarray([r['train_index'] for r in rows]);f=np.asarray([r['fold'] for r in rows]);oof=np.zeros((len(rows),6));sparse.save_npz(OUT/'pair_features.npz',x)
    for held in range(3):
        m=fit(x,targets,parents,f!=held,OUT/f'oof{held}.npz');oof[f==held]=m.predict_proba(x[f==held]);print('OOF',held,float((oof[f==held].argmax(1)==targets[f==held]).mean()),flush=True)
    for held in (0,1):fit(x,targets,parents,f==1-held,OUT/f'nested{held}.npz')
    fit(x,targets,parents,np.ones(len(rows),bool),OUT/'full.npz');np.savez(OUT/'oof_scores.npz',probability=oof,labels=targets,parents=parents,folds=f)
    te=f==2;pred=oof[te].argmax(1);report=dict(held_pairs=int(te.sum()),held_correct=int((pred==targets[te]).sum()),by_class={LABELS[k]:dict(n=int((targets[te]==k).sum()),correct=int(((targets[te]==k)&(pred==k)).sum())) for k in range(6)},test_opened=False)
    dump(OUT/'held_relation_classification.json',report);print('HELD',report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
