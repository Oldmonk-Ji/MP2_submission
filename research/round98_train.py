from round22_common import ROOT,load,read,dump,sha
from round98_influence import labels,checks
from round98_features import documents,matrix
from sklearn.linear_model import LogisticRegression
from collections import Counter,defaultdict
from threadpoolctl import threadpool_limits
from scipy import sparse
import numpy as np,time
OUT=ROOT/'experiments/round98'

def fit(x,y,parents,mask,path):
    count=Counter(parents[mask]);ww=np.asarray([1/count[k] for k in parents[mask]])
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=82082).fit(x[mask],y[mask],sample_weight=ww)
    assert np.array_equal(m.classes_,np.arange(3));np.savez(path,weight=m.coef_.astype(np.float32),bias=m.intercept_.astype(np.float32));dump(path.with_suffix('.json'),dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_quantity_labels=int(mask.sum()),C=1,iterations=m.n_iter_.tolist()));return m

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();assert not (OUT/'training_plan.json').exists();assert sha(ROOT/'research/round98_influence.py')==read(OUT/'rule_freeze.json')['source_sha256'];checks()
    dump(OUT/'training_plan.json',dict(arms=['filtered','occurrence'],C=1,labels='0 no local influence; 1 positive; 2 negative exact derivative',difference='filtered removes duplicate numeric values and 0/1, occurrence retains reliable source slots; same feature definitions and fit budget',source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round98*.py')},test_opened=False))
    b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)}
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds']
    excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')};sources=[dict(r,kind='strict') for r in read(ROOT/'experiments/round63/accepted_reference_programs.json')]+[dict(r,kind='reviewed') for r in read(ROOT/'experiments/round51/reviewed_programs.json')];table=defaultdict(dict);conflicts=set()
    for r in sources:
        idx=r['train_index']
        if idx in excluded:continue
        for row in labels(b[index[idx]]['question'],r['program'],r['kind']=='reviewed'):
            k=row['quantity'];previous=table[idx].get(k)
            if previous is not None and previous['label']!=row['label']:conflicts.add(idx)
            table[idx][k]=row
    rows=[];docs=[]
    for idx,slots in sorted(table.items()):
        if idx in conflicts:continue
        i=index[idx];ds=documents(b[i]['question'])
        for k,row in sorted(slots.items()):rows.append(dict(train_index=idx,fold=int(folds[i]),**row));docs.append(ds[k])
    x=matrix(docs);targets=np.asarray([r['label'] for r in rows]);parents=np.asarray([r['train_index'] for r in rows]);f=np.asarray([r['fold'] for r in rows]);oldok=np.asarray([not r['old_excluded'] for r in rows]);dump(OUT/'labels.json',rows);sparse.save_npz(OUT/'quantity_features.npz',x)
    dump(OUT/'supervision_stats.json',dict(parents=len(set(parents)),labels=len(rows),restored_labels=int((~oldok).sum()),conflicting_parents=sorted(conflicts),reviewed_holdout_excluded=True,reference_labels_are_weak=True));print('LABELS',read(OUT/'supervision_stats.json'),flush=True)
    reports=[]
    for arm in ('filtered','occurrence'):
        path=OUT/arm;path.mkdir();eligible=oldok if arm=='filtered' else np.ones(len(rows),bool);oof=np.zeros((len(rows),3))
        for held in range(3):
            m=fit(x,targets,parents,eligible&(f!=held),path/f'oof{held}.npz');oof[f==held]=m.predict_proba(x[f==held])
        for held in (0,1):fit(x,targets,parents,eligible&(f==1-held),path/f'nested{held}.npz')
        fit(x,targets,parents,eligible,path/'full.npz');np.savez(path/'oof_scores.npz',probability=oof,labels=targets,parents=parents,folds=f)
        report=dict(arm=arm,held_total=int((f==2).sum()),held_correct=int((oof[f==2].argmax(1)==targets[f==2]).sum()),restored_held_total=int(((f==2)&~oldok).sum()),restored_held_correct=int((oof[(f==2)&~oldok].argmax(1)==targets[(f==2)&~oldok]).sum()),by_class={str(k):dict(n=int(((f==2)&(targets==k)).sum()),correct=int(((f==2)&(targets==k)&(oof.argmax(1)==k)).sum())) for k in range(3)});reports.append(report);print('HELD',report,flush=True)
    dump(OUT/'held_results.json',reports);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,auxiliary_supervision_only=True,test_opened=False))
