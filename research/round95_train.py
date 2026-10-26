"""Add conservative TRAIN-only relation labels shared by correct candidates."""
from round22_common import ROOT,load,read,dump,sha
from round82_relations import probabilities,LABELS
from round94_context import features as target_feature,matrix as target_matrix,join
from sklearn.linear_model import LogisticRegression
from collections import Counter
from itertools import combinations
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round95'

def consensus(packet,labels):
    positive=np.flatnonzero(labels)
    if len(positive)<2 or any(not packet[j] for j in positive):return {}
    roots=[dict(r) for j in positive for r in packet[j]]
    return {k:v for k,v in roots[0].items() if all(r.get(k)==v for r in roots[1:])}

def fit(x,y,parents,source_weight,mask,path):
    count=Counter(parents[mask]);ww=np.asarray([1/count[p] for p in parents[mask]])*source_weight[mask]
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=82082).fit(x[mask],y[mask],sample_weight=ww)
    assert np.array_equal(m.classes_,np.arange(6));np.savez(path,weight=m.coef_.astype(np.float32),bias=m.intercept_.astype(np.float32))
    dump(path.with_suffix('.json'),dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_pairs=int(mask.sum()),C=1,iterations=m.n_iter_.tolist(),pseudo_parent_weight=.5))
    return m

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter()
    dump(OUT/'plan.json',dict(rule='At least two final-correct candidates, each with a supported program; retain only pair labels present and equal in EVERY extracted supported root across these candidates. Add only parents without existing round82 reference relation labels. Exclude all 24 reviewed holdout parents.',pseudo_parent_weight=.5,C=1,source_sha256=sha(__file__),feature_sha256=sha(ROOT/'research/round94_context.py'),factorial_ablation='round82 base; round93 new labels; round94 target interactions; round95 both; fixed unchanged C and training recipe',fit_audit=read(ROOT/'experiments/train_fit_candidate_relation_consensus_audit_20261026.json'),test_opened=False,labels_are_weak_not_human_gold=True))
    b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)};old=read(ROOT/'experiments/round82/labels.json');old_parents={r['train_index'] for r in old};excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')}
    packets=read(ROOT/'experiments/round83/train_packets.json');ptr=np.load(ROOT/'experiments/round83/train_pair_ptr.npy');allx=sparse.load_npz(ROOT/'experiments/round83/train_pair_features.npz');oldx=sparse.load_npz(ROOT/'experiments/round82/pair_features.npz')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    new=[];new_indices=[]
    for i,idx in enumerate(ids):
        if int(idx) in old_parents or int(idx) in excluded:continue
        labels=consensus(packets[i],y[i]);n=int((1+np.sqrt(1+8*(ptr[i+1]-ptr[i])))/2);pairs=list(combinations(range(n),2));assert len(pairs)==ptr[i+1]-ptr[i]
        for k,label in sorted(labels.items()):
            left,right=pairs[k];new.append(dict(train_index=int(idx),fold=int(folds[i]),left=left,right=right,label=int(label),source='correct_candidate_unanimous_weak'));new_indices.append(ptr[i]+k)
    rows=old+new;x=join(sparse.vstack((oldx,allx[new_indices]),format='csr'),target_matrix([target_feature(b[index[r['train_index']]]['question'],r['left'],r['right']) for r in rows]));targets=np.asarray([r['label'] for r in rows]);parents=np.asarray([r['train_index'] for r in rows]);f=np.asarray([r['fold'] for r in rows]);weights=np.r_[np.ones(len(old)),np.full(len(new),.5)];sparse.save_npz(OUT/'pair_features.npz',x);dump(OUT/'labels.json',rows)
    stats=dict(original_parents=len(old_parents),new_parents=len({r['train_index'] for r in new}),original_pairs=len(old),new_pairs=len(new),new_fit_parents=len({r['train_index'] for r in new if r['fold']!=2}),new_held_parents=len({r['train_index'] for r in new if r['fold']==2}),held_review_excluded=not bool(set(parents)&excluded))
    dump(OUT/'supervision_stats.json',stats);print('SUPERVISION',stats,flush=True);oof=np.zeros((len(rows),6))
    for held in range(3):
        m=fit(x,targets,parents,weights,f!=held,OUT/f'oof{held}.npz');oof[f==held]=m.predict_proba(x[f==held]);print('OOF',held,float((oof[f==held].argmax(1)==targets[f==held]).mean()),flush=True)
    for held in (0,1):fit(x,targets,parents,weights,f==1-held,OUT/f'nested{held}.npz')
    fit(x,targets,parents,weights,np.ones(len(rows),bool),OUT/'full.npz');np.savez(OUT/'oof_scores.npz',probability=oof,labels=targets,parents=parents,folds=f)
    te=(f==2)&(np.arange(len(rows))<len(old));pred=oof[te].argmax(1);report=dict(original_reference_held_pairs=int(te.sum()),original_reference_held_correct=int((pred==targets[te]).sum()),by_class={LABELS[k]:dict(n=int((targets[te]==k).sum()),correct=int(((targets[te]==k)&(pred==k)).sum())) for k in range(6)},test_opened=False)
    dump(OUT/'held_relation_classification.json',report);print('HELD',report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
