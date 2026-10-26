"""Build reliable arithmetic relation targets before candidate selector integration."""
from round22_common import ROOT,load,read,dump,sha
from round82_relations import normalize,relations,pair_features,matrix,LABELS
from round29_programs import value
from round44_programs import locations
from sklearn.linear_model import LogisticRegression
from collections import Counter,defaultdict
from threadpoolctl import threadpool_limits
from scipy import sparse
import numpy as np,time
OUT=ROOT/'experiments/round82'

def fit(x,y,parents,mask,path):
    count=Counter(parents[mask]);ww=np.asarray([1/count[p] for p in parents[mask]])
    m=LogisticRegression(C=1,solver='lbfgs',max_iter=700,random_state=82082).fit(x[mask],y[mask],sample_weight=ww)
    assert np.array_equal(m.classes_,np.arange(6));path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,weight=m.coef_.astype(np.float32),bias=m.intercept_.astype(np.float32))
    dump(path.with_suffix('.json'),dict(training_parent_ids=sorted(map(int,set(parents[mask]))),training_pairs=int(mask.sum()),C=1,iterations=m.n_iter_.tolist(),class_counts={LABELS[k]:int((y[mask]==k).sum()) for k in range(6)}))
    return m

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(source='https://aclanthology.org/D15-1202.pdf',adaptation='Normalize ADD/SUB and MUL/DIV chains; skip source-quantity reuse and unsupported operators; classifier is TRAIN-only auxiliary evidence, not a hard validity rule.',C=1,dimension=16384,labels=LABELS,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round82*.py')}))
    b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)}
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:folds=z['folds']
    heldparents={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')};sources=[]
    for r in read(ROOT/'experiments/round63/accepted_reference_programs.json'):sources.append(dict(**r,source_kind='strict_reference_weak'))
    for r in read(ROOT/'experiments/round51/reviewed_programs.json'):
        if r['train_index'] not in heldparents:sources.append(dict(**r,source_kind='reviewed'))
    table=defaultdict(dict);conflicts=set();stats=Counter();checked=0;rng=np.random.default_rng(82082)
    for r in sources:
        idx=r['train_index'];assert idx not in heldparents;assert idx in index;p=r['program'];rel=relations(p);stats['source_programs']+=1
        if rel is None:stats['unsupported_or_reused_source']+=1;continue
        if not rel:stats['no_quantity_pair']+=1;continue
        norm=normalize(p);n=max(v[1] for _,v in locations(p) if v[0]=='N')+1
        for repeat in range(4):
            qs=[dict(value=str(v)) for v in rng.integers(1,30,size=n)]
            try:assert value(p,qs)==value(norm,qs);checked+=1
            except ZeroDivisionError:pass
        for pair,label in rel.items():
            old=table[idx].get(pair)
            if old is not None and old!=label:conflicts.add(idx)
            table[idx][pair]=label
        stats['accepted_source_programs']+=1
    rows=[];fs=[]
    for idx,pairs in sorted(table.items()):
        if idx in conflicts:continue
        ii=index[idx];q=b[ii]['question']
        for (i,j),label in sorted(pairs.items()):rows.append(dict(train_index=idx,fold=int(folds[ii]),left=i,right=j,label=label));fs.append(pair_features(q,i,j))
    x=matrix(fs);targets=np.asarray([r['label'] for r in rows]);parents=np.asarray([r['train_index'] for r in rows]);f=np.asarray([r['fold'] for r in rows]);sparse.save_npz(OUT/'pair_features.npz',x);dump(OUT/'labels.json',rows)
    dump(OUT/'supervision_stats.json',dict(counts=dict(stats),supervised_parents=len(set(parents)),pairs=len(rows),conflicting_parents=sorted(conflicts),normalization_exact_arithmetic_checks=checked,held_review_parents_excluded=True,fit_parents=len(set(parents[f!=2])),held_parents=len(set(parents[f==2])),labels_are_reference_aligned_not_new_human_gold=True));print('SUPERVISION',read(OUT/'supervision_stats.json'),flush=True)
    oof=np.zeros((len(rows),6));seen=np.zeros(len(rows),int)
    for held in range(3):
        m=fit(x,targets,parents,f!=held,OUT/f'oof{held}.npz');oof[f==held]=m.predict_proba(x[f==held]);seen[f==held]+=1
        print('OOF',held,'accuracy',float((oof[f==held].argmax(1)==targets[f==held]).mean()),flush=True)
    assert (seen==1).all()
    for held in (0,1):fit(x,targets,parents,f==1-held,OUT/f'nested{held}.npz')
    fit(x,targets,parents,np.ones(len(rows),bool),OUT/'full.npz');np.savez(OUT/'oof_scores.npz',probability=oof,labels=targets,parents=parents,folds=f)
    te=f==2;majority=Counter(targets[~te]).most_common(1)[0][0];pred=oof[te].argmax(1)
    report=dict(held_pairs=int(te.sum()),held_correct=int((pred==targets[te]).sum()),majority_correct=int((targets[te]==majority).sum()),majority_class=LABELS[majority],by_class={LABELS[k]:dict(n=int((targets[te]==k).sum()),correct=int(((targets[te]==k)&(pred==k)).sum())) for k in range(6)},test_opened=False)
    dump(OUT/'held_relation_classification.json',report);print('HELD_RELATIONS',report,flush=True);dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False,auxiliary_supervision_only=True))
