"""Add reviewed/reference-grounded relation supervision to signed flow learning."""
from round22_common import ROOT,load,read,dump,sha
from round80_features import flow,matrix
from round80_selector import augment
from round79_selector import design,project
from round68_selector import components
from round46_graph import graph
from round52_local import roles
from round3_features import grounding
from round3_train import pair_data
from round4_gates import apply_gate
from round44_programs import edits,valid
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits
from collections import Counter,defaultdict
import numpy as np,lightgbm as lgb,time,shutil
OUT=ROOT/'experiments/round81'

def difference(pairs):
    pp=[];nn=[]
    for p in pairs:
        q=p['question'];g=graph(q);rs,_=roles(q);pp.append(flow(q,g,rs,p['positive'],True));nn.append(flow(q,g,rs,p['negative'],True))
    return matrix(pp)-matrix(nn)

def fit(x,y,ix,ids,path,aux,parents):
    rows=(ix[:,None]*4+np.arange(4)).ravel();xx,yy,ww=pair_data(x[rows],y[ix]);use=np.flatnonzero(np.isin(parents,ids[ix]));counts=Counter(parents[use]);aw=np.asarray([1/counts[p] for p in parents[use]])
    xx=sparse.vstack([xx,aux[use],-aux[use]],format='csr');yy=np.r_[yy,np.ones(len(use)),np.zeros(len(use))];ww=np.r_[ww,aw,aw]
    m=LogisticRegression(C=1,fit_intercept=False,solver='liblinear',max_iter=1500,random_state=80080).fit(xx,yy,sample_weight=ww);w=m.coef_[0].astype(np.float32)
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,weight=w);dump(path.with_suffix('.json'),dict(training_ids=ids[ix].tolist(),auxiliary_parents=sorted(map(int,set(parents[use]))),auxiliary_pairs=len(use),C=1,iterations=m.n_iter_.tolist(),auxiliary_weighting='Each relation parent has same total pair weight as each mixed-answer candidate parent; no held relation labels'))
    return w

if __name__=='__main__':
    threadpool_limits(4);OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(arm='supervised',control='round80 signed',C=1,meta_rounds=200,meta_leaves=7,nested=True,test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round81*.py')},auxiliary_source='round67: 706 reviewed pairs plus 2155 strict reference/actual-wrong-candidate pairs; weak references are not human semantic gold'))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');weights=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']]
    pairs=read(ROOT/'experiments/round67/constraint_pairs.json');parents=np.asarray([p['train_index'] for p in pairs]);heldreview=read(ROOT/'experiments/round56/held_source_review_before_reference.json');assert not set(parents)&{p['train_index'] for p in heldreview};assert set(parents)<=set(ids)
    aux=difference(pairs);sparse.save_npz(OUT/'auxiliary_difference.npz',aux);dump(OUT/'supervision.json',dict(pairs=len(pairs),parents=len(set(parents)),identical_feature_pairs=int((aux.getnnz(axis=1)==0).sum()),held_relation_parents_excluded=True))
    xx=sparse.load_npz(ROOT/'experiments/round80/train_signed.npz');vxx=sparse.load_npz(ROOT/'experiments/round80/validation_signed.npz')
    with np.load(ROOT/'experiments/round68/oof.npz') as z:assert np.array_equal(z['ids'],ids);ss=z['components']
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:assert np.array_equal(z['ids'],ids);nested=z['components'];folds=z['folds']
    oof=np.zeros((len(b),4));ns=np.zeros_like(oof)
    for held in range(3):
        ix=np.flatnonzero(folds!=held);te=np.flatnonzero(folds==held);w=fit(xx,y,ix,ids,OUT/f'oof{held}.npz',aux,parents);rows=(te[:,None]*4+np.arange(4)).ravel();oof[te]=np.asarray(xx[rows]@w).reshape(-1,4)
    ns[folds==2]=oof[folds==2]
    for held in (0,1):
        ix=np.flatnonzero(folds==1-held);te=np.flatnonzero(folds==held);w=fit(xx,y,ix,ids,OUT/f'nested{held}.npz',aux,parents);rows=(te[:,None]*4+np.arange(4)).ravel();ns[te]=np.asarray(xx[rows]@w).reshape(-1,4)
    w=fit(xx,y,np.arange(len(b)),ids,OUT/'full.npz',aux,parents);vscores=np.asarray(vxx@w).reshape(-1,4);np.savez(OUT/'scores.npz',oof=oof,nested=ns,validation=vscores)
    hp=[]
    for r in heldreview:
        counts=Counter();seen=set()
        for kind,p,d in edits(r['program'],r['quantities']):
            ans=valid(p,r['quantities']);key=repr(p)
            if ans is None or ans==r['source_answer'] or key in seen or counts[kind]>=4:continue
            counts[kind]+=1;seen.add(key);hp.append(dict(train_index=r['train_index'],question=r['question'],positive=r['program'],negative=p,kind=kind))
    hx=difference(hp);relation=[]
    for label,file in [('final_only',ROOT/'experiments/round80/signed/oof2.npz'),('supervised',OUT/'oof2.npz')]:
        with np.load(file) as z:margin=np.asarray(hx@z['weight'])
        groups=defaultdict(list)
        for p,m in zip(hp,margin):groups[p['kind']].append(bool(m>0))
        relation.append(dict(arm=label,n=len(hp),correct=int((margin>0).sum()),ties=int((abs(margin)<1e-10).sum()),by_kind={k:dict(n=len(v),correct=sum(v)) for k,v in groups.items()}))
    dump(OUT/'held_relations.json',relation);print('RELATIONS',relation,flush=True)
    roots={}
    for split in ('train','validation'):
        with np.load(ROOT/f'experiments/round78/{split}_features.npz') as z:roots[split]=(z['root'],z['extra'])
    with np.load(ROOT/'experiments/round79/full_space.npz') as w:context=project(a,w);vc=project(va,w)
    with np.load(ROOT/'experiments/round79/diagnostic_space.npz') as w:nc=project(a,w)
    g=grounding(b);vs,vg=components(v,va,ROOT/'experiments/round25/selected_bundle');x,bs=design(b,ss,weights,g,*roots['train'],context,'context_influence');nx,nbs=design(b,nested,weights,g,*roots['train'],nc,'context_influence');vx,vbs=design(v,vs,weights,vg,*roots['validation'],vc,'context_influence')
    params=dict(objective='lambdarank',lambdarank_truncation_level=1,metric='None',num_leaves=7,min_data_in_leaf=100,learning_rate=.03,lambda_l2=50,seed=68068,verbosity=-1,num_threads=4,deterministic=True,force_col_wise=True)
    fitmask=folds!=2;held=folds==2;ax=augment(nx,ns);ds=lgb.Dataset(ax[fitmask].reshape(-1,ax.shape[-1]),label=y[fitmask].ravel(),group=np.full(fitmask.sum(),4),init_score=nbs[fitmask].ravel());hm=lgb.train(params,ds,num_boost_round=200);hc=apply_gate([r for r,k in zip(b,held) if k],nbs[held]+hm.predict(ax[held].reshape(-1,ax.shape[-1])).reshape(-1,4),'parse_and_repair',4);dump(OUT/'supervised_held.json',dict(correct=int(y[held][np.arange(held.sum()),hc].sum()),n=int(held.sum()),nested=True))
    ax=augment(x,oof);av=augment(vx,vscores);ds=lgb.Dataset(ax.reshape(-1,ax.shape[-1]),label=y.ravel(),group=np.full(len(b),4),init_score=bs.ravel());m=lgb.train(params,ds,num_boost_round=200)
    c=apply_gate(v,vbs+m.predict(av.reshape(-1,av.shape[-1])).reshape(-1,4),'parse_and_repair',4);bc=apply_gate(v,vbs,'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc];r=dict(arm='supervised',correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()))
    bundle=OUT/'bundle_supervised';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');m.save_model(str(bundle/'meta.txt'));shutil.copy2(ROOT/'experiments/round79/full_space.npz',bundle/'space.npz');shutil.copy2(OUT/'full.npz',bundle/'flow.npz');dump(bundle/'config.json',dict(arm='signed',weights=weights))
    pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=c.tolist();dump(OUT/'validation_supervised_predictions.json',pred);r['asset_bytes']=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert r['asset_bytes']<=16*1024**2;dump(OUT/'validation_records.json',[r]);print('VAL',r,flush=True)
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm='supervised',correct=r['correct'],bundle='experiments/round81/bundle_supervised'),selected='round81' if r['correct']>754 else 'round79',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
