"""Replace half of ambiguous candidate-positive supervision with reference programs.

Each mixed TRAIN parent retains exactly unit positive-pair mass. References
never enter inference, held labels never enter a diagnostic fit, C stays fixed.
"""
from round22_common import ROOT,load,read,dump,sha
from round103_features import row_features,matrix
from round101_templates import canonical,execute
from round29_programs import quantities
from features import parse_answer
from fractions import Fraction as F
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time
OUT=ROOT/'experiments/round105'

def fit(x,xr,ridx,y,ix,ids,path):
    pos=[];neg=[];ww=[];ref=[];refneg=[];rw=[];refparents=[];mass=[]
    for i in ix:
        pp=np.flatnonzero(y[i]);nn=np.flatnonzero(y[i]==0)
        if not len(nn):continue
        use=ridx[i]>=0;scale=.5 if use else 1.
        for a in pp:
            for b in nn:pos.append(4*i+a);neg.append(4*i+b);ww.append(scale/(len(pp)*len(nn)))
        if use:
            refparents.append(int(ids[i]))
            for b in nn:ref.append(ridx[i]);refneg.append(4*i+b);rw.append(.5/len(nn))
        mass.append(scale+(.5 if use else 0))
    assert np.allclose(mass,1.)
    d=sparse.vstack((x[pos]-x[neg],xr[ref]-x[refneg]),format='csr');weight=np.r_[ww,rw];xx=sparse.vstack((d,-d),format='csr');yy=np.r_[np.ones(len(weight)),np.zeros(len(weight))]
    m=LogisticRegression(C=1,fit_intercept=False,solver='liblinear',max_iter=1500,random_state=80080).fit(xx,yy,sample_weight=np.tile(weight,2));w=m.coef_[0].astype(np.float32);np.savez(path,weight=w)
    dump(path.with_suffix('.json'),dict(training_ids=ids[ix].tolist(),reference_supervised_parents=refparents,C=1,iterations=m.n_iter_.tolist(),all_parent_positive_pair_mass_one=True,candidate_positive_pairs=len(ww),reference_positive_pairs=len(rw)));return w

if __name__=='__main__':
    threadpool_limits(4);tick=time.perf_counter();assert not (OUT/'training_plan.json').exists()
    dump(OUT/'training_plan.json',dict(predeclared_plan=read(ROOT/'experiments/next_after_round104_predeclared.json'),fit_audit=read(OUT/'fit_audit.json')['counts'],source_sha256=sha(__file__),feature_sha256=sha(ROOT/'research/round103_features.py'),test_opened=False))
    b,a,y,ids=load('train');index={int(k):i for i,k in enumerate(ids)};x=sparse.load_npz(ROOT/'experiments/round103/train_local.npz');vx=sparse.load_npz(ROOT/'experiments/round103/validation_local.npz')
    excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')};sources=read(ROOT/'experiments/round63/accepted_reference_programs.json')+read(ROOT/'experiments/round51/reviewed_programs.json');accepted=defaultdict(dict);rejected=0
    for r in sources:
        idx=r['train_index']
        if idx in excluded:continue
        i=index[idx];q=b[i]['question'];truth={F(parse_answer(b[i]['candidates'][j]['text'])) for j in np.flatnonzero(y[i])}
        try:p=canonical(r['program']);okay=execute(p,[F(x['value']) for x in quantities(q)]) in truth
        except (ValueError,ZeroDivisionError,IndexError):okay=False
        if okay:accepted[idx][repr(p)]=r['program']
        else:rejected+=1
    docs=[];ridx=np.full(len(b),-1,dtype=int);reference_ids=[]
    for idx,programs in sorted(accepted.items()):
        i=index[idx];ridx[i]=len(docs);reference_ids.append(idx);docs.extend(row_features(b[i],[list(programs.values())],'local'))
    xr=matrix(docs);sparse.save_npz(OUT/'reference_features.npz',xr);np.save(OUT/'reference_row_index.npy',ridx);dump(OUT/'reference_ids.json',reference_ids)
    dump(OUT/'supervision_stats.json',dict(reference_parents=len(reference_ids),rejected_programs=rejected,reference_covered_mixed_parents=int(((ridx>=0)&(y.min(1)!=y.max(1))).sum()),held_review_excluded=not bool(set(reference_ids)&excluded)));print('SUPERVISION',read(OUT/'supervision_stats.json'),flush=True)
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:folds=z['folds'];assert np.array_equal(ids,z['ids'])
    oof=np.zeros((len(b),4));ns=np.zeros_like(oof)
    for held in range(3):
        ix=np.flatnonzero(folds!=held);te=np.flatnonzero(folds==held);w=fit(x,xr,ridx,y,ix,ids,OUT/f'oof{held}.npz');rows=(te[:,None]*4+np.arange(4)).ravel();oof[te]=np.asarray(x[rows]@w).reshape(-1,4)
    ns[folds==2]=oof[folds==2]
    for held in (0,1):
        ix=np.flatnonzero(folds==1-held);te=np.flatnonzero(folds==held);w=fit(x,xr,ridx,y,ix,ids,OUT/f'nested{held}.npz');rows=(te[:,None]*4+np.arange(4)).ravel();ns[te]=np.asarray(x[rows]@w).reshape(-1,4)
    w=fit(x,xr,ridx,y,np.arange(len(b)),ids,OUT/'full.npz');vs=np.asarray(vx@w).reshape(-1,4);np.savez(OUT/'reference_scores.npz',oof=oof,nested=ns,validation=vs);te=folds==2;pred=oof[te].argmax(1);report=dict(correct=int(y[te][np.arange(te.sum()),pred].sum()),n=int(te.sum()),candidate_parse_gate_not_applied=True);dump(OUT/'held_standalone.json',report);print('STANDALONE',report,flush=True)
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,training_ids=ids.tolist(),test_opened=False))
