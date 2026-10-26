"""Diagnose strict trace ambiguity before changing its learning objective."""
from round22_common import ROOT,load,read,dump,sha
from round107_latent import root_features,predict
from round103_features import matrix
from round109_alignment import annotate
from round44_programs import locations,tree
from round175_roles import kind
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,time,sys
OUT=ROOT/'experiments/round197'

if __name__=='__main__':
    threadpool_limits(4);mode=sys.argv[1];assert mode in ('fit','full');tick=time.perf_counter()
    if mode=='fit':
        OUT.mkdir();dump(OUT/'plan.json',dict(method='Diagnose root decisions in frozen strict179 traces using same103 per-root features. Reference-equivalent labels from115; exclude roots containing genuine ordinal slots according to frozen175 types. Missing or non-equivalent programs are not declared semantic errors. Only candidates with both aligned and unaligned alternatives receive relative alignment supervision.',source_sha256=sha(__file__),test_opened=False))
        for split in ('train','validation'):
            b,a,y,ids=load(split);packets=read(ROOT/f'experiments/round194/{split}_programs.json');chunks=[];counts=[]
            for begin in range(0,len(b),128):
                fs=[]
                for row,packet in zip(b[begin:begin+128],packets[begin:begin+128]):
                    ff,cc=root_features(row,packet);fs.extend(ff);counts.extend(cc)
                chunks.append(matrix(fs))
                if begin%512==0:print('ROOTS',split,begin,flush=True)
            x=sparse.vstack(chunks,format='csr');ptr=np.r_[0,np.cumsum(counts)];sparse.save_npz(OUT/f'{split}_root_features.npz',x);np.save(OUT/f'{split}_root_ptr.npy',ptr)
            with np.load(ROOT/'experiments/round195/full.npz') as z:w=z['weight']
            old=sparse.load_npz(ROOT/f'experiments/round194/{split}_features.npz');error=float(np.max(np.abs(predict(x,ptr,w,'mean').ravel()-old@w)));assert error<2e-5,error
            dump(OUT/f'{split}_checks.json',dict(root_count=x.shape[0],nnz=x.nnz,mean_score_max_error=error,test_opened=False));print('FEATURES',split,x.shape,x.nnz,flush=True)
    else:
        assert sha(__file__)==read(OUT/'plan.json')['source_sha256'];assert read(OUT/'fit_stats.json')['fit_only']
    b,a,y,ids=load('train');ix={int(k):i for i,k in enumerate(ids)};packets=read(ROOT/'experiments/round194/train_programs.json');ptr=np.load(OUT/'train_root_ptr.npy')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:f=z['folds'];assert np.array_equal(ids,z['ids'])
    raw=read(ROOT/f'experiments/round115/{mode}_sources.json');sources=[]
    for r in raw:
        row=b[ix[r['train_index']]]
        if not any(n[0]=='N' and kind(row['question'],n[1])=='ordinal' for _,n in locations(tree(r['program']))):sources.append(r)
    excluded={r['train_index'] for r in read(ROOT/'experiments/round56/held_source_review_before_reference.json')};keep=f!=2 if mode=='fit' else np.ones(len(b),bool)
    gold,active,report=annotate(b,ids,packets,ptr,sources,excluded,keep,lambda i:print('ALIGN',mode,i,flush=True));active &= y.ravel()>0
    np.save(OUT/f'{mode}_gold_roots.npy',gold);np.save(OUT/f'{mode}_active_candidates.npy',active)
    report.update(reference_programs_ordinal_filtered=len(raw)-len(sources),fit_only=mode=='fit',test_opened=False,seconds=time.perf_counter()-tick)
    if mode=='fit':
        x=sparse.load_npz(OUT/'train_root_features.npz');correct=total=0;chance=0.;examples=[]
        for fold in (0,1):
            with np.load(ROOT/f'experiments/round195/nested{fold}.npz') as z:s=np.asarray(x@z['weight']).ravel()
            for c in np.flatnonzero(active&np.repeat(f==fold,4)):
                lo,hi=ptr[c:c+2];ok=bool(gold[lo+np.argmax(s[lo:hi])]);correct+=ok;total+=1;chance+=gold[lo:hi].mean()
                if not ok and len(examples)<20:examples.append(dict(train_index=int(ids[c//4]),question=b[c//4]['question'],candidate=int(c%4),roots=int(hi-lo),aligned_roots=int(gold[lo:hi].sum())))
        report.update(nested_reference_choice_correct=correct,nested_reference_choice_n=total,uniform_expected_correct=chance);dump(OUT/'fit_root_choice_errors.json',examples)
    dump(OUT/f'{mode}_stats.json',report);print({k:v for k,v in report.items() if k!='alignment_parent_ids'},flush=True)
