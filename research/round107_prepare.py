from round22_common import ROOT,load,read,dump,sha
from round107_latent import root_features,predict,checks
from round103_features import matrix
from scipy import sparse
import numpy as np,time
OUT=ROOT/'experiments/round107'

if __name__=='__main__':
    OUT.mkdir();tick=time.perf_counter();dump(OUT/'plan.json',dict(predeclared=read(ROOT/'experiments/next_after_round106.json'),arms=['uniform_control','pool_only','latent_trained'],C=1,max_iterations=200,gradient_checks=checks(),source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round107*.py')},test_opened=False))
    for split in ('train','validation'):
        b,a,y,ids=load(split);packets=read(ROOT/f'experiments/round102/{split}_programs.json');chunks=[];counts=[]
        for begin in range(0,len(b),128):
            fs=[]
            for row,packet in zip(b[begin:begin+128],packets[begin:begin+128]):
                f,c=root_features(row,packet);fs.extend(f);counts.extend(c)
            chunks.append(matrix(fs))
            if begin%512==0:print('ROOT_FEATURES',split,begin,flush=True)
        x=sparse.vstack(chunks,format='csr');ptr=np.r_[0,np.cumsum(counts)];sparse.save_npz(OUT/f'{split}_root_features.npz',x);np.save(OUT/f'{split}_root_ptr.npy',ptr)
        count=np.array(counts);owner=np.repeat(np.arange(len(count)),count);agg=sparse.csr_matrix((1/count[owner],(owner,np.arange(len(owner)))),shape=(len(count),len(owner)));mean=agg@x;old=sparse.load_npz(ROOT/f'experiments/round103/{split}_local.npz');diff=mean-old;error=float(np.max(np.abs(diff.data),initial=0));assert error<2e-5,error
        with np.load(ROOT/'experiments/round103/local/full.npz') as z:w=z['weight']
        score_error=float(np.max(np.abs(predict(x,ptr,w,'mean').ravel()-old@w)));assert score_error<2e-5,score_error
        report=dict(n=len(b),root_count=x.shape[0],nnz=x.nnz,mean_feature_max_error=error,mean_score_max_error=score_error,test_opened=False);dump(OUT/f'{split}_preparation_checks.json',report);print('CHECK',split,report,flush=True)
    dump(OUT/'preparation_complete.json',dict(seconds=time.perf_counter()-tick,test_opened=False))
