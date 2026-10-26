from round22_common import ROOT,load,read,dump,sha
from round208_selector import predict
from round208_features import build, predict as root_predict
from round107_latent import checks
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');out=ROOT/'experiments/round208';base=ROOT/'experiments/round206';x=sparse.load_npz(base/'validation_root_features.npz');ptr=np.load(base/'validation_root_ptr.npy');xx,pp=build(b[:8]);assert np.array_equal(pp,ptr[:33]);assert not (xx-x[:ptr[32]]).nnz
    report=dict(test_opened=False,source_sha256=sha(__file__),gradient_checks=checks(),fresh_root_features_match=True)
    for arm in ('pool_only','latent'):
        with np.load(out/f'bundle_{arm}/strict.npz') as w:s=root_predict(x,ptr,w['weight'])
        with np.load(ROOT/f'experiments/round207/{arm}_scores.npz') as z:assert np.array_equal(s,z['validation'])
    for arm in ('control','mean','pool_only','latent'):
        path=out/f'bundle_{arm}';got=predict(b[:8],{'embeddings':a['embeddings'][:8]},path);assert got==read(out/f'validation_{arm}_predictions.json')['selected'][:8];assert got==predict(b[:3],{'embeddings':a['embeddings'][:3]},path)+predict(b[3:8],{'embeddings':a['embeddings'][3:8]},path)
        for q in ('Which solution is reliable?','Ann has 5 apples. How many apples does Ann have?'):
            row=dict(question=q,candidates=[dict(index=j,text='#### 5') for j in range(4)]);p=predict([row],{'embeddings':np.zeros((1,4,1024),np.float32)},path);assert len(p)==1 and p[0] in range(4)
        report[arm]=dict(fresh_predictions_match=True,batch_invariance=True,zero_pair_inputs=True)
    dump(out/'feature_checks.json',report);print(report,flush=True)
