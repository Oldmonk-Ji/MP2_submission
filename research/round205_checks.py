from round22_common import ROOT,load,read,dump,sha
from round205_selector import predict
from round203_features import build
from round201_trace import checks
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np

if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');out=ROOT/'experiments/round205';x=sparse.load_npz(ROOT/'experiments/round203/validation_features.npz');report=dict(trace_checks=checks(),test_opened=False,source_sha256=sha(__file__))
    with np.load(out/'bundle_strict/strict.npz') as w:s=np.asarray(x@w['weight']).reshape(-1,4)
    with np.load(ROOT/'experiments/round204/scores.npz') as z:assert np.array_equal(s,z['validation'])
    fresh=build(b[:8]);assert not (fresh-x[:32]).nnz
    report.update(all_validation_export_features_match=True,fresh_features_match=True)
    for arm in ('control','strict'):
        path=out/f'bundle_{arm}';got=predict(b[:8],{'embeddings':a['embeddings'][:8]},path);assert got==read(out/f'validation_{arm}_predictions.json')['selected'][:8]
        assert got==predict(b[:3],{'embeddings':a['embeddings'][:3]},path)+predict(b[3:8],{'embeddings':a['embeddings'][3:8]},path)
        for q in ('Which solution is reliable?','Ann has 5 apples. How many apples does Ann have?'):
            row=dict(question=q,candidates=[dict(index=j,text='#### 5') for j in range(4)]);p=predict([row],{'embeddings':np.zeros((1,4,1024),np.float32)},path);assert len(p)==1 and p[0] in range(4)
        report[arm]=dict(fresh_predictions_match=True,batch_invariance=True,zero_pair_inputs=True)
    perm=[2,0,3,1];row=dict(b[0],candidates=[dict(b[0]['candidates'][j],index=k) for k,j in enumerate(perm)]);assert not (build([row])-fresh[:4][perm]).nnz;report['strict_feature_candidate_permutation']=True
    dump(out/'feature_checks.json',report);print(report,flush=True)
