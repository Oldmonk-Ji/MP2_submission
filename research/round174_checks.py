from round22_common import ROOT,load,read,dump,sha
from round174_selector import predict
from round172_features import build
from round162_decisions import summarize,checks
from round159_factors import probabilities
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');base=ROOT/'experiments/round162';out=ROOT/'experiments/round174';bundle=out/'bundle_literal'
    x=sparse.load_npz(base/'validation_features.npz');u=sparse.load_npz(ROOT/'experiments/round172/validation_source.npz');c=sparse.load_npz(base/'validation_constant.npz');ptr=np.load(base/'validation_ptr.npy');uptr=np.load(base/'validation_uptr.npy');cptr=np.load(base/'validation_cptr.npy');packets=read(base/'validation_packets.json')
    with np.load(bundle/'literal.npz') as w:score=summarize(probabilities(u,w,'source'),probabilities(c,w,'constant'),probabilities(x,w,'scope'),uptr,cptr,ptr,packets)
    with np.load(out/'literal_scores.npz') as z:assert np.array_equal(score,z['validation'])
    actual=build(b[:8],'local');assert not (actual[7]-x[:ptr[8]]).nnz;assert not (actual[10]-u[:uptr[8]]).nnz;assert not (actual[11]-c[:cptr[8]]).nnz;assert actual[9]==packets[:8];assert np.array_equal(actual[12],uptr[:9]);assert np.array_equal(actual[13],cptr[:9])
    report=dict(**checks(),all_validation_features_match_export=True,fresh_feature_blocks_match_cache=True,test_opened=False,source_sha256=sha(__file__))
    for arm in ('control','literal'):
        path=out/f'bundle_{arm}';got=predict(b[:8],{'embeddings':a['embeddings'][:8]},path);assert got==read(out/f'validation_{arm}_predictions.json')['selected'][:8]
        assert got==predict(b[:3],{'embeddings':a['embeddings'][:3]},path)+predict(b[3:8],{'embeddings':a['embeddings'][3:8]},path)
        for q in ('Which solution is reliable?','Ann has 5 apples. How many apples does Ann have?'):
            rows=[dict(question=q,candidates=[dict(index=j,text='#### 5') for j in range(4)])];answer=predict(rows,{'embeddings':np.zeros((1,4,1024),np.float32)},path);assert len(answer)==1 and answer[0] in range(4)
        report[arm+'_fresh_predictions_match']=True;report[arm+'_batch_invariance']=True;report[arm+'_zero_pair_inputs']=True
    dump(out/'feature_checks.json',report);print(report,flush=True)
