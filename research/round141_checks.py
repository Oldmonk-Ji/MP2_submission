from round22_common import ROOT,load,read,dump,sha
from round141_selector import predict
from round140_features import build
from round140_literals import summarize,checks
from round139_features import probabilities
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');base=ROOT/'experiments/round140';x=sparse.load_npz(base/'validation_features.npz');ptr=np.load(base/'validation_ptr.npy');packets=read(base/'validation_packets.json')
    with np.load(ROOT/'experiments/round141/bundle_literal/literal.npz') as w:score=summarize(probabilities(x,w),ptr,packets)
    with np.load(ROOT/'experiments/round141/literal_scores.npz') as z:assert np.array_equal(score,z['validation'])
    actual=build(b[:8],'local');d=actual[7]-x[:ptr[8]];assert not d.nnz;assert np.array_equal(actual[8],ptr[:9]) and actual[9]==packets[:8]
    report=dict(**checks(),all_validation_literal_features_match_export=True,fresh_literal_extraction_matches=True,source_sha256=sha(__file__),test_opened=False)
    for arm in ('control','literal'):
        bundle=ROOT/f'experiments/round141/bundle_{arm}';got=predict(b[:8],{'embeddings':a['embeddings'][:8]},bundle);assert got==read(ROOT/f'experiments/round141/validation_{arm}_predictions.json')['selected'][:8]
        assert got==predict(b[:3],{'embeddings':a['embeddings'][:3]},bundle)+predict(b[3:8],{'embeddings':a['embeddings'][3:8]},bundle)
        for q in ('Which solution is reliable?','Ann has 5 apples. How many apples does Ann have?'):
            rows=[dict(question=q,candidates=[dict(index=j,text='#### 5') for j in range(4)])];answer=predict(rows,{'embeddings':np.zeros((1,4,1024),np.float32)},bundle);assert len(answer)==1 and answer[0] in range(4)
        report[arm+'_fresh_predictions_match']=True;report[arm+'_batch_invariance']=True;report[arm+'_zero_pair_inputs_supported']=True
    dump(ROOT/'experiments/round141/feature_checks.json',report);print(report)
