from round22_common import ROOT,load,read,dump,sha
from round178_selector import predict
from round162_features import build as old_build
from round177_features import build as typed_build
from round162_decisions import summarize,checks
from round176_types import checks as type_checks
from round159_factors import probabilities
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');base=ROOT/'experiments/round162';out=ROOT/'experiments/round178';c=sparse.load_npz(base/'validation_constant.npz');ptr=np.load(base/'validation_ptr.npy');uptr=np.load(base/'validation_uptr.npy');cptr=np.load(base/'validation_cptr.npy');packets=read(base/'validation_packets.json');report=dict(**checks(),types=type_checks(),test_opened=False,source_sha256=sha(__file__))
    for arm in ('corrected_labels','corrected_types'):
        inputs=ROOT/'experiments/round177' if arm=='corrected_types' else base;build=typed_build if arm=='corrected_types' else old_build;x=sparse.load_npz(inputs/'validation_features.npz');u=sparse.load_npz(inputs/'validation_source.npz')
        with np.load(out/f'bundle_{arm}'/'literal.npz') as w:score=summarize(probabilities(u,w,'source'),probabilities(c,w,'constant'),probabilities(x,w,'scope'),uptr,cptr,ptr,packets)
        with np.load(out/f'literal_scores_{arm}.npz') as z:assert np.array_equal(score,z['validation'])
        actual=build(b[:8],'local');assert not (actual[7]-x[:ptr[8]]).nnz;assert not (actual[10]-u[:uptr[8]]).nnz;assert actual[9]==packets[:8];report[arm+'_all_val_export_features_match']=True;report[arm+'_fresh_features_match']=True
    for arm in ('control','corrected_labels','corrected_types'):
        path=out/f'bundle_{arm}';got=predict(b[:8],{'embeddings':a['embeddings'][:8]},path);assert got==read(out/f'validation_{arm}_predictions.json')['selected'][:8]
        assert got==predict(b[:3],{'embeddings':a['embeddings'][:3]},path)+predict(b[3:8],{'embeddings':a['embeddings'][3:8]},path)
        for q in ('Which solution is reliable?','Ann has 5 apples. How many apples does Ann have?'):
            rows=[dict(question=q,candidates=[dict(index=j,text='#### 5') for j in range(4)])];answer=predict(rows,{'embeddings':np.zeros((1,4,1024),np.float32)},path);assert len(answer)==1 and answer[0] in range(4)
        report[arm+'_fresh_predictions_match']=True;report[arm+'_batch_invariance']=True;report[arm+'_zero_pair_inputs']=True
    dump(out/'feature_checks.json',report);print(report,flush=True)
