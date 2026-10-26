from round22_common import ROOT,load,read,dump,sha
from round133_context import probabilities as additive,question_context,checks as additive_checks
from round136_inference import probabilities as interactive,checks as interaction_checks
from round134_selector import predict as additive_predict
from round136_selector import predict as interactive_predict
from round83_features import summarize
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');x=sparse.load_npz(ROOT/'experiments/round83/validation_pair_features.npz');ptr=np.load(ROOT/'experiments/round83/validation_pair_ptr.npy');packets=read(ROOT/'experiments/round83/validation_packets.json');q=question_context(a);owners=np.repeat(np.arange(len(b)),np.diff(ptr));report=dict(additive_checks=additive_checks(),interaction_checks=interaction_checks(),source_sha256=sha(__file__),test_opened=False)
    for n,arm,prob,fn in [(134,'context',additive,additive_predict),(136,'interaction',interactive,interactive_predict)]:
        bundle=ROOT/f'experiments/round{n}/bundle_{arm}'
        with np.load(bundle/'relation.npz') as w:r=summarize(prob(x,q,owners,w),ptr,packets)
        with np.load(ROOT/f'experiments/round{n}/relation_scores.npz') as z:assert np.array_equal(r,z['validation'])
        rows=b[:8];aa={'embeddings':a['embeddings'][:8]};got=fn(rows,aa,bundle);assert got==read(ROOT/f'experiments/round{n}/validation_{arm}_predictions.json')['selected'][:8]
        assert got==fn(rows[:3],{'embeddings':aa['embeddings'][:3]},bundle)+fn(rows[3:],{'embeddings':aa['embeddings'][3:]},bundle)
        empty_quantity=[dict(question='Which solution is reliable?',candidates=[dict(index=j,text='#### 1') for j in range(4)])]
        answer=fn(empty_quantity,{'embeddings':np.zeros((1,4,1024),np.float32)},bundle);assert len(answer)==1 and answer[0] in range(4)
        report[f'round{n}_all_validation_relation_features_match_export']=True;report[f'round{n}_fresh_predictions_match']=True;report[f'round{n}_batch_invariance']=True;report[f'round{n}_zero_quantity_input_supported']=True
    dump(ROOT/'experiments/round136/feature_checks.json',report);print(report)
