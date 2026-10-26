from round22_common import ROOT,load,read,dump,sha
from round153_features import build,summarize
from round154_selector import predict
from round82_relations import probabilities
from scipy import sparse
from threadpoolctl import threadpool_limits
import numpy as np,copy
if __name__=='__main__':
    threadpool_limits(4);b,a,y,ids=load('validation');base=ROOT/'experiments/round153';out=ROOT/'experiments/round154_corrected';bundle=out/'bundle_markup'
    with np.load(base/'validation_features.npz') as z:r,e=z['root'],z['extra']
    with np.load(ROOT/'experiments/round78/validation_features.npz') as z:changed=np.flatnonzero((r!=z['root']).any((1,2))|(e!=z['extra']).any((1,2)))
    selected=sorted(set(range(8))|set(changed[:8].tolist()));rows=[b[i] for i in selected];emb={'embeddings':a['embeddings'][selected]};actual=build(rows,'local');assert np.array_equal(actual[0],r[selected]);assert np.array_equal(actual[1],e[selected])
    cr=(np.array(selected)[:,None]*4+np.arange(4)).ravel();ptr=np.load(base/'validation_ptr.npy');pr=np.concatenate([np.arange(ptr[i],ptr[i+1]) for i in selected]);packets=read(base/'validation_packets.json')
    for pos,name,rr in [(2,'flow',cr),(3,'pairs',pr),(6,'structure',cr)]:assert not (actual[pos]-sparse.load_npz(base/f'validation_{name}.npz')[rr]).nnz
    assert actual[5]==[packets[i] for i in selected]
    for name in ('flow','structure'):
        xx=sparse.load_npz(base/f'validation_{name}.npz')
        with np.load(bundle/f'{name}.npz') as w:score=np.asarray(xx@w['weight']).reshape(-1,4)
        with np.load(out/f'{name}_scores.npz') as z:assert np.array_equal(score,z['validation'])
    with np.load(bundle/'relation.npz') as w:score=summarize(probabilities(sparse.load_npz(base/'validation_pairs.npz'),w),ptr,packets)
    with np.load(out/'relation_scores.npz') as z:assert np.array_equal(score,z['validation'])
    got=predict(rows,emb,bundle);assert got==[read(out/'validation_markup_predictions.json')['selected'][i] for i in selected]
    assert got==predict(rows[:3],{'embeddings':emb['embeddings'][:3]},bundle)+predict(rows[3:],{'embeddings':emb['embeddings'][3:]},bundle)
    for q in ('Which solution is reliable?','Ann has 5 apples. How many apples does Ann have?'):
        rr=[dict(question=q,candidates=[dict(index=j,text='#### 5') for j in range(4)])];answer=predict(rr,{'embeddings':np.zeros((1,4,1024),np.float32)},bundle);assert len(answer)==1 and answer[0] in range(4)
    dump(out/'feature_checks.json',dict(fresh_rows_checked=len(rows),changed_rows_checked=int(sum(i in changed for i in selected)),all_validation_auxiliary_scores_match_export=True,fresh_program_features_match_cache=True,fresh_predictions_match=True,batch_invariance=True,zero_source_pair_inputs=True,test_opened=False,source_sha256=sha(__file__)));print('PASS',len(rows),flush=True)
