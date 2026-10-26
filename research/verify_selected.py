"""Focused contract checks for the selected model, using validation inputs only."""
from data_access import ROOT,load
import hashlib
import json
import numpy as np
from selector import scores,predict,predict_with_confidence


def main():
    b,a,_=load('validation');ids=[0,13,52,77,111,250,520]
    bank=[dict(b[i],id='unseen-'+str(k)) for k,i in enumerate(ids)]
    arrays={'embeddings':a['embeddings'][ids]}
    path=ROOT/'experiments/round2/selected_bundle'
    chosen,conf=predict_with_confidence(bank,arrays,path)
    expected=json.loads((ROOT/'experiments/round2/reliability/official-validation-predictions-0.json').read_text())['selected']
    assert chosen==[expected[i] for i in ids]
    assert all(0<=p<=1 for p in conf)
    permutation=np.array([2,0,3,1])
    shuffled=[dict(r,candidates=[dict(r['candidates'][int(i)],index=j) for j,i in enumerate(permutation)]) for r in bank]
    assert np.allclose(scores(shuffled,{'embeddings':arrays['embeddings'][:,permutation]},path),
                       scores(bank,arrays,path)[:,permutation],atol=1e-4)
    for n in [1,3,7]:
        assert predict(bank[:n],{'embeddings':arrays['embeddings'][:n]},path)==chosen[:n]
    assert predict([],{'embeddings':np.empty((0,4,1024))},path)==[]
    result={'status':'passed','checks':['unknown IDs','variable N 0/1/3/7','candidate permutation score equivariance',
                                     'confidence range','official prediction consistency'],'test_access':False,
            'selected_model':json.loads((path/'config.json').read_text()),
            'model_sha256':hashlib.sha256((path/'member0/model.npz').read_bytes()).hexdigest()}
    (ROOT/'experiments/round2/interface_checks.json').write_text(json.dumps(result,indent=2))
    print('Selected-model contract checks passed.')


if __name__=='__main__':main()
