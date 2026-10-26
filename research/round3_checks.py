"""Focused correctness checks for arithmetic boundaries and inference exports."""
from data_access import ROOT,CODE,load
import json
import numpy as np
from features import safe_arithmetic
from round3_features import equations,build
from round3_selector import scores,predict


def main():
    assert equations('0.6/9 = 2/15')[0][2] != equations('0.6/9 = 2/15')[0][3]
    assert abs(equations('12 * 5 = 60.')[0][2]-60)<1e-10
    assert not equations('x + 2 = 5')
    for expr in ['__import__("os")','2**10000','1/0']:
        try:safe_arithmetic(expr)
        except (ValueError,ZeroDivisionError):pass
        else:raise AssertionError(expr)
    # The audit hook rejects this request before the file is opened.
    try:open(CODE/'data/gsm8k_test.jsonl')
    except PermissionError:pass
    else:raise AssertionError('test isolation guard failed')
    b,a,_=load('validation');ids=[0,13,52,77,111,250,520]
    b=[dict(b[i],id='unseen-'+str(i)) for i in ids];a={'embeddings':a['embeddings'][ids]}
    perm=np.array([2,0,3,1]);bp=[dict(r,candidates=[dict(r['candidates'][int(i)],index=j) for j,i in enumerate(perm)]) for r in b]
    ap={'embeddings':a['embeddings'][:,perm]}
    for kind in ('base','ground','answer','answer_ground'):
        x=build(b,a,kind);xp=build(bp,ap,kind)
        assert np.isfinite(x).all() and np.allclose(xp,x[:,perm],atol=1e-5)
    checked=[]
    for path in (ROOT/'experiments/round3/full').iterdir():
        if not (path/'validation_scores.npz').exists():continue
        s=scores(b,a,path);expected=np.load(path/'validation_scores.npz')['scores'][ids]
        assert np.allclose(s,expected,atol=2e-4,rtol=1e-4),path.name
        assert np.allclose(scores(bp,ap,path),s[:,perm],atol=2e-4,rtol=1e-4),path.name
        choices=predict(b,a,path)
        for n in (1,3,7):assert predict(b[:n],{'embeddings':a['embeddings'][:n]},path)==choices[:n]
        assert predict([],{'embeddings':np.empty((0,4,1024))},path)==[]
        checked.append(path.name)
    result=dict(status='passed',models=checked,test_inputs_or_references_read=False,
        checks=['bounded arithmetic','symbolic boundary rejection','finite features','candidate permutation',
                'variable N 0/1/3/7','unknown IDs','exported scores','test access blocked before file open'])
    (ROOT/'experiments/round3/checks.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
