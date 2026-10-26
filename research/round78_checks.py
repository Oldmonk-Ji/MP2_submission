"""Independent arithmetic and scope checks for signed program influence."""
from round78_features import dual
from round22_common import ROOT,dump,sha
import numpy as np

if __name__=='__main__':
    values=[3.,11.,2.,1.,4.]
    correct=('MUL',('SUB',('MUL',('N',0),('N',1)),('N',4)),('N',2))
    wrong=('SUB',('MUL',('MUL',('N',0),('N',1)),('N',2)),('N',4))
    a,da=dual(correct,values);b,db=dual(wrong,values)
    assert a==58 and b==62;assert np.array_equal(da,[22,6,29,0,-2]);assert np.array_equal(db,[22,6,33,0,-1])
    programs=[correct,wrong,('DIV',('N',0),('N',2)),('DIV',('MUL',('N',0),('N',1)),('SUB',('N',4),('N',2)))]
    for p in programs:
        value,grad=dual(p,values)
        for i in range(len(values)):
            lo=values.copy();hi=values.copy();lo[i]-=1e-5;hi[i]+=1e-5
            difference=(dual(p,hi)[0]-dual(p,lo)[0])/2e-5
            assert np.isclose(grad[i],difference,rtol=1e-5,atol=1e-5)
    dump(ROOT/'experiments/round78/arithmetic_checks.json',dict(scope_example=dict(correct_answer=a,wrong_scope_answer=b,correct_gradient=da.tolist(),wrong_gradient=db.tolist()),finite_difference_checks=20,passed=True,source_hash=sha(__file__)))
    print('PASS: exact scope example and 20 finite-difference checks',flush=True)
