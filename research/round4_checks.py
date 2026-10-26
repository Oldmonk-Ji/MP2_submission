"""Targeted tests for quantity derivation, strict repair and unchanged benchmark."""
from data_access import ROOT,CODE,forbid_test,load
import json
import numpy as np
from round4_quantities import trace,quantities,arithmetic_steps
from round4_evidence import program_value,unit_checks,evidence
from round4_pairs import pairs,PairModel
from round4_semantics import QuantityModel


def main():
    assert quantities('A distance of .6 miles and 1 1/2 hours.')[0]['value']==.6
    assert quantities('A distance of .6 miles and 1 1/2 hours.')[1]['value']==1.5
    for word,n in [('two',2),('twice',2),('half',.5),('dozen',12)]:assert quantities(word+' items')[0]['value']==n
    for sign,want in [('+',[1,1]),('-',[1,-1])]:
        t=trace('Alex has 10 books and buys 4 books.',f'10{sign}4={14 if sign=="+" else 6}\n#### {14 if sign=="+" else 6}')
        assert np.allclose(t['gradient'],want)
    q='A dog runs 24 miles per hour and a rabbit 15, with a .6 mile gap. How many minutes?'
    t=trace(q,'24-15=9\n0.6/9=2/15\n2/15*60=8\n#### 8',repair_subexpressions=True)
    assert abs(program_value(t['program'],t['quantities'])-4)<1e-8
    t=trace('A worker gets 20 per hour and works 5 hours.','- $20 * 5 = 100$\n#### 100',repair_subexpressions=True)
    assert t['bad']==0 and program_value(t['program'],t['quantities'])==100
    assert not arithmetic_steps('x + 2 = 5',strict=True)
    assert unit_checks('1 hour = 3 half-hours')==(0,1)
    assert unit_checks('8 ounces = 1 pound')==(0,1)
    assert unit_checks('60 minutes = 1 hour')==(1,0)
    assert pairs(('/',('q',1),('q',0)))=={(0,1):5}
    assert pairs(('*',('q',0),('q',1)))=={(0,1):3}
    # Exercise the guard directly, without issuing any real test file open.
    try:forbid_test('open',(str(CODE/'data/gsm8k_test.jsonl'),))
    except PermissionError:pass
    else:raise AssertionError('test guard')
    b,_,_=load('train');b=b[:7];p=[2,0,3,1];bp=[dict(r,candidates=[r['candidates'][i] for i in p]) for r in b]
    transforms=[evidence,QuantityModel(ROOT/'experiments/round4/quantity_model').features,
                PairModel(ROOT/'experiments/round4/pair_model').features]
    for f in transforms:
        x=f(b);xp=f(bp);assert np.isfinite(x).all() and np.allclose(xp,x[:,p],atol=1e-5)
    result=dict(status='passed',checks=['leading decimals and mixed fractions','written numbers','derivative direction',
        'fractional intermediate propagation','Markdown marker is not unary minus','symbolic boundary',
        'unit equality','directed operation pair labels','test guard without file open','finite features and candidate permutation'])
    (ROOT/'experiments/round4/checks.json').write_text(json.dumps(result,indent=2));print(result)


if __name__=='__main__':main()
