from data_access import ROOT
import json
import numpy as np
from round5_graph import extract,candidate,NAMES
from round5_rate_solver import solve
from round5_age_graph import solve as age_solve

def main():
    # Small code fixtures only; never added to training or evaluation data.
    q='Ann has 2 trays and works 2 days. How many trays does Ann have?'
    g=extract(q);assert len(g['nodes'])==2
    assert g['nodes'][0]['span']!=g['nodes'][1]['span']
    assert g['nodes'][0]['unit']=='tray' and g['nodes'][1]['unit']=='day'
    f,a=candidate(g,q,'2 = 2\n#### 2');assert f[NAMES.index('ambiguous_bindings')]==1
    assert not a['bindings']  # no arbitrary first-occurrence assignment
    q='Tom rents a machine for 2 hours a day for 3 days. It costs $75 an hour. How much did he pay?'
    p=solve(q);assert p and p['value']==450
    g=extract(q);assert [n['per'] for n in g['nodes']]==['day',None,'hour']
    f,a=candidate(g,q,'2 * 3 = 6\n6 * 75 = 450\n#### 450')
    assert len(f)==len(NAMES) and np.isfinite(f).all()
    assert solve('Ann has 3 apples and buys 4 apples. How many apples does she have?') is None
    # Unstated constants, ratios, temporal changes cannot pass the rate solver.
    assert solve('Ann earns twice $20 per hour for 3 hours. How much does Ann earn?') is None
    assert abs(age_solve('In 5 years Matt will be twice James age. James is 30 years old now. How old is Matt now?')['value']-65)<1e-7
    assert abs(age_solve("Jacob is 24 years now. Tony is half Jacob's age. In 6 years how old will Tony be?")['value']-18)<1e-7
    assert abs(age_solve('Herbert is 10 years younger than Kris. Kris is 24 years old now. How old will Herbert be next year?')['value']-15)<1e-7
    assert age_solve('Ann is twice as old as Beth. How old is Ann?') is None
    result=dict(passed=True,checks=['occurrence preservation','ambiguous binding abstention','explicit three-edge rate closure','rate denominators','finite feature dimensions','unsupported addition abstention','ratio abstention'],test_access=False)
    result['checks']+=['shared future age offset','decimal ratio not sentence boundary','next year target shift','underdetermined age graph abstention']
    (ROOT/'experiments/round5/checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))
if __name__=='__main__':main()
