from data_access import ROOT
import json
from round6_bindings import extract,solve_time,solve_ledger

def main():
    checks=[]
    a=extract('Ann buys 3 pens for $2 each. She paid with a $10 bill. How much change will she get?')
    assert solve_ledger(a)['value']==4;checks.append('unit price multiplied by quantity')
    b=extract('Ann buys 3 shirts for $60. There is a 40% off sale. How much did she pay per shirt after the discount?')
    assert solve_ledger(b)['value']==12;checks.append('total price not mistaken for unit price')
    c=extract('Ann has 4 notebooks. She ordered 6 more notebooks and then lost 2. How many notebooks does Ann have now?')
    assert solve_ledger(c)['value']==8;checks.append('initial plus signed changes')
    for suffix,expected in [('',28),(' today',20)]:
        d=extract(f'In 4 years Ann will be twice Bob age{suffix}. Bob is 12 years old. How old is Ann?')
        assert abs(solve_time(d)['value']-expected)<1e-7
    checks.append('shared future versus current RHS time')
    e=extract('Ann buys 3 pens for $2 each. Bob buys 4 books for $5 each. How much change will Ann get?')
    assert solve_ledger(e) is None;checks.append('multiple owners abstain')
    e=extract('Ann buys 3 pens for $2 each. She paid with a $10 bill and bought $1 candy. How much change will she get?')
    assert solve_ledger(e) is None;checks.append('unbound quantity prevents partial ledger answer')
    e=extract('Ann buys 3 books for $10 per book. How much change will she get?')
    assert not any(f['kind']=='total_price' for f in e['facts']);checks.append('unsupported per-unit tail never interpreted as total')
    out=dict(passed=True,checks=checks,fixtures='Code fixtures only; never used as training or independent audit examples',test_access=False)
    (ROOT/'experiments/round6/checks.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
