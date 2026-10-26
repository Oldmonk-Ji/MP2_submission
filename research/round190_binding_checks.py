from round22_common import ROOT,dump,sha
from round186_trace import extract
from round44_programs import locations
from round29_programs import quantities
if __name__=='__main__':
    q='Ann has 12 apples and 4 boxes of 3 pencils and loses 2 pencils.'
    p,_=extract(q,'4 * 3 = 12\n12 - 2 = 10\n#### 10');assert p and all(not any(n==('N',0) for _,n in locations(t)) for t in p)
    q='A shop reduces prices by 20 percent. Ann buys 5 shirts at 60 dollars.'
    p,_=extract(q,'5 * 60 = 300\n#### 300');assert p and all(not any(n==('N',0) for _,n in locations(t)) for t in p)
    q='Once the water reaches 10 degrees, add 2 degrees.';qs=quantities(q);slot=next(i for i,n in enumerate(qs) if n['raw'].lower()=='once')
    p,_=extract(q,'1 * 10 = 10\n10 + 2 = 12\n#### 12');assert p and all(not any(n==('N',slot) for _,n in locations(t)) for t in p)
    report=dict(verified_intermediate_precedence=True,direct_quantity_before_fraction_alias=True,temporal_once_not_quantity=True,heuristic_not_semantic_proof=True,test_opened=False,source_sha256=sha(__file__));dump(ROOT/'experiments/round190/binding_checks.json',report);print(report)
