"""Exact source-occurrence influence; unknown source bindings stay unlabeled."""
from fractions import Fraction as F
from collections import Counter
from round29_programs import quantities
from round44_programs import locations

def dual(p,values):
    if p[0]=='N':return values[p[1]],{p[1]:F(1)}
    if p[0]=='L':return F(p[1]),{}
    a,da=dual(p[1],values);b,db=dual(p[2],values);keys=da.keys()|db.keys()
    if p[0]=='ADD':return a+b,{i:da.get(i,0)+db.get(i,0) for i in keys}
    if p[0]=='SUB':return a-b,{i:da.get(i,0)-db.get(i,0) for i in keys}
    if p[0]=='MUL':return a*b,{i:b*da.get(i,0)+a*db.get(i,0) for i in keys}
    if p[0]=='DIV':return a/b,{i:(b*da.get(i,0)-a*db.get(i,0))/(b*b) for i in keys}
    raise ValueError('Unsupported derivative')

def labels(q,p,reviewed=False):
    qs=quantities(q);values=[F(x['value']) for x in qs];_,deriv=dual(p,values)
    used={n[1] for _,n in locations(p) if n[0]=='N'};literals={F(n[1]) for _,n in locations(p) if n[0]=='L'};counts=Counter(values);result=[]
    for i,v in enumerate(values):
        # A missing source leaf can be a numeric literal with unresolved binding.
        if not reviewed and i not in used and (counts[v]>1 or v in literals or v in (0,1)):continue
        d=deriv.get(i,F(0));result.append(dict(quantity=i,label=0 if d==0 else 1 if d>0 else 2,derivative=str(d),duplicate_value=counts[v]>1,old_excluded=counts[v]>1 or v in (0,1),used=i in used))
    return result

def checks():
    q='There are 2 boxes with 2 balls each.';p=('MUL',('N',0),('N',1));ls=labels(q,p)
    assert len(ls)==2 and all(r['label']==1 for r in ls)
    q='Ann has 5 dollars and gives 5 dollars.';p=('SUB',('N',0),('N',1));ls=labels(q,p)
    assert [r['label'] for r in ls]==[1,2]
    assert labels(q,('ADD',('L','5'),('L','1')))==[]
    v,d=dual(('DIV',('N',0),('ADD',('N',0),('N',1))),[F(3),F(2)])
    assert v==F(3,5) and d=={0:F(2,25),1:F(-3,25)}
    return dict(equal_numbers_remain_distinct=True,opposite_effects_preserved=True,unresolved_unused_not_labeled=True,exact_reused_source_derivative=True)
