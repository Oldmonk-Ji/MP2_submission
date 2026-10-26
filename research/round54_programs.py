"""Fit-only follow-up: Markdown, solved-variable use and omitted final operation.

No held-error text was consulted. Completion candidates are numeric hypotheses,
not semantic gold, and use only already executed/stated candidate intermediates.
"""
import ast,re,hashlib
from fractions import Fraction as F
from itertools import combinations
import round51_programs as old
from round51_programs import value,graph,canonical,tree,LIMIT,valid,constants
from features import parse_answer

def normalized(text):
    # Remove Markdown bold before arithmetic '*' normalization, preserving powers.
    text=re.sub(r'(?<![\d)])\*\*|\*\*(?![\d(])','',text)
    s=old_normalized(text)
    s=re.sub(r'(?<=\d)\s*\((?=\d)', '*(',s)
    # Embedded slash units, e.g. 10 pictures/day * 365 days/year.
    s=re.sub(r'(?<=\d)\s+[A-Za-z][A-Za-z /-]{1,45}?(?=\s*[+*/=])',' ',s)
    return s

old_normalized=old.normalized

def numeric(n,env):
    if isinstance(n,ast.Constant) and type(n.value) in (int,float):return F(str(n.value))
    if isinstance(n,ast.Name):return env[n.id]
    if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -numeric(n.operand,env)
    if isinstance(n,ast.BinOp):
        a,b=numeric(n.left,env),numeric(n.right,env)
        if isinstance(n.op,ast.Add):return a+b
        if isinstance(n.op,ast.Sub):return a-b
        if isinstance(n.op,ast.Mult):return a*b
        if isinstance(n.op,ast.Div):return a/b
        if isinstance(n.op,ast.Pow) and b.denominator==1 and 0<=b<=4:return a**int(b)
    raise ValueError('unsupported')

def candidate_roots(q,text,g):
    oldnorm=old.normalized
    try:
        old.normalized=normalized
        # Append numeric substitutions immediately after source equations, so the
        # existing source binder still has to justify every numeric operand.
        extended=[];known={};recent=[]
        for line in text.splitlines():
            extended.append(line)
            for a,b,names in old.equations(line):
                try:
                    if names:
                        name=next(iter(names))
                        if name in known:
                            la=ast.unparse(a);rb=ast.unparse(b)
                            la=re.sub(r'\b'+re.escape(name)+r'\b',f'({known[name]})',la)
                            rb=re.sub(r'\b'+re.escape(name)+r'\b',f'({known[name]})',rb)
                            extended.append(la+' = '+rb)
                        # Only numerically linear equations establish a binding.
                        def delta(t,v):return numeric(a,{name:F(v)})-numeric(b,{name:F(v)})
                        d0,d1,d2=delta(None,0),delta(None,1),delta(None,2)
                        if d1!=d0 and d2-d1==d1-d0:known[name]=-d0/(d1-d0)
                    else:
                        av,bv=numeric(a,{}),numeric(b,{})
                        if av==bv and abs(av)<=10**12:recent.append(av)
                except (ValueError,KeyError,ZeroDivisionError,OverflowError):continue
        answer=parse_answer(text)
        if answer is None:return []
        target=q.rstrip('? .').split('.')[-1].lower();endings=[];vals=list(dict.fromkeys(recent[-12:]))
        for a,b in combinations(vals,2):
            if ('total' in target or 'how many' in target or 'how much' in target) and a+b==F(answer):endings.append(f'({a}) + ({b}) = {answer}')
            if re.search(r'more|less|left|difference|remaining',target):
                if a-b==F(answer):endings.append(f'({a}) - ({b}) = {answer}')
                if b-a==F(answer):endings.append(f'({b}) - ({a}) = {answer}')
        for v in vals:
            if re.search(r'cent|dollar',q,re.I):
                if v*100==F(answer):endings.append(f'({v}) * 100 = {answer}')
                if v/100==F(answer):endings.append(f'({v}) / 100 = {answer}')
        # Count-valued questions may require whole items; these are hypotheses.
        if re.search(r'how many.*(?:bill|deer|can|box|card|bus|trip|slice)',q,re.I):
            extended.insert(0,'Whole items may require rounding.')
        trace='\n'.join(extended+endings)+'\n#### '+answer
        return old.candidate_roots(q,trace,g)
    finally:old.normalized=oldnorm

def proposals(row):
    prior=old.proposals(row);g=prior['graph'];answers=[parse_answer(c['text']) for c in row['candidates']];new={}
    for c in row['candidates']:
        for p,kind in candidate_roots(row['question'],c['text'],g):
            p=canonical(p);v=valid(p,g['quantities'],constants(row['question']))
            if v is not None:new[repr(p)]=dict(program=p,action='trace_completion_'+kind,decision={},answer=v,matches=[j for j,a in enumerate(answers) if a==v])
    # Keep the same64 budget. Give newly source-connected candidate roots a slot
    # before generic old edits; no reward or candidate position enters order.
    oldkeys={repr(canonical(tree(h['program']))):h for h in prior['hypotheses']}
    chosen=[];seen=set()
    for key in sorted(new,key=lambda k:hashlib.sha256(k.encode()).digest()):
        if len(chosen)>=16:break
        chosen.append(new[key]);seen.add(key)
    for h in prior['hypotheses']:
        key=repr(canonical(tree(h['program'])))
        if key not in seen:chosen.append(h);seen.add(key)
        if len(chosen)==LIMIT:break
    return dict(prior,hypotheses=chosen)

def checks():
    q='A trip takes 20 minutes running and 10 minutes jogging. Return takes 3 times as long. How long to return?'
    g=graph(q);r=candidate_roots(q,'**20 + 10 = 30 minutes**\n**3 * 30 = 90 minutes**\n#### 90',g)
    assert any(value(p,g['quantities'])==90 for p,k in r)
    q='400 adults and 200 children. Adult tickets cost twice child tickets. Total 16000 dollars. Adult ticket price?'
    g=graph(q);r=candidate_roots(q,'400 * 2*x + 200*x = 16000\nx = 16\n2*x = 32\n#### 32',g)
    assert any(value(p,g['quantities'])==32 for p,k in r)
    return dict(bold_math=True,solved_variable_use=True)
