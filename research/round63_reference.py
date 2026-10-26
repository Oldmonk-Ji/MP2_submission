"""TRAIN-reference-only strict trace alignment; no guessed omitted operations."""
import ast,re
from fractions import Fraction as F
from round29_programs import quantities,canonical,value
from round44_programs import locations
OPS={ast.Add:'ADD',ast.Sub:'SUB',ast.Mult:'MUL',ast.Div:'DIV'}
CONSTANTS={F(v) for v in (0,1,2,3,4,5,7,10,12,24,60,100,1000)}

def align(question,reference):
    qs=quantities(question);mapping={};cache={}
    for i,q in enumerate(qs):mapping.setdefault(F(q['value']),[]).append(('N',i))
    steps=[]
    def visit(n):
        if isinstance(n,ast.Constant) and type(n.value) in (int,float):
            v=F(str(n.value));choices={repr(p):p for p in mapping.get(v,[])+cache.get(v,[])}
            if len(choices)==1:return next(iter(choices.values()))
            if not choices and v in CONSTANTS:return ('L',str(v))
            raise ValueError('ambiguous or ungrounded leaf')
        if isinstance(n,ast.BinOp) and type(n.op) in OPS:return canonical((OPS[type(n.op)],visit(n.left),visit(n.right)))
        raise ValueError('unsupported expression')
    for m in re.finditer(r'<<([^<>]+)>>',reference):
        raw=m[1]
        if raw.count('=')!=1:continue
        expr,result=raw.split('=')
        try:
            p=visit(ast.parse(expr.replace(',','').replace('×','*'),mode='eval').body)
            target=F(result.replace(',','').strip())
            if value(p,qs)!=target or p[0] in ('N','L') or len(list(locations(p)))>31:continue
            cache.setdefault(target,[]).append(p)
            steps.append(dict(program=p,evidence=m.group(),reference_span=list(m.span()),value=str(target)))
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,IndexError):continue
    return qs,steps

if __name__=='__main__':
    from round22_common import ROOT,CODE,load,dump,sha
    import json,numpy as np
    out=ROOT/'experiments/round63';out.mkdir(exist_ok=False)
    from common import read_jsonl
    b,_,_,ids=load('train');refs=read_jsonl(CODE/'data/gsm8k_train.jsonl')
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    # Freeze algorithm before opening diagnostic fold references in this run.
    dump(out/'alignment_freeze.json',dict(source_sha256=sha(__file__),reference_trace_not_independent_human_gold=True,unknown_policy='reject ambiguous quantities/intermediates and missing conversions',test_opened=False))
    rows=[]
    for row,idx,fold in zip(b,ids,f):
        ref=refs[int(idx)];assert row['question']==ref['question']
        qs,steps=align(row['question'],ref['answer'])
        if steps:rows.append(dict(train_index=int(idx),fold=int(fold),question=row['question'],quantities=qs,steps=steps))
    dump(out/'aligned_reference_steps.json',rows)
    stats={}
    for name,flag in [('fit',False),('held',True)]:
        part=[r for r in rows if (r['fold']==2)==flag];stats[name]=dict(parents=len(part),steps=sum(len(r['steps']) for r in part))
    dump(out/'alignment_stats.json',stats);print(stats,flush=True)
