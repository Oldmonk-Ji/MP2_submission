"""Occurrence-preserving arithmetic program supervision and candidate inputs.

No semantic binding is inferred from equal numeric values when more than one
source is possible. Such leaves stay literal and ambiguity is reported. A
candidate's claimed intermediate values are preserved; arithmetic repair never
silently changes its final answer. Reference answers are training-only callers.
"""
import ast,re
from fractions import Fraction
from collections import Counter
import numpy as np
from round4_quantities import TOKEN,WORDS,arithmetic_steps
from features import parse_answer

OPS={ast.Add:'ADD',ast.Sub:'SUB',ast.Mult:'MUL',ast.Div:'DIV'}
SYMBOLS=['PAD','BOS','EOS','UNK','ADD','SUB','MUL','DIV','LIT','END','NEG','DOT','SLASH']+[str(i) for i in range(10)]+[f'N{i}' for i in range(24)]
SYMBOL_INDEX={s:i for i,s in enumerate(SYMBOLS)}


def quantities(text):
    out=[]
    for m in TOKEN.finditer(text):
        raw=m.group().lower()
        try:v=Fraction(WORDS[raw]) if raw in WORDS else sum((Fraction(p) for p in raw.replace(',','').split()),Fraction(0))
        except (ValueError,ZeroDivisionError):continue
        if abs(v)>10**10:continue
        out.append(dict(value=str(v),start=m.start(),end=m.end(),raw=m.group()))
        if len(out)==24:break
    return out


def canonical(p):
    if p[0] in ('N','L'):return p
    a,b=canonical(p[1]),canonical(p[2])
    if p[0] in ('ADD','MUL') and repr(a)>repr(b):a,b=b,a
    return (p[0],a,b)


def value(p,qs):
    if p[0]=='N':return Fraction(qs[p[1]]['value'])
    if p[0]=='L':return Fraction(p[1])
    a,b=value(p[1],qs),value(p[2],qs)
    if p[0]=='ADD':v=a+b
    elif p[0]=='SUB':v=a-b
    elif p[0]=='MUL':v=a*b
    else:v=a/b
    if abs(v)>10**15:raise ValueError('unbounded')
    return v


def serialize(p):
    if p[0]=='N':return [f'N{p[1]}']
    if p[0]=='L':
        return ['LIT']+[{'-':'NEG','.':'DOT','/':'SLASH'}.get(c,c) for c in str(Fraction(p[1]))]+['END']
    return [p[0]]+serialize(p[1])+serialize(p[2])


def parse(question,text,reference=False):
    qs=quantities(question);direct={}
    for i,q in enumerate(qs):direct.setdefault(Fraction(q['value']),[]).append(i)
    cache={};roots=[];ambiguous=bad=0
    # Include unannotated numeric steps as well as <<...>> annotations. Several
    # official references omit an intermediate from their annotations.
    if reference:
        steps=[]
        for line in text.split('\n'):
            annotated=arithmetic_steps(line,reference=True,strict=True)
            # Prefer explicit annotations on that line; replacing them inline
            # creates spurious equalities such as a+b=a+b=result.
            steps.extend(annotated or arithmetic_steps(re.sub(r'<<[^<>]*>>','',line),reference=False,strict=True))
    else:steps=arithmetic_steps(text,reference=False,strict=True)
    def visit(node,budget):
        nonlocal ambiguous
        budget[0]+=1
        if budget[0]>80:raise ValueError('large')
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):
            v=Fraction(str(node.value));options=direct.get(v,[])
            # A computed result and an original mention with the same number
            # are competing bindings. Keep uncertainty instead of guessing.
            if v in cache and not options and len(cache[v])==1:return next(iter(cache[v].values()))
            if len(options)==1 and v not in cache:return ('N',options[0])
            if len(options)>1 or v in cache:ambiguous+=1
            return ('L',str(v))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):
            p=visit(node.operand,budget)
            return ('SUB',('L','0'),p) if isinstance(node.op,ast.USub) else p
        if isinstance(node,ast.BinOp) and type(node.op) in OPS:
            return (OPS[type(node.op)],visit(node.left,budget),visit(node.right,budget))
        raise ValueError('symbolic')
    for left,right,a,b in steps:
        try:
            p=canonical(visit(ast.parse(left,mode='eval').body,[0]));pv=value(p,qs)
            if len(serialize(p))>90:continue
            rv=Fraction(str(b));bad+=abs(float(pv-rv))>1e-5*max(1,abs(float(rv)))
            cache.setdefault(rv,{})[repr(p)]=p;roots.append((rv,p))
        except (ValueError,ZeroDivisionError,SyntaxError,OverflowError,RecursionError):continue
    ans=parse_answer(text);answer=Fraction(ans) if ans is not None else None
    found=[p for v,p in roots if answer is not None and abs(float(v-answer))<1e-7*max(1,abs(float(answer)))]
    program=found[-1] if found else (('L',str(answer)) if answer is not None else ('L','0'))
    seq=serialize(program)
    try:consistent=answer is not None and abs(float(value(program,qs)-answer))<1e-6*max(1,abs(float(answer)))
    except (ValueError,ZeroDivisionError,OverflowError):consistent=False
    used={int(t[1:]) for t in seq if re.fullmatch(r'N\d+',t)}
    source_values={Fraction(q['value']) for q in qs}
    literals=[]
    def collect(p):
        if p[0]=='L':literals.append(Fraction(p[1]))
        elif p[0]!='N':collect(p[1]);collect(p[2])
    collect(program)
    unresolved=sum(v not in source_values and v not in (0,1,2,3,4,5,7,10,12,24,60,100,1000,Fraction(1,2),Fraction(1,4),Fraction(1,100)) for v in literals)
    return dict(quantities=qs,program=program,tokens=seq+['EOS'],root_found=bool(found),
        arithmetic_consistent=consistent,ambiguous_leaves=ambiguous,bad_steps=int(bad),
        source_quantities_used=len(used),steps=len(steps),answer=str(answer) if answer is not None else None,
        unresolved_literals=unresolved,informative=bool(found) and bool(used) and consistent,
        clean=bool(found) and bool(used) and consistent and ambiguous==0 and unresolved==0)


def question_tokens(question,qs=None):
    qs=quantities(question) if qs is None else qs
    tokenize=lambda s:re.findall(r"[a-z]+(?:'[a-z]+)?|[?.!,;%$]",s.lower())
    parts=[];last=0
    for i,q in enumerate(qs):
        parts+=tokenize(question[last:q['start']])+[f'numslot{i}']+tokenize(q['raw'])
        parts+=['digit'+c if c.isdigit() else 'fractionbar' for c in q['value']]
        last=q['end']
    parts+=tokenize(question[last:]);return parts


def vocabulary(bank,limit=4500):
    counts=Counter(t for r in bank for t in question_tokens(r['question']))
    fixed=['<pad>','<unk>']+[f'numslot{i}' for i in range(24)]
    words=sorted((t for t in counts if t not in fixed),key=lambda t:(-counts[t],t))[:limit-len(fixed)]
    return {w:i for i,w in enumerate(fixed+words)}


def encode_questions(bank,vocab,max_length=160):
    out=np.zeros((len(bank),max_length),np.int64)
    for i,row in enumerate(bank):
        ts=question_tokens(row['question'])
        if len(ts)>max_length:ts=ts[:max_length//2]+ts[-max_length//2:]
        ids=[vocab.get(t,1) for t in ts];out[i,:len(ids)]=ids
    return out


def encode_programs(rows,max_length=96):
    out=np.zeros((len(rows),max_length),np.int64)
    for i,r in enumerate(rows):
        ts=r['tokens'];assert len(ts)<=max_length
        out[i,:len(ts)]=[SYMBOL_INDEX[t] for t in ts]
    return out


def evidence(rows):
    return np.asarray([[r['root_found'],r['arithmetic_consistent'],np.log1p(r['ambiguous_leaves']),
        np.log1p(r['bad_steps']),np.log1p(r['source_quantities_used']),np.log1p(len(r['tokens'])),np.log1p(r['steps'])] for r in rows],np.float32)
