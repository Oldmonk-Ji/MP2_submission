"""Arithmetic repair and explicit unit equalities, never reference-answer repair."""
import re
import numpy as np
from round4_quantities import trace

UNITS={'second':('time',1),'minute':('time',60),'hour':('time',3600),'day':('time',86400),
       'week':('time',604800),'half-hour':('time',1800),'ounce':('mass',1),'pound':('mass',16),
       'inch':('length',1),'foot':('length',12),'feet':('length',12),'yard':('length',36),
       'cent':('money',1),'dollar':('money',100)}
UNIT_PATTERN='(?:'+'|'.join(sorted(UNITS,key=len,reverse=True))+')s?'


def unit_checks(text):
    s=re.sub(r'[*$\\]','',text.lower());good=bad=0
    pattern=r'(\d+(?:\.\d+)?)\s*('+UNIT_PATTERN+r')\s*=\s*(\d+(?:\.\d+)?)\s*('+UNIT_PATTERN+r')\b'
    for a,u,b,v in re.findall(pattern,s):
        u=u if u in UNITS else u[:-1];v=v if v in UNITS else v[:-1]
        if UNITS[u][0]!=UNITS[v][0]:continue
        match=abs(float(a)*UNITS[u][1]-float(b)*UNITS[v][1])<1e-6
        good+=match;bad+=not match
    return good,bad


def program_value(program,quantities):
    seen={};calls=0
    def f(p):
        nonlocal calls
        if id(p) in seen:return seen[id(p)]
        calls+=1
        if calls>500:raise ValueError('oversize dependency graph')
        op=p[0]
        if op=='q':v=quantities[p[1]]['value']
        elif op=='c':v=p[1]
        else:
            a,b=f(p[1]),f(p[2])
            if op=='+':v=a+b
            elif op=='-':v=a-b
            elif op=='*':v=a*b
            elif op=='/' and abs(b)>1e-12:v=a/b
            else:raise ValueError('invalid operation')
        if not np.isfinite(v) or abs(v)>1e15:raise ValueError('unbounded value')
        seen[id(p)]=v;return v
    return f(program)


def close(a,b):return a is not None and b is not None and abs(a-b)<=1e-4*max(1,abs(a),abs(b))


def evidence(bank):
    result=[]
    for row in bank:
        # Keep the initial residual ablation's feature version reproducible.
        # Final explicit gates use strict_math=True through the trace default.
        traces=[trace(row['question'],c['text'],repair_subexpressions=True,strict_math=False) for c in row['candidates']]
        corrected=[]
        for t in traces:
            try:v=program_value(t['program'],t['quantities']) if t['root_found'] else None
            except (ValueError,OverflowError,TypeError):v=None
            corrected.append(v)
        for j,(t,c) in enumerate(zip(traces,row['candidates'])):
            good,bad=unit_checks(c['text']);v=corrected[j]
            repaired=v is not None and not close(v,t['answer'])
            supports=sum(close(vv,t['answer']) and other['collisions']==0 for vv,other in zip(corrected,traces))
            repair_support=sum(close(vv,t['answer']) and other['collisions']==0 and not close(vv,other['answer']) for vv,other in zip(corrected,traces))
            compatible=sum(close(v,other['answer']) for other in traces)
            result.append([float(t['root_found']),float(repaired),float(repaired and t['collisions']==0),
                float(repaired and compatible>0 and t['collisions']==0),supports/4,repair_support/4,
                np.log1p(good),np.log1p(bad),t['boxed_parse_disagreement'],
                np.log1p(t['bad']),np.log1p(t['collisions']),np.log1p(t['unexplained'])])
    return np.asarray(result,np.float32).reshape(len(bank),4,12)
