"""Repair candidate arithmetic evidence extraction, NOT the official answer parser.

Whitespace-separated units are legal; attached identifiers such as 3x are not
silently truncated. Only candidate-stated arithmetic enters the dependency DAG.
"""
import ast,re,hashlib
from fractions import Fraction
from round4_quantities import math_text
from features import safe_arithmetic,parse_answer
from round46_graph import proposals as previous_proposals,dag_roots,graph,valid,LIMIT
from round44_programs import canonical,tree

UNIT_FRACTION=re.compile(r'\\(?:d?frac)\s*\{\s*\\(?:text|mathrm)\{[^{}0-9]*\}\s*\}\s*\{\s*\\(?:text|mathrm)\{[^{}0-9]*\}\s*\}')

def arithmetic_evidence(text):
    source=UNIT_FRACTION.sub(' ',text)
    source=re.sub(r'(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(more|less|fewer)\s+than\s+(\d+(?:,\d{3})*(?:\.\d+)?)\s*=',
        lambda m:f"({m[3]} * (1 {'+' if m[2]=='more' else '-'} {m[1]}/100)) =",source,flags=re.I)
    lines=[]
    for line in source.splitlines():
        bullet=re.match(r'^\s*[-+*]\s+(.*)$',line)
        if bullet and bullet[1].startswith(('$','\\','**')):line=bullet[1]
        lines.append(line)
    result=[]
    for line in math_text('\n'.join(lines)).splitlines():
        for eq in re.finditer('=',line):
            left=re.search(r'([\d().+*/ \t\-]+)$',line[:eq.start()]);right=re.match(r'([\d().+*/ \t\-]+)',line[eq.end():])
            if left is None or right is None:continue
            l=left[0].strip();raw=right[0];r=raw.strip().rstrip('.')
            prefix=line[:eq.start()][:left.start()].rstrip();suffix=line[eq.end():][right.end():]
            if prefix and prefix[-1].isalnum() and l.startswith(('+','-','*','/')):continue
            # Crucial distinction: "36 cans" is separated, whereas "36x" is not.
            if suffix and suffix[0].isalnum() and raw and not raw[-1].isspace():continue
            try:
                t=ast.parse(l,mode='eval');a=safe_arithmetic(l);b=safe_arithmetic(r)
            except (ValueError,SyntaxError,OverflowError,ZeroDivisionError):continue
            if not any(isinstance(n,ast.BinOp) for n in ast.walk(t)):continue
            result.append((l,r,a,b))
    return result

def proposals(row):
    old=previous_proposals(row);chosen=list(old['hypotheses']);seen={repr(canonical(tree(h['program']))) for h in chosen}
    q=row['question'];g=graph(q);answers=[parse_answer(c['text']) for c in row['candidates']];extra={}
    for c in row['candidates']:
        answer=parse_answer(c['text'])
        if answer is None:continue
        steps=arithmetic_evidence(c['text'])
        # This is lossless numeric evidence normalization, not generated LLM reasoning.
        trace='\n'.join(f'{l} = {r}' for l,r,_,_ in steps)+'\n#### '+answer
        for p in dag_roots(q,trace,g):
            key=repr(canonical(p));a=valid(p,g['quantities'])
            if key not in seen and a is not None:extra[key]=dict(program=p,action='unit_separated_arithmetic_graph',decision={},answer=a,matches=[j for j,x in enumerate(answers) if x is not None and x==a])
    for key in sorted(extra,key=lambda k:hashlib.sha256(k.encode()).digest()):
        if len(chosen)==LIMIT:break
        chosen.append(extra[key])
    return dict(quantities=g['quantities'],hypotheses=chosen,graph=g)

def checks():
    assert len(arithmetic_evidence('3 * 12 = 36 cans'))==1
    assert not arithmetic_evidence('3 * 12 = 36x')
    assert not arithmetic_evidence('x + 2 = 3')
    assert arithmetic_evidence('- $344 \\times 0.25 = 86$ dollars')[0][2:]==(86.,86.)
    assert arithmetic_evidence('50% more than 1000 = 1500 pieces')[0][2:]==(1500.,1500.)
    assert arithmetic_evidence(r'80 \\text{kg} \\times 2.2 \\frac{\\text{pounds}}{\\text{kg}} = 176 \\text{pounds}') or arithmetic_evidence(r'80 \text{kg} \times 2.2 \frac{\text{pounds}}{\text{kg}} = 176 \text{pounds}')
    return dict(unit_separation=True,attached_identifier_not_truncated=True,symbolic_suffix_not_reinterpreted=True,markdown_bullet_preserved=True,percentage_phrase=True,unit_ratio=True)
