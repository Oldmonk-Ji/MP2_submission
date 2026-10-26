"""Input-only selector features. No external models, data, IDs, or answers."""
import ast
from collections import Counter
from fractions import Fraction
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DASE7506-MP2-main' / 'code'))
sys.path.insert(0, str(ROOT / 'starter'))
from common import parse_answer
from method import text_features

NUMBER = re.compile(r'(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?')


def safe_arithmetic(expression):
    """Evaluate bounded numeric arithmetic only; never execute candidate text."""
    if len(expression) > 100:
        raise ValueError('long expression')
    tree = ast.parse(expression.strip(), mode='eval')
    nodes = list(ast.walk(tree))
    if len(nodes) > 50:
        raise ValueError('complex expression')
    def visit(n):
        if isinstance(n, ast.Expression):
            return visit(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = visit(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, ast.Add): return a+b
            if isinstance(n.op, ast.Sub): return a-b
            if isinstance(n.op, ast.Mult): return a*b
            if abs(b) < 1e-12: raise ValueError('division by zero')
            return a/b
        raise ValueError('unsupported expression')
    value = visit(tree)
    if not np.isfinite(value) or abs(value) > 1e15:
        raise ValueError('unbounded result')
    return value


def arithmetic_features(text):
    s = text.replace('\\times', '*').replace('\\cdot', '*').replace('×', '*')
    s = s.replace('\\div', '/').replace('÷', '/').replace('−', '-')
    # Simple numeric LaTeX fractions; unsupported symbolic equations are skipped.
    for _ in range(3):
        s = re.sub(r'\\(?:d?frac)\{([\d.\s+*/()\-]+)\}\{([\d.\s+*/()\-]+)\}', r'((\1)/(\2))', s)
    s = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
    matches = re.findall(r'(?<![\w.])([\d().+*/ \-]{1,100})\s*=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)', s)
    good = bad = rounded = 0
    for left, right in matches:
        if not any(op in left for op in '+*/-'):
            continue
        try:
            a, b = safe_arithmetic(left), safe_arithmetic(right)
        except (ValueError, SyntaxError, OverflowError, ZeroDivisionError):
            continue
        diff = abs(a-b)
        if diff <= 1e-6 * max(1, abs(b)):
            good += 1
        elif diff <= .02:
            rounded += 1
        else:
            bad += 1
    return [np.log1p(good), np.log1p(bad), np.log1p(rounded),
            good / max(good+bad+rounded, 1), float(bad > 0)]


def numeric_features(bank):
    rows = []
    for row in bank:
        qnums = [float(s.replace(',', '')) for s in NUMBER.findall(row['question'])]
        qset = set(qnums)
        scale = max([abs(x) for x in qnums] + [1.0])
        for candidate in row['candidates']:
            text = candidate['text']
            answer = parse_answer(text)
            f = Fraction(answer) if answer is not None else Fraction(0)
            value = float(f)
            nums = [float(s.replace(',', '')) for s in NUMBER.findall(text)]
            nset = set(nums)
            rows.append(arithmetic_features(text) + [
                float(f.denominator == 1), float(value < 0), float(value == 0),
                np.log1p(abs(value)), np.log1p(f.denominator),
                np.log1p(abs(value)/scale),
                sum(abs(n-value) < 1e-8 for n in nums)/max(len(nums), 1),
                len(nset & qset)/max(len(qset), 1),
                len(nset-qset)/max(len(nset), 1),
                np.log1p(len(qnums)), np.log1p(len(nset)),
            ])
    return np.asarray(rows, dtype=np.float32)


def checked_arithmetic(text):
    """V2: compare both sides of adjacent numeric equalities, including fractions.

    Kept separate from V1 so existing checkpoints retain their exact features.
    Symbolic expressions or units form boundaries; this is not a semantic proof.
    """
    s=text.replace('\\times','*').replace('\\cdot','*').replace('×','*')
    s=s.replace('\\div','/').replace('÷','/').replace('−','-')
    s=re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))','',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'\1',s)
    s=s.replace('\\left','').replace('\\right','')
    for _ in range(4):
        s=re.sub(r'\\(?:d?frac)\{([\d.\s+*/()\-]+)\}\{([\d.\s+*/()\-]+)\}',r'((\1)/(\2))',s)
        s=re.sub(r'\\boxed\{([\d.\s+*/()\-]+)\}',r'(\1)',s)
    s=s.replace('\\,','').replace('\\!','').replace('\\$','').replace('$','')
    good=bad=rounding=0;errors=[]
    for line in s.splitlines():
        for match in re.finditer('=',line):
            left=re.search(r'([\d().+*/ \t\-]+)$',line[:match.start()])
            right=re.match(r'([\d().+*/ \t\-]+)',line[match.end():])
            if left is None or right is None:continue
            l,r=left.group(1).strip(),right.group(1).strip()
            if not any(c in l+r for c in '+-*/'):continue
            try:a,b=safe_arithmetic(l),safe_arithmetic(r)
            except (ValueError,SyntaxError,OverflowError,ZeroDivisionError):continue
            diff=abs(a-b);relative=diff/max(1,abs(a),abs(b));errors.append(relative)
            if relative<=1e-6:good+=1
            elif diff<=.02 or relative<=.001:rounding+=1
            else:bad+=1
    return [np.log1p(good),np.log1p(bad),np.log1p(rounding),good/max(1,good+bad+rounding),
            float(bad>0),max(errors,default=0)]


def prepare(bank, arrays):
    e = np.asarray(arrays['embeddings'], dtype=np.float32)
    return {'embedding': e, 'base': text_features(bank).reshape(len(bank),4,-1).astype(np.float32),
            'numeric': numeric_features(bank).reshape(len(bank),4,-1),
            'checks2':np.asarray([checked_arithmetic(c['text']) for row in bank for c in row['candidates']],
                                  dtype=np.float32).reshape(len(bank),4,-1)}


def matrix(prepared, kind):
    e, b, numeric = (prepared[k] for k in ('embedding', 'base', 'numeric'))
    if kind == 'embedding': parts = [e]
    elif kind == 'hybrid': parts = [e,b]
    elif kind == 'rich': parts = [e,b,numeric]
    elif kind == 'rich_checked':parts=[e,b,numeric[:,:,5:],prepared['checks2']]
    elif kind == 'features_only': parts = [b,numeric]
    elif kind == 'features_checked':parts=[b,numeric[:,:,5:],prepared['checks2']]
    else:
        mean = e.mean(axis=1, keepdims=True)
        parts = [e, e-mean, e*mean]
        if kind == 'context': parts += [b]
        elif kind == 'context_rich': parts += [b,numeric]
        elif kind == 'context_no_vote': parts += [b[:,:,[0]+list(range(4,14))],numeric]
        elif kind == 'context_no_style': parts += [b[:,:,:4],numeric]
        elif kind == 'context_no_arithmetic': parts += [b,numeric[:,:,5:]]
        elif kind == 'context_checked':parts += [b,numeric[:,:,5:],prepared['checks2']]
        elif kind == 'checked_no_vote':parts += [b[:,:,[0]+list(range(4,14))],numeric[:,:,5:],prepared['checks2']]
        elif kind == 'checked_no_style':parts += [b[:,:,:4],numeric[:,:,5:],prepared['checks2']]
        else: raise ValueError(kind)
    return np.ascontiguousarray(np.concatenate(parts,axis=2),dtype=np.float32)


def metadata(bank):
    answer_rows = [[parse_answer(c['text']) for c in r['candidates']] for r in bank]
    patterns = []
    for answers in answer_rows:
        counts = Counter(a for a in answers if a is not None)
        patterns.append('-'.join(str(v) for v in sorted(counts.values(),reverse=True)) or 'invalid')
    hard = np.array([len(set(a for a in row if a is not None)) >= 2 for row in answer_rows])
    valid = np.array([[a is not None for a in row] for row in answer_rows])
    return {'answers':answer_rows,'patterns':patterns,'hard':hard,'valid':valid}
