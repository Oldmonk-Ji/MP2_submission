import sys
sys.path.insert(0, 'research')
sys.path.insert(0, 'DASE7506-MP2-main/code')
import ast
from fractions import Fraction as F
from round22_common import load
from round29_programs import quantities
from round51_programs import equations, value
from round152_markup import clean
from round175_roles import kind
from round201_trace import numeric, Unknown, unique, choose, hint, names_in
from round201_spans import align

v, va, vy, _ = load('validation')
row = v[133]
q = row['question']
print('Q:', q)
qs = quantities(q)
print('QUANTITIES:', [(n['value'], kind(q, i)) for i, n in enumerate(qs)])

c = row['candidates'][1]
direct = {}
for i, n in enumerate(qs):
    if kind(q, i) != 'ordinal':
        direct.setdefault(F(n['value']), []).append(('N', i))
cache = {}
tainted = set()
line_context = ''
metadata = {}

def bind(v, node=None):
    if v in tainted:
        raise Unknown('unresolved_prior_result')
    ps = list(direct.get(v, [])) + list(cache.get(v, []))
    for i, n in enumerate(qs):
        nv = F(n['value']); p = ('N', i); fraction = None
        k = kind(q, i)
        if k == 'percent':
            fraction = ('DIV', p, ('L', '100')); fv = nv / 100
        elif k == 'fraction_denominator' and nv:
            fraction = ('DIV', ('L', '1'), p); fv = 1 / nv
        elif k == 'fraction' and 0 < nv < 1:
            fraction = p; fv = nv
        if fraction is not None:
            if fv == v: ps.append(fraction)
            if 1 - fv == v: ps.append(('SUB', ('L', '1'), fraction))
            if 1 + fv == v: ps.append(('ADD', ('L', '1'), fraction))
            if fv and 1 / fv == v: ps.append(('DIV', ('L', '1'), fraction))
    if v in (F(0), F(1)):
        ps.append(('L', str(v)))
    if not ps:
        raise Unknown('ungrounded_operand')
    return unique(choose(q, qs, ps, v, line_context, metadata, None))

OPS = {ast.Add: 'ADD', ast.Sub: 'SUB', ast.Mult: 'MUL', ast.Div: 'DIV'}

def visit(n):
    if isinstance(n, ast.Constant) and type(n.value) in (int, float):
        return bind(F(str(n.value)), n)
    if isinstance(n, ast.BinOp) and type(n.op) in OPS:
        a, b = visit(n.left), visit(n.right)
        return unique([(OPS[type(n.op)], x, y) for x, y in __import__('itertools').product(a, b)])
    raise Unknown('unsupported_expression')

for li, line in enumerate(clean(c['text']).splitlines()):
    eqs = list(equations(line))
    if not eqs:
        continue
    for aa, bb, nm in eqs:
        try:
            left, right = numeric(aa), numeric(bb)
            target = right
            print('L%d: %r | left=%s right=%s names=%s' % (li, line.strip()[:70], left, right, nm))
            if left != right:
                raise Unknown('incorrect_numeric_equality')
            expr = aa if any(isinstance(n, ast.BinOp) for n in ast.walk(aa)) else bb
            ps = visit(expr)
            if any(value(p, qs) != target for p in ps):
                raise Unknown('lineage_value_mismatch')
            merged = unique(cache.get(target, []) + ps)
            cache[target] = merged
            print('   ACCEPT -> cache[%s] has %d programs' % (target, len(merged)))
        except Exception as e:
            if 'target' in dir() and target is not None:
                tainted.add(target)
            print('   FAIL:', e, '| tainted:', sorted(tainted))
