import sys
sys.path.insert(0, 'research')
sys.path.insert(0, 'DASE7506-MP2-main/code')
from round51_programs import equations
from round152_markup import clean

lines = [
    '$45 / 400 = \\$0.1125 per square foot',
    '\\left\\lfloor \\frac{120}{5} \\right\\rfloor = 24 \\text{ students}',
    '\\frac{1}{4} \\times 8 + 7 = 2 + 7 = 9',
    '24 - 15 = 9',
]
for line in lines:
    print('RAW:', repr(line))
    cl = clean(line)
    print('CLEAN:', repr(cl))
    try:
        print('EQ:', [(str(a), str(b), nm) for a, b, nm in equations(cl)])
    except Exception as e:
        print('EQERR', e)
    print()
