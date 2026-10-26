from round22_common import ROOT,read,dump,sha
if __name__=='__main__':
    r=ROOT/'research';assert read(ROOT/'experiments/round202/fit_audit.json')['stats']['fit_parents']==3161
    mapping={'round179':'round202','round180':'round203','round181':'round204','round182':'round205'}
    for source,target in [('round180_features.py','round203_features.py'),('round180_prepare.py','round203_prepare.py'),('round181_train.py','round204_train.py'),('round182_train.py','round205_train.py'),('round182_selector.py','round205_selector.py'),('round182_checks.py','round205_checks.py')]:
        s=(r/source).read_text(encoding='utf-8')
        for a,b in mapping.items():s=s.replace(a,b)
        s=s.replace('round202_trace','round201_trace');(r/target).write_text(s,encoding='utf-8')
    dump(ROOT/'experiments/round201/frozen_before_holdout.json',dict(source_hashes={str(p.relative_to(ROOT)):sha(p) for p in r.glob('round201*.py')},fit_audit=read(ROOT/'experiments/round202/fit_audit.json'),test_opened=False))
