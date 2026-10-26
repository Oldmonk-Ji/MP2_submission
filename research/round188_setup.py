from round22_common import ROOT,read,dump,sha
if __name__=='__main__':
    r=ROOT/'research';assert read(ROOT/'experiments/round187/fit_audit.json')['stats']['fit_parents']==3161
    mapping={'round179':'round187','round180':'round188','round181':'round189','round182':'round190'}
    for source,target in [('round180_features.py','round188_features.py'),('round180_prepare.py','round188_prepare.py'),('round181_train.py','round189_train.py'),('round182_train.py','round190_train.py'),('round182_selector.py','round190_selector.py'),('round182_checks.py','round190_checks.py')]:
        s=(r/source).read_text(encoding='utf-8')
        for a,b in mapping.items():s=s.replace(a,b)
        s=s.replace('round187_trace','round186_trace')
        (r/target).write_text(s,encoding='utf-8')
    dump(ROOT/'experiments/round186/frozen_before_holdout.json',dict(source_sha256=sha(r/'round186_trace.py'),fit_audit=read(ROOT/'experiments/round187/fit_audit.json'),test_opened=False))
