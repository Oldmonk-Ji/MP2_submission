from round22_common import ROOT,read
if __name__=='__main__':
    r=ROOT/'research';assert read(ROOT/'experiments/round196/training_complete.json')['test_opened']==False
    mapping={'round180':'round194','round181':'round195','round182':'round196','round183':'round197','round184':'round198','round185':'round199'}
    for source,target in [('round183_prepare.py','round197_prepare.py'),('round184_train.py','round198_train.py'),('round185_features.py','round199_features.py'),('round185_train.py','round199_train.py'),('round185_selector.py','round199_selector.py'),('round185_checks.py','round199_checks.py')]:
        s=(r/source).read_text(encoding='utf-8')
        for a,b in mapping.items():s=s.replace(a,b)
        s=s.replace("assert r['correct']==756;assert c.tolist()", "assert r['correct']==read(ROOT/'experiments/round196/validation_records.json')[1]['correct'];assert c.tolist()")
        (r/target).write_text(s,encoding='utf-8')
