from round22_common import ROOT,read
if __name__=='__main__':
    r=ROOT/'research';assert read(ROOT/'experiments/round205/training_complete.json')['test_opened']==False
    mapping={'round180':'round203','round181':'round204','round182':'round205','round183':'round206','round184':'round207','round185':'round208'}
    for source,target in [('round183_prepare.py','round206_prepare.py'),('round184_train.py','round207_train.py'),('round185_features.py','round208_features.py'),('round185_train.py','round208_train.py'),('round185_selector.py','round208_selector.py'),('round185_checks.py','round208_checks.py')]:
        s=(r/source).read_text(encoding='utf-8')
        for a,b in mapping.items():s=s.replace(a,b)
        s=s.replace("assert r['correct']==756;assert c.tolist()", "assert r['correct']==read(ROOT/'experiments/round205/validation_records.json')[1]['correct'];assert c.tolist()")
        (r/target).write_text(s,encoding='utf-8')
