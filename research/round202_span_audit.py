from round22_common import ROOT,read,dump
from round201_spans import align
from round51_programs import equations
from round152_markup import clean
from collections import Counter
if __name__=='__main__':
    counts=Counter();examples=[]
    for r in read(ROOT/'experiments/round200/fit_occurrence_examples.json'):
        for a,b,names in equations(clean(r['line'])):
            if names:continue
            counts['numeric_equations']+=1;d=align(r['line'],a,b,r['question']);counts['uniquely_aligned']+=bool(d)
            if d:
                counts['nodes_with_unit']+=sum(bool(v['unit']) for v in d.values());counts['nodes_with_state']+=sum(bool(v['state']) for v in d.values());counts['nodes_with_owner']+=sum(bool(v['owner']) for v in d.values())
                if len(examples)<15:examples.append(dict(**r,positions=list(d.values())))
    dump(ROOT/'experiments/round202/fit_span_audit.json',dict(counts=counts,examples=examples,test_opened=False,source='TRAIN fit-only review200',rules_changed=False));print(counts)
