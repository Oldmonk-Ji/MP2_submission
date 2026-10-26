from round22_common import ROOT,read,dump
import re
from collections import Counter
if __name__=='__main__':
    out=ROOT/'experiments/round200';out.mkdir();examples=[]
    for r in read(ROOT/'experiments/round191/fit_same_value_review.json'):
        for line in r['candidate_lines']:
            c=Counter(re.findall(r'(?<![\w.])\d+(?:\.\d+)?(?!\w)',line))
            if any(v>1 for v in c.values()) and re.search(r'\d\s+[A-Za-z]|\\text',line):examples.append(dict(train_index=r['train_index'],question=r['question'],line=line))
    dump(out/'fit_occurrence_examples.json',examples);print('fit annotated repeated-number lines',len(examples))
    for r in examples[:12]:print(r)
