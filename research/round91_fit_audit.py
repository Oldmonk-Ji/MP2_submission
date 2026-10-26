from round22_common import ROOT,load,read,dump,sha
from round91_consistency import build,constraints,checks
from round22_selector import normalized
from round4_gates import apply_gate
import numpy as np

if __name__=='__main__':
    out=ROOT/'experiments/round91';out.mkdir();checks();b,a,y,ids=load('train')
    with np.load(ROOT/'experiments/round77/nested/components.npz') as z:s=z['components'];f=z['folds']
    w=[m['weight'] for m in read(ROOT/'experiments/round25/selected_bundle/config.json')['members']]
    score=sum(ww*normalized(s[:,:,j]) for j,ww in enumerate(w));c=apply_gate(b,score,'parse_and_repair',4);ix=np.flatnonzero(f!=2);x=build([b[i] for i in ix]);detected=x[:,:,5]>0
    examples=[]
    for k,i in enumerate(ix):
        if detected[k].any() and y[i,c[i]]==0:
            examples.append(dict(train_index=int(ids[i]),question=b[i]['question'],chosen=int(c[i]),labels=y[i].tolist(),conflicts=detected[k].tolist(),constraints=[constraints(cc['text']) for cc in b[i]['candidates']]))
    report=dict(fit_questions=len(ix),candidates_with_symbolic_constraints=int(x[:,:,0].sum()),conflict_candidates=int(detected.sum()),conflict_correct=int((detected*y[ix]).sum()),conflict_wrong=int((detected*(1-y[ix])).sum()),error_examples=examples,checks=checks(),test_opened=False,train_fit_only=True)
    dump(out/'fit_audit.json',report);dump(out/'rule_freeze.json',dict(source='research/round91_consistency.py',sha256=sha(ROOT/'research/round91_consistency.py'),fit_audit_sha256=sha(out/'fit_audit.json'),purpose='Soft internal equation consistency evidence, not semantic truth or hard veto',test_opened=False))
    print({k:v for k,v in report.items() if k!='error_examples'},'error_examples',len(examples),flush=True)
