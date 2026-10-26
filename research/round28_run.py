"""Finite combination of independently developed rounds 24 and 25."""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round24_selector import branch_scores
from round25_selector import scores
from round22_selector import normalized
from round4_gates import apply_gate
from features import metadata
import datetime,shutil
import numpy as np

OUT=ROOT/'experiments/round28'


def main():
    OUT.mkdir();dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        alphas=[0.,.1,.25,.5],training_n=4743,validation_n=798,test_opened=False,
        method='Finite combination of round25 ensemble and existing round24 selected linear residual; no new fit',
        selection='Complete validation; correct, fewer harms, smaller alpha',
        caveat='Adaptive validation reuse; improvement is development evidence, not independent test performance'))
    b,a,y,_=load('validation');base=scores(b,a,ROOT/'experiments/round25/selected_bundle')
    delta=normalized(branch_scores(b,a,ROOT/'experiments/round24/selected_bundle/branch'))
    with np.load(ROOT/'experiments/round22/validation_comparison.npz') as z:old=z['incumbent_choice']
    oldok=y[np.arange(len(b)),old].astype(bool);hard=metadata(b)['hard'];records=[];choices={}
    for alpha in [0.,.1,.25,.5]:
        c=apply_gate(b,base+alpha*delta,'parse_and_repair',4);ok=y[np.arange(len(b)),c].astype(bool)
        name=f'alpha{alpha}';choices[name]=c
        r=dict(name=name,alpha=alpha,correct=int(ok.sum()),n=len(b),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),
            repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()))
        records.append(r);print(r,flush=True)
    winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,test_opened=False))
    np.savez(OUT/'validation_choices.npz',**choices)
    bundle=OUT/'selected_bundle';bundle.mkdir()
    shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    shutil.copytree(ROOT/'experiments/round24/selected_bundle/branch',bundle/'branch')
    dump(bundle/'fusion.json',dict(alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))
    print('SELECTED',winner,'bytes',size,flush=True)


if __name__=='__main__':main()
