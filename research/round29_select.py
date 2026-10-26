"""Complete validation selection of frozen semantic checkpoints, never TEST."""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round29_programs import parse,encode_questions,encode_programs
from round29_model import predict_encoded,centered
from round25_selector import scores as baseline_scores
from features import metadata
import numpy as np
import torch
import datetime,functools,shutil,importlib,sys
import round4_gates


def main(number,partial=False):
    out=ROOT/f'experiments/round{number}'
    assert not (out/'selection.json').exists()
    prefix='pilot_' if partial else ''
    variants=read(out/'plan.json')['variants']
    dump(out/(prefix+'validation_plan.json'),dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        epochs=[15,40,80],variants=variants,
        usable=['clean','informative'],alpha=[.1,.25,.5],baseline_correct=752,
        selection='All 798 questions, correct then fewer harms then smaller alpha then asset size',test_opened=False))
    b,a,y,_=load('validation');base=baseline_scores(b,a,ROOT/'experiments/round25/selected_bundle')
    old=round4_gates.apply_gate(b,base,'parse_and_repair',4);oldok=y[np.arange(len(b)),old].astype(bool)
    assert oldok.sum()==752
    parsed=[[parse(r['question'],c['text']) for c in r['candidates']] for r in b]
    p=encode_programs([c for r in parsed for c in r]).reshape(len(b),4,-1)
    masks={kind:np.asarray([[c[kind] for c in r] for r in parsed]) for kind in ['clean','informative']}
    hard=metadata(b)['hard'];records=[];choices={};deltas={}
    module=importlib.import_module(f'round{number}_model');device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    def record(name,c,**kw):
        ok=y[np.arange(len(b)),c].astype(bool);choices[name]=c
        rec=dict(name=name,correct=int(ok.sum()),n=len(b),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),
            repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()),**kw)
        records.append(rec);return rec
    record('control',old,alpha=0.,model=None,mask=None,semantic_bytes=0)
    for variant in variants:
        for epoch in [15,40,80]:
            name=f'{variant}_e{epoch}';path=out/'full'/variant/f'epoch{epoch}'
            if partial and not (path/'config.json').exists():continue
            model,vocab=module.load(path);model.to(device)
            q=encode_questions(b,vocab,max_length=256);s,_=predict_encoded(model,q,p,device)
            for kind,mask in masks.items():
                delta=centered(s,mask);deltas[name+'_'+kind]=delta
                for alpha in [.1,.25,.5]:
                    c=round4_gates.apply_gate(b,base+alpha*delta,'parse_and_repair',4)
                    r=record(f'{name}_{kind}_a{alpha}',c,alpha=alpha,model=name,mask=kind,
                        semantic_bytes=sum(f.stat().st_size for f in path.iterdir() if f.name in ['config.json','model.npz','vocabulary.json']))
                    print('VAL',r,flush=True)
            del model;dump(out/(prefix+'validation_records.json'),records)
    round4_gates.trace=original
    winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha'],-r['semantic_bytes']))
    dump(out/(prefix+'selection.json'),dict(selected=winner,records=records,test_opened=False))
    np.savez(out/(prefix+'validation_choices.npz'),**choices);np.savez(out/(prefix+'validation_deltas.npz'),**deltas)
    print('SELECTED',winner,flush=True)
    if partial or winner['model'] is None:return
    bundle=out/'selected_bundle';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    variant,ep=winner['model'].rsplit('_e',1);shutil.copytree(out/'full'/variant/f'epoch{ep}',bundle/'semantic')
    dump(bundle/'config.json',dict(round=number,alpha=winner['alpha'],mask=winner['mask']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets)
    assert size<=16*1024**2,size
    dump(out/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))


if __name__=='__main__':main(int(sys.argv[1]),'--partial' in sys.argv)
