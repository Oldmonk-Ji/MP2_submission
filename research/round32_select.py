from data_access import ROOT
from round22_common import load,read,dump,sha
from round29_programs import parse,encode_questions
from round29_model import centered
from round32_graph import load as load_model,encode_graphs,predict_encoded
from round25_selector import scores
from features import metadata
import numpy as np
import torch
import datetime,functools,shutil,sys
import round4_gates


def main(number):
    out=ROOT/f'experiments/round{number}';assert not (out/'selection.json').exists()
    dump(out/'validation_plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),epochs=[10,25,50],
        masks=['clean','informative'],alphas=[.1,.25,.5],validation_n=798,test_opened=False))
    b,a,y,_=load('validation');base=scores(b,a,ROOT/'experiments/round25/selected_bundle')
    old=round4_gates.apply_gate(b,base,'parse_and_repair',4);oldok=y[np.arange(len(b)),old].astype(bool);assert oldok.sum()==752
    parsed=[[parse(r['question'],c['text']) for c in r['candidates']] for r in b]
    roles,pairs=encode_graphs([c for r in parsed for c in r],[r['question'] for r in b for c in r['candidates']])
    roles=roles.reshape(len(b),4,24);pairs=pairs.reshape(len(b),4,24,24)
    masks={kind:np.asarray([[c[kind] for c in row] for row in parsed]) for kind in ['clean','informative']}
    hard=metadata(b)['hard'];records=[dict(name='control',correct=752,n=798,hard_correct=309,hard_n=355,
        repaired=0,damaged=0,net=0,epoch=None,alpha=0.,mask=None)];choices={'control':old};deltas={}
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    encoder=encode_questions
    if number==33:
        from round33_text import encode_questions as encoder
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    for epoch in [10,25,50]:
        path=out/'full'/f'epoch{epoch}';model,vocab=load_model(path);model.to(device)
        q=encoder(b,vocab,max_length=256);s=predict_encoded(model,q,roles,pairs,device)
        for kind,mask in masks.items():
            delta=centered(s,mask);deltas[f'e{epoch}_{kind}']=delta
            for alpha in [.1,.25,.5]:
                name=f'e{epoch}_{kind}_a{alpha}';c=round4_gates.apply_gate(b,base+alpha*delta,'parse_and_repair',4)
                ok=y[np.arange(len(b)),c].astype(bool);choices[name]=c
                r=dict(name=name,epoch=epoch,mask=kind,alpha=alpha,correct=int(ok.sum()),n=len(b),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),
                    repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-oldok.sum()))
                records.append(r);print('VAL',r,flush=True)
        dump(out/'validation_records.json',records)
    round4_gates.trace=original
    winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha']))
    dump(out/'selection.json',dict(selected=winner,records=records,test_opened=False))
    np.savez(out/'validation_choices.npz',**choices);np.savez(out/'validation_deltas.npz',**deltas)
    print('SELECTED',winner,flush=True)
    if winner['epoch'] is None:return
    bundle=out/'selected_bundle';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    shutil.copytree(out/'full'/f"epoch{winner['epoch']}",bundle/'semantic')
    dump(bundle/'config.json',dict(round=number,mask=winner['mask'],alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2
    dump(out/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))


if __name__=='__main__':main(int(sys.argv[1]))
