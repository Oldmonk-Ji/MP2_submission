"""Correct train/inference candidate population mismatch: no extraction-based score masking."""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round29_programs import parse,encode_questions,encode_programs
from round29_model import predict_encoded
from round25_selector import scores
from features import metadata
import importlib,torch,numpy as np,functools,shutil,datetime
import round4_gates

OUT=ROOT/'experiments/round37'


def main():
    OUT.mkdir();dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        hypothesis='Candidate ranking objectives trained on every candidate; extraction masks at inference discarded learned information. Apply matching unmasked learned score.',
        evidence='TRAIN grouped diagnostic raw scores: round30 747/948, round31 761/948. This is not baseline blend accuracy.',
        source_rounds=[29,30,31,32,33],epochs_program=[15,40,80],epochs_graph=[10,25,50],alpha=[.1,.25,.5],
        training_n=4743,validation_n=798,test_opened=False,
        caveat='Unparsed candidate literals may convey answer priors or extraction quality; improvement alone would not prove semantic reasoning. No hard semantic override.'))
    b,a,y,_=load('validation');base=scores(b,a,ROOT/'experiments/round25/selected_bundle');old=round4_gates.apply_gate(b,base,'parse_and_repair',4)
    oldok=y[np.arange(len(b)),old].astype(bool);assert oldok.sum()==752;hard=metadata(b)['hard']
    parsed=[[parse(r['question'],c['text']) for c in r['candidates']] for r in b]
    pp=encode_programs([p for rr in parsed for p in rr]).reshape(len(b),4,-1)
    from round32_graph import encode_graphs,predict_encoded as predict_graph,load as graph_load
    gr,gp=encode_graphs([p for rr in parsed for p in rr],[r['question'] for r in b for c in r['candidates']]);gr=gr.reshape(len(b),4,24);gp=gp.reshape(len(b),4,24,24)
    device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(4)
    records=[dict(name='control',source=None,alpha=0.,correct=752,n=798,hard_correct=309,hard_n=355,repaired=0,damaged=0,net=0)];choices={'control':old};raw={};paths={}
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    for number in [29,30,31,32,33]:
        root=ROOT/f'experiments/round{number}'
        if number<=31:
            variants=read(root/'plan.json')['variants'];entries=[(f'{var}_e{ep}',root/'full'/var/f'epoch{ep}') for var in variants for ep in [15,40,80]]
            model_load=importlib.import_module(f'round{number}_model').load
        else:entries=[(f'e{ep}',root/'full'/f'epoch{ep}') for ep in [10,25,50]];model_load=graph_load
        for name,path in entries:
            key=f'r{number}_{name}';model,vocab=model_load(path);model.to(device);encoder=encode_questions
            if number==33:
                from round33_text import encode_questions as encoder
            q=encoder(b,vocab,256)
            if number<=31:s,_=predict_encoded(model,q,pp,device)
            else:s=predict_graph(model,q,gr,gp,device)
            delta=s-s.mean(1,keepdims=True);raw[key]=s;paths[key]=(number,path)
            for alpha in [.1,.25,.5]:
                cn=f'{key}_a{alpha}';c=round4_gates.apply_gate(b,base+alpha*delta,'parse_and_repair',4);ok=y[np.arange(len(b)),c].astype(bool);choices[cn]=c
                rec=dict(name=cn,source=key,alpha=alpha,correct=int(ok.sum()),n=len(b),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-752))
                records.append(rec);print('VAL',rec,flush=True)
    round4_gates.trace=original;winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,test_opened=False));np.savez(OUT/'validation_choices.npz',**choices);np.savez(OUT/'validation_scores.npz',**raw);print('SELECTED',winner,flush=True)
    if winner['source'] is None:return
    number,path=paths[winner['source']];bundle=OUT/'selected_bundle';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');dest=bundle/'semantic';dest.mkdir()
    for f in ['config.json','model.npz','vocabulary.json']:shutil.copyfile(path/f,dest/f)
    dump(bundle/'config.json',dict(source_round=number,alpha=winner['alpha'],mask='all'))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2,size
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))


if __name__=='__main__':main()
