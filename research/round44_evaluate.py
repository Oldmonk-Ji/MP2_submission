from data_access import ROOT,CODE,read_jsonl
from round22_common import read,dump,load,sha
from round44_selector import infer_features,program_scores,select
from round25_selector import scores
from round4_gates import apply_gate
from round40_evaluate import metrics
from scipy import sparse
import numpy as np,time,json,shutil,sys

OUT=ROOT/'experiments/round44';ARMS=['sft','reward','reward_relations'];SEEDS=[44044,44045]

def prepare_validation():
    # Fixed code only; this stage does not load validation reference answers.
    for p,h in read(OUT/'training_freeze.json')['hashes'].items():assert sha(ROOT/p)==h,p
    bank=read_jsonl(CODE/'data/candidates/gsm8k_validation.jsonl');tick=time.perf_counter();x,mask,matches,packets=infer_features(bank)
    sparse.save_npz(OUT/'validation_features.npz',x);np.savez(OUT/'validation_inputs.npz',mask=mask,matches=matches)
    (OUT/'validation_proposals.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in packets),encoding='utf-8')
    dump(OUT/'validation_input_preparation.json',dict(n=len(bank),has_proposal=int(mask.any(1).sum()),seconds=time.perf_counter()-tick,references_read=False,test_opened=False))
    print(read(OUT/'validation_input_preparation.json'),flush=True)

def evaluate():
    assert (OUT/'training_complete.json').exists()
    for p,h in read(OUT/'training_freeze.json')['hashes'].items():assert sha(ROOT/p)==h,p
    b,a,y,ids=load('validation');bank=read_jsonl(CODE/'data/candidates/gsm8k_validation.jsonl');basepath=ROOT/'experiments/round25/selected_bundle'
    bs=scores(b,a,basepath);base=apply_gate(b,bs,'parse_and_repair',4);assert int(y[np.arange(len(y)),base].sum())==752
    x=sparse.load_npz(OUT/'validation_features.npz')
    with np.load(OUT/'validation_inputs.npz') as z:mask=z['mask'];matches=z['matches']
    records=[];choices=dict(control=base);scored={};basebytes=sum(p.stat().st_size for p in basepath.rglob('*') if p.is_file());proofs=[dict(answer=None)]*len(b)
    r=metrics(b,y,base,base,proofs);r.update(name='control',asset_bytes=basebytes,eligible=True);records.append(r)
    for arm in ARMS:
        paths=[OUT/f'full/{arm}_s{seed}' for seed in SEEDS];s=program_scores(x,paths);scored[arm]=s
        for mode in ('execute_top','fixed_residual'):
            name=arm+'_'+mode;c=select(b,bs,s,mask,matches,mode);choices[name]=c
            cfg=dict(arm=arm,mode=mode,policies=['policy0','policy1'],alpha=.25)
            cfgbytes=len((json.dumps(cfg,ensure_ascii=False,indent=2)+'\n').replace('\n','\r\n').encode())
            assets=basebytes+sum((path/f).stat().st_size for path in paths for f in ('config.json','model.npz'))+cfgbytes
            r=metrics(b,y,base,c,proofs);r.update(name=name,arm=arm,mode=mode,asset_bytes=assets,eligible=assets<=16*1024**2,
                proposal_n=int(mask.any(1).sum()),oracle_correct_program_n=int((matches&y[:,None,:].astype(bool)).any((1,2)).sum()))
            records.append(r);dump(OUT/f'validation_{name}_predictions.json',dict(protocol='dase7506-mp2-open-hard-verifier-v5',split='validation',ids=[r['id'] for r in bank],selected=c.astype(int).tolist()));print('VALIDATION',r,flush=True)
    selected=max((r for r in records if r['eligible']),key=lambda r:(r['correct'],-r['harms'],r['name']=='control',-r['asset_bytes']))
    trial=max((r for r in records if r['name']!='control' and r['eligible']),key=lambda r:(r['correct'],-r['harms'],-r['asset_bytes']))
    bundle=OUT/('selected_bundle' if selected['name']!='control' else 'evaluation_bundle');bundle.mkdir()
    chosen=selected if selected['name']!='control' else trial;shutil.copytree(basepath,bundle/'baseline')
    for j,seed in enumerate(SEEDS):
        dest=bundle/f'policy{j}';dest.mkdir()
        for f in ('config.json','model.npz'):shutil.copyfile(OUT/f'full/{chosen["arm"]}_s{seed}'/f,dest/f)
    dump(bundle/'config.json',dict(arm=chosen['arm'],mode=chosen['mode'],policies=['policy0','policy1'],alpha=.25))
    actual=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert actual<=16*1024**2
    chosen['asset_bytes']=actual
    dump(OUT/'selection.json',dict(selected=selected,best_noncontrol=trial,records=records,evaluation_bundle=str(bundle.relative_to(ROOT)),
        evaluation_bundle_name=chosen['name'],test_opened=False))
    np.savez(OUT/'validation_choices.npz',**choices);np.savez(OUT/'validation_scores.npz',baseline=bs,**scored);print('SELECTED',selected,flush=True)

if __name__=='__main__':globals()[sys.argv[1]]()
