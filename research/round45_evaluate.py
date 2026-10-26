from data_access import ROOT,CODE,read_jsonl
from round22_common import load,read,dump,sha
from round22_selector import dense_forward,normalized
from round25_selector import scores
from round44_selector import program_scores,select
from round3_features import build
from round4_gates import apply_gate
from round40_evaluate import metrics
from round45_rlvr import ARMS,SEEDS
from scipy import sparse
import numpy as np,shutil

OUT=ROOT/'experiments/round45'
def size(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
def make_bundle(row,path):
    base=ROOT/'experiments/round25/selected_bundle';level=row['level'];arm=row['arm']
    if level=='candidate':
        shutil.copytree(base,path)
        for j,seed in zip((4,5),SEEDS[level]):
            for f in ('config.json','model.npz'):shutil.copyfile(OUT/f'full/{level}_{arm}_s{seed}'/f,path/f'member{j}'/f)
        cfg=read(path/'config.json');cfg.update(level=level,arm=arm);dump(path/'config.json',cfg)
    else:
        path.mkdir();shutil.copytree(base,path/'baseline')
        for j,seed in enumerate(SEEDS[level]):
            dest=path/f'policy{j}';dest.mkdir()
            for f in ('config.json','model.npz'):shutil.copyfile(OUT/f'full/{level}_{arm}_s{seed}'/f,dest/f)
        dump(path/'config.json',dict(level=level,arm=arm,mode=row['mode'],policies=['policy0','policy1'],alpha=.25))
    assert size(path)<=16*1024**2

def main():
    assert (OUT/'training_complete.json').exists()
    for p,h in read(OUT/'plan.json')['source_and_input_hashes'].items():assert sha(ROOT/p)==h,p
    b,a,y,ids=load('validation');bank=read_jsonl(CODE/'data/candidates/gsm8k_validation.jsonl');basepath=ROOT/'experiments/round25/selected_bundle'
    bs=scores(b,a,basepath);base=apply_gate(b,bs,'parse_and_repair',4);assert int(y[np.arange(len(y)),base].sum())==752
    x=build(b,a,'ground');rx=sparse.load_npz(ROOT/'experiments/round44/validation_features.npz')
    with np.load(ROOT/'experiments/round44/validation_inputs.npz') as z:mask=z['mask'];matches=z['matches']
    without=bs.copy();removed=0
    for j in (4,5):
        with np.load(basepath/f'member{j}/model.npz') as w:without-=.125*normalized(dense_forward(x,w))
        removed+=size(basepath/f'member{j}')
    proofs=[dict(answer=None)]*len(b);records=[];choices=dict(control=base);allscore=dict(control=bs)
    r=metrics(b,y,base,base,proofs);r.update(name='control',asset_bytes=size(basepath),asset_eligible=True);records.append(r)
    for level in ('candidate','relation'):
        for arm in ARMS[level]:
            paths=[OUT/f'full/{level}_{arm}_s{seed}' for seed in SEEDS[level]]
            assets=size(basepath)+sum((p/f).stat().st_size for p in paths for f in ('model.npz','config.json'))+2048
            if level=='candidate':
                s=without.copy();assets-=removed
                for p in paths:
                    with np.load(p/'model.npz') as w:s+=.125*normalized(dense_forward(x,w))
                modes=['fixed_members']
            else:s=program_scores(rx,paths);modes=['execute_top','fixed_residual']
            allscore[level+'_'+arm]=s
            for mode in modes:
                name=level+'_'+arm+'_'+mode;c=apply_gate(b,s,'parse_and_repair',4) if level=='candidate' else select(b,bs,s,mask,matches,mode)
                r=metrics(b,y,base,c,proofs);r.update(name=name,level=level,arm=arm,mode=mode,asset_bytes_upper_bound=assets,asset_eligible=assets<=16*1024**2)
                records.append(r);choices[name]=c
                dump(OUT/f'validation_{name}_predictions.json',dict(protocol='dase7506-mp2-open-hard-verifier-v5',split='validation',ids=[q['id'] for q in bank],selected=c.astype(int).tolist()))
                print('VALIDATION',r,flush=True)
    key=lambda r:(r['correct'],-r['harms'],r['name']=='control',-r.get('asset_bytes',r.get('asset_bytes_upper_bound',0)))
    selected=max((r for r in records if r['asset_eligible']),key=key);bundles=[]
    for level in ('candidate','relation'):
        best=max((r for r in records if r.get('level')==level and r['asset_eligible']),key=key)
        path=OUT/('selected_bundle' if best['name']==selected['name'] else f'evaluation_bundle_{level}')
        make_bundle(best,path);best['asset_bytes']=size(path)
        bundles.append(dict(name=best['name'],level=level,bundle=str(path.relative_to(ROOT)),correct=best['correct'],asset_bytes=size(path)))
    dump(OUT/'selection.json',dict(selected=selected,records=records,evaluation_bundles=bundles,test_opened=False,
        eligibility_scope='Size checked. Fresh interface/runtime verification follows; reference-machine timing not certified.'))
    np.savez(OUT/'validation_choices.npz',**choices);np.savez(OUT/'validation_scores.npz',**allscore);print('SELECTED',selected,flush=True)

if __name__=='__main__':main()
