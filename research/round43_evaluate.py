from data_access import ROOT,CODE,read_jsonl
from round22_common import load,read,dump,sha
from round22_selector import normalized,dense_forward
from round25_selector import scores
from round3_features import build
from round4_gates import apply_gate
from round43_dpo import objective,ARMS,SEEDS
from round40_evaluate import metrics
import numpy as np,torch,shutil,time

OUT=ROOT/'experiments/round43'
def checks():
    y=torch.tensor([[1.,0.,1.,0.]]);ref=torch.tensor([[.4,-.2,.8,.3]]);s=ref.clone().requires_grad_()
    loss=objective(s,y,'dpo',ref);assert abs(loss.item()-np.log(2))<1e-6;loss.backward()
    assert (s.grad[y.bool()]<0).all() and (s.grad[~y.bool()]>0).all()
    order=torch.tensor([3,1,0,2]);assert torch.allclose(loss,objective(s[:,order],y[:,order],'dpo',ref[:,order]))
    assert torch.allclose(loss,objective(s+17,y,'dpo',ref))
    allgood=objective(s,torch.ones_like(y),'dpo',ref);assert torch.isfinite(allgood)
    dump(OUT/'objective_checks.json',dict(reference_initial_loss_log2=True,preference_gradient_direction=True,
        candidate_permutation_invariant=True,common_logit_shift_invariant_for_preferences=True,no_fake_negative_all_correct=True))

def main():
    assert sha(ROOT/'research/round43_dpo.py')==read(OUT/'plan.json')['source_hash'];checks()
    b,a,y,ids=load('validation');bank=read_jsonl(CODE/'data/candidates/gsm8k_validation.jsonl');start=time.perf_counter()
    basepath=ROOT/'experiments/round25/selected_bundle';bs=scores(b,a,basepath);base=apply_gate(b,bs,'parse_and_repair',4);assert int(y[np.arange(len(y)),base].sum())==752
    x=build(b,a,'ground');without=bs.copy();basebytes=sum(p.stat().st_size for p in basepath.rglob('*') if p.is_file())
    removed=0
    for j,seed in zip((4,5),SEEDS):
        cfg=read(basepath/f'member{j}/config.json');assert cfg['seed']==seed and cfg['features']=='ground'
        with np.load(basepath/f'member{j}/model.npz') as w:without-=.125*normalized(dense_forward(x,w))
        removed+=sum(p.stat().st_size for p in (basepath/f'member{j}').iterdir() if p.is_file())
    records=[];pred=dict(control=base);scored=dict(control=bs);proofs=[dict(answer=None)]*len(b)
    r=metrics(b,y,base,base,proofs);r.update(arm='control',asset_bytes=basebytes);records.append(r)
    for arm in ARMS:
        s=without.copy();assetbytes=basebytes-removed
        for seed in SEEDS:
            path=OUT/f'full/{arm}_s{seed}'
            with np.load(path/'model.npz') as w:s+=.125*normalized(dense_forward(x,w))
            assetbytes+=sum((path/f).stat().st_size for f in ('config.json','model.npz'))
        c=apply_gate(b,s,'parse_and_repair',4);r=metrics(b,y,base,c,proofs);r.update(arm=arm,asset_bytes=assetbytes,eligible=assetbytes<=16*1024**2)
        records.append(r);pred[arm]=c;scored[arm]=s
        dump(OUT/f'validation_{arm}_predictions.json',dict(protocol='dase7506-mp2-open-hard-verifier-v5',split='validation',ids=[r['id'] for r in bank],selected=c.astype(int).tolist()))
        print('VALIDATION',r,flush=True)
    chosen=max(records,key=lambda r:(r['correct'],-r['harms'],r['arm']=='control',-r['asset_bytes']))
    if chosen['arm']!='control':
        bundle=OUT/'selected_bundle';shutil.copytree(basepath,bundle)
        for j,seed in zip((4,5),SEEDS):
            for name in ('config.json','model.npz'):shutil.copyfile(OUT/f'full/{chosen["arm"]}_s{seed}'/name,bundle/f'member{j}'/name)
        assert sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file())==chosen['asset_bytes']
    dump(OUT/'selection.json',dict(selected=chosen,records=records,seconds=time.perf_counter()-start,test_opened=False))
    np.savez(OUT/'validation_choices.npz',**pred);np.savez(OUT/'validation_scores.npz',**scored)
    print('SELECTED',chosen,flush=True)

if __name__=='__main__':main()
