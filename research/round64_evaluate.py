from round22_common import ROOT,load,read,dump,sha
from round64_selector import select
from round25_selector import scores as baseline
from round4_gates import apply_gate
import numpy as np,shutil
OUT=ROOT/'experiments/round64'
if __name__=='__main__':
    assert sha(ROOT/'research/round64_selector.py')==read(OUT/'plan.json')['selector_sha256']
    b,_,y,ids=load('train');v,va,vy,_=load('validation')
    with np.load(ROOT/'experiments/round63/diagnostic_scores.npz') as z:ix=z['held'];s=(z['sft_48048']+z['sft_48049'])/2
    with np.load(ROOT/'experiments/round63/train_targets.npz') as z:bs=z['base_scores'][ix];m=z['mask'][ix];k=z['matches'][ix]
    hb=[b[i] for i in ix];diagnostic=[]
    for pool in ('sum','mean','max'):
        c=select(hb,bs,s,m,k,pool);diagnostic.append(dict(pool=pool,correct=int(y[ix][np.arange(len(ix)),c].sum()),n=len(ix)))
    dump(OUT/'diagnostic.json',diagnostic)
    with np.load(ROOT/'experiments/round63/validation_sft_scores.npz') as z:s=z['scores'];m=z['mask'];k=z['matches']
    bs=baseline(v,va,ROOT/'experiments/round25/selected_bundle');bc=apply_gate(v,bs,'parse_and_repair',4);records=[]
    for pool in ('sum','mean','max'):
        c=select(v,bs,s,m,k,pool);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
        r=dict(pool=pool,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),predictions=c.tolist());records.append(r);print({k:v for k,v in r.items() if k!='predictions'},flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms']))
    bundle=OUT/'bundle';shutil.copytree(ROOT/'experiments/round63/bundle_sft',bundle);dump(bundle/'config.json',dict(pool=best['pool']))
    pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=best['predictions'];dump(OUT/f"validation_{best['pool']}_predictions.json",pred)
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['pool'],correct=best['correct'],bundle='experiments/round64/bundle'),selected='round64' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
