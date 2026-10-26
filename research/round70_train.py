from round22_common import ROOT,load,read,dump,sha
from round59_selector import features
from round70_model import Model
from round70_selector import scores
from round25_selector import scores as baseline
from round22_selector import normalized
from round4_gates import apply_gate
from torch import nn
import torch,numpy as np,time,copy,shutil
OUT=ROOT/'experiments/round70'

def fit(x,o,y,ids,path,attention):
    torch.manual_seed(70070);rng=np.random.default_rng(70070);mean=x.mean((0,1));scale=x.std((0,1)).clip(.01)
    xx=torch.tensor(np.clip((x-mean)/scale,-8,8),device='cuda');oo=torch.tensor(1-o,device='cuda');yy=torch.tensor(y,device='cuda')
    m=Model(x.shape[-1],attention).cuda();ema=copy.deepcopy(m);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.1)
    history=[]
    for epoch in range(120):
        m.train();total=0
        for ix in np.array_split(rng.permutation(len(x)),int(np.ceil(len(x)/128))):
            s=m(xx[ix],oo[ix]);t=yy[ix];loss=-(s.log_softmax(1)*t/t.sum(1,keepdim=True)).sum(1).mean()+.25*nn.functional.binary_cross_entropy_with_logits(s,t)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);opt.step()
            with torch.no_grad():
                for ep,mp in zip(ema.parameters(),m.parameters()):ep.lerp_(mp,.02)
            total+=float(loss.detach())*len(ix)
        history.append(total/len(x))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez(path,mean=mean,scale=scale,**{'state:'+k:v.cpu().numpy() for k,v in ema.state_dict().items()})
    dump(path.with_suffix('.json'),dict(training_ids=ids.tolist(),seed=70070,epochs=120,attention=attention,history=history))
    ema.eval()
    with torch.no_grad():expected=ema(xx[:16],oo[:16]).cpu().numpy()
    actual=scores(x[:16],o[:16],path,attention);assert np.allclose(expected,actual,atol=2e-4)
    permutation=[2,0,3,1]
    with torch.no_grad():p=ema(xx[:16,permutation],oo[:16][:,permutation][:,:,permutation]).cpu().numpy()
    assert np.allclose(p,expected[:,permutation],atol=2e-4)

if __name__=='__main__':
    OUT.mkdir();torch.set_num_threads(4);tick=time.perf_counter();dump(OUT/'plan.json',dict(arms=['independent','attention'],seed=70070,epochs=120,alpha=[.1,.25,.5,1.],test_opened=False,source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'research').glob('round70*.py')}))
    b,a,y,ids=load('train');v,va,vy,_=load('validation');x,o=features(b,a);vx,vo=features(v,va)
    with np.load(ROOT/'experiments/round56/train_targets.npz') as z:f=z['folds']
    bs=normalized(baseline(v,va,ROOT/'experiments/round25/selected_bundle'));bc=apply_gate(v,bs,'parse_and_repair',4);records=[]
    for attention in (False,True):
        arm='attention' if attention else 'independent'
        fit(x[f!=2],o[f!=2],y[f!=2],ids[f!=2],OUT/'diagnostic'/f'{arm}.npz',attention);h=scores(x[f==2],o[f==2],OUT/'diagnostic'/f'{arm}.npz',attention)
        hb=[r for r,k in zip(b,f) if k==2];hc=apply_gate(hb,h,'parse_and_repair',4);dump(OUT/f'{arm}_held.json',dict(n=len(hb),correct=int(y[f==2][np.arange(len(hb)),hc].sum())));print('HELD',arm,int(y[f==2][np.arange(len(hb)),hc].sum()),flush=True)
        fit(x,o,y,ids,OUT/'full'/f'{arm}.npz',attention);s=scores(vx,vo,OUT/'full'/f'{arm}.npz',attention);np.savez(OUT/f'{arm}_scores.npz',validation=s,held=h)
        for alpha in (.1,.25,.5,1.):
            c=apply_gate(v,(1-alpha)*bs+alpha*normalized(s),'parse_and_repair',4);ok=vy[np.arange(len(v)),c];bo=vy[np.arange(len(v)),bc]
            r=dict(arm=arm,alpha=alpha,correct=int(ok.sum()),repairs=int(((ok==1)&(bo==0)).sum()),harms=int(((ok==0)&(bo==1)).sum()),predictions=c.tolist());records.append(r);print({k:v for k,v in r.items() if k!='predictions'},flush=True)
    dump(OUT/'validation_records.json',records);best=max(records,key=lambda r:(r['correct'],-r['harms'],-r['alpha']))
    bundle=OUT/'bundle';bundle.mkdir();shutil.copy2(OUT/'full'/f"{best['arm']}.npz",bundle/'model.npz')
    if best['alpha']<1:shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    dump(bundle/'config.json',dict(attention=best['arm']=='attention',alpha=best['alpha']));size=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file());assert size<=16*1024**2
    pred=read(ROOT/'experiments/round63/validation_sft_predictions.json');pred['selected']=best['predictions'];dump(OUT/f"validation_{best['arm']}_predictions.json",pred)
    dump(OUT/'final_feature_selection.json',dict(best_new=dict(arm=best['arm'],correct=best['correct'],bundle='experiments/round70/bundle'),selected='round70' if best['correct']>752 else 'round25',test_target_fixed_before_test=True,test_used_for_selection=False))
    dump(OUT/'training_complete.json',dict(seconds=time.perf_counter()-tick,asset_bytes=size,test_opened=False,cpu_gpu_agreement=True,candidate_permutation_equivariant=True))
