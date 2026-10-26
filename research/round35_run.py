"""Matched relation auxiliary on/off; jointly fit all 4743, no stacked in-sample readout."""
from data_access import ROOT
from round22_common import load,read,dump,sha
from round29_programs import encode_questions,parse
from round29_model import trim,save
from round32_graph import encode_graphs
from round35_model import JointModel,predict_encoded
from round25_selector import scores as baseline_scores
from round22_selector import normalized
from features import metadata
import numpy as np
import torch
from torch import nn
import json,time,datetime,functools,shutil
import round4_gates

OUT=ROOT/'experiments/round35'


def main():
    OUT.mkdir();dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='End-to-end candidate embedding and question relation score interaction, auxiliary relation CE on/off',
        training_n=4743,validation_n=798,test_opened=False,seed=35035,epochs=[10,25,50],
        auxiliary_weights=[0.,.5],alphas=[.1,.25,.5],
        initialization='Question encoder only from official TRAIN scratch round29 full reference_only epoch15',
        control='Same features, initialization, optimizer and exposures; relation supervision on/off',
        loss='uniform-positive candidate cross entropy + .25 candidate BCE; optional auxiliary role and pair CE',
        no_stacked_readout=True))
    torch.set_num_threads(4)
    b,a,y,ids=load('train');v,va,vy,_=load('validation');dump(OUT/'training_ids.json',ids.tolist())
    rows=[json.loads(s) for s in open(ROOT/'experiments/round29/training_programs.jsonl',encoding='utf-8')]
    assert [r['train_index'] for r in rows]==ids.tolist()
    source=ROOT/'experiments/round29/full/reference_only/epoch15';vocab=read(source/'vocabulary.json')
    q=encode_questions(b,vocab,256);vq=encode_questions(v,vocab,256)
    rr,rp=encode_graphs([r['reference'] for r in rows],[r['question'] for r in rows],strict=True)
    cr,cp=encode_graphs([c for r in rows for c in r['candidates']],[r['question'] for r in rows for c in r['candidates']])
    cr=cr.reshape(len(b),4,24);cp=cp.reshape(len(b),4,24,24)
    vp=[parse(r['question'],c['text']) for r in v for c in r['candidates']]
    vr,pp=encode_graphs(vp,[r['question'] for r in v for c in r['candidates']]);vr=vr.reshape(len(v),4,24);pp=pp.reshape(len(v),4,24,24)
    device='cuda' if torch.cuda.is_available() else 'cpu';e=np.asarray(a['embeddings'],np.float32);ve=np.asarray(va['embeddings'],np.float32)
    all_scores={}
    def supervised(logp,target):
        mask=target.ge(0);loss=-logp.gather(-1,target.clamp_min(0)[...,None]).squeeze(-1)*mask
        dims=tuple(range(1,target.ndim));per=loss.sum(dims)/mask.sum(dims).clamp_min(1);valid=mask.reshape(len(mask),-1).any(1)
        return per[valid].mean() if valid.any() else logp.sum()*0
    for auxiliary in [0.,.5]:
        torch.manual_seed(35035);rng=np.random.default_rng(35035);model=JointModel(len(vocab));state=model.state_dict()
        with np.load(source/'model.npz') as z:
            for k in state:
                if k=='words.weight' or k.startswith('encoder.'):state[k]=torch.from_numpy(z[k].copy())
        for name,target,n in [('role_prior',rr,3),('pair_prior',rp,6)]:
            count=np.bincount(target[target>=0],minlength=n)+1;state[name]=torch.from_numpy((count/count.sum()).astype(np.float32))
        state['embedding_mean']=torch.from_numpy(e.mean((0,1)));state['embedding_scale']=torch.from_numpy(e.std((0,1)).clip(.001))
        model.load_state_dict(state);model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1)
        history=[];tick=time.perf_counter()
        for epoch in range(1,51):
            model.train();total=0.
            for ix in np.array_split(rng.permutation(len(b)),int(np.ceil(len(b)/48))):
                qq=torch.as_tensor(trim(q[ix]),device=device);logp=model(qq)
                rt=torch.as_tensor(cr[ix],device=device);pt=torch.as_tensor(cp[ix],device=device);et=torch.as_tensor(e[ix],device=device)
                s=model.joint_score(logp,rt,pt,et);target=torch.as_tensor(y[ix],device=device)
                loss=-(s.log_softmax(1)*target/target.sum(1,keepdim=True)).sum(1).mean()+.25*nn.functional.binary_cross_entropy_with_logits(s,target)
                if auxiliary:loss+=auxiliary*(supervised(logp[0],torch.as_tensor(rr[ix],device=device))+supervised(logp[1],torch.as_tensor(rp[ix],device=device)))
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step();total+=float(loss.detach())*len(ix)
            rec=dict(epoch=epoch,loss=total/len(b),seconds=time.perf_counter()-tick);history.append(rec)
            if epoch%5==0:print('TRAIN',auxiliary,rec,flush=True)
            if epoch in [10,25,50]:
                name=f'aux{auxiliary}_e{epoch}';dest=OUT/'models'/name;dest.mkdir(parents=True);save(model,dest/'model.npz');dump(dest/'vocabulary.json',vocab)
                dump(dest/'config.json',dict(training_n=len(b),auxiliary=auxiliary,epoch=epoch,seed=35035));dump(dest/'history.json',history)
                all_scores[name]=predict_encoded(model,vq,vr,pp,ve,device)
        dump(OUT/f'aux{auxiliary}_history.json',history)
    np.savez(OUT/'validation_scores.npz',**all_scores)
    base=baseline_scores(v,va,ROOT/'experiments/round25/selected_bundle');old=round4_gates.apply_gate(v,base,'parse_and_repair',4)
    oldok=vy[np.arange(len(v)),old].astype(bool);assert oldok.sum()==752;hard=metadata(v)['hard']
    records=[dict(name='control',model=None,alpha=0.,correct=752,n=798,hard_correct=309,hard_n=355,repaired=0,damaged=0,net=0)];choices={'control':old}
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    for name,s in all_scores.items():
        for alpha in [.1,.25,.5]:
            key=f'{name}_a{alpha}';c=round4_gates.apply_gate(v,base+alpha*normalized(s),'parse_and_repair',4);ok=vy[np.arange(len(v)),c].astype(bool);choices[key]=c
            rec=dict(name=key,model=name,alpha=alpha,correct=int(ok.sum()),n=len(v),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-752))
            records.append(rec);print('VAL',rec,flush=True)
    round4_gates.trace=original;winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,test_opened=False));np.savez(OUT/'validation_choices.npz',**choices);print('SELECTED',winner,flush=True)
    if winner['model'] is None:return
    bundle=OUT/'selected_bundle';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline')
    dest=bundle/'semantic';dest.mkdir()
    for f in ['config.json','model.npz','vocabulary.json']:shutil.copyfile(OUT/'models'/winner['model']/f,dest/f)
    dump(bundle/'config.json',dict(alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2,size
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))


if __name__=='__main__':main()
