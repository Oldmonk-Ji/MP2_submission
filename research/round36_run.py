from data_access import ROOT
from round22_common import load,read,dump,sha
from round36_model import vocabulary,encode,batch,TextModel,predict_data
from round29_model import save
from round25_selector import scores as baseline_scores
from round22_selector import normalized
from features import metadata
import numpy as np
import torch
from torch import nn
import datetime,time,functools,shutil
import round4_gates

OUT=ROOT/'experiments/round36'


def main():
    OUT.mkdir();dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        method='Shared bidirectional recurrent question/candidate encoder with token cross-attention and entity identities; no arithmetic extraction gate',
        training_n=4743,validation_n=798,test_opened=False,seed=36036,epochs=[10,25,50],text_enabled=[False,True],
        vocabulary=6000,width=48,question_limit=192,candidate_limit=320,alphas=[.1,.25,.5],
        control='Same initialization and utility head, token feature channel zeroed versus enabled. Official embeddings present in both.',
        inference='Question and original candidate text plus original supplied embeddings; no reference input',
        loss='uniform positive cross entropy + .25 BCE',no_external_pretrained=True))
    torch.set_num_threads(4);b,a,y,ids=load('train');v,va,vy,_=load('validation');dump(OUT/'training_ids.json',ids.tolist())
    vocab=vocabulary(b);data,stat=encode(b,a,vocab);vd,vstat=encode(v,va,vocab);dump(OUT/'tokenization.json',dict(train=stat,validation=vstat))
    device='cuda' if torch.cuda.is_available() else 'cpu';saved={}
    for enabled in [False,True]:
        torch.manual_seed(36036);rng=np.random.default_rng(36036);model=TextModel(len(vocab),enabled)
        model.mean.copy_(torch.from_numpy(data['e'].mean((0,1))));model.scale.copy_(torch.from_numpy(data['e'].std((0,1)).clip(.001)))
        model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.1);history=[];tick=time.perf_counter()
        for epoch in range(1,51):
            model.train();total=0.
            for ix in np.array_split(rng.permutation(len(b)),int(np.ceil(len(b)/32))):
                s=model(**batch(data,ix,device));target=torch.as_tensor(y[ix],device=device)
                loss=-(s.log_softmax(1)*target/target.sum(1,keepdim=True)).sum(1).mean()+.25*nn.functional.binary_cross_entropy_with_logits(s,target)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();total+=float(loss.detach())*len(ix)
            rec=dict(epoch=epoch,loss=total/len(b),seconds=time.perf_counter()-tick);history.append(rec)
            if epoch%5==0:print('TRAIN',enabled,rec,flush=True)
            if epoch in [10,25,50]:
                name=f'text{int(enabled)}_e{epoch}';dest=OUT/'models'/name;dest.mkdir(parents=True);save(model,dest/'model.npz');dump(dest/'vocabulary.json',vocab)
                dump(dest/'config.json',dict(text_enabled=enabled,training_n=len(b),epoch=epoch,seed=36036));dump(dest/'history.json',history)
                saved[name]=predict_data(model,vd,device)
            dump(OUT/f'text{int(enabled)}_history.json',history)
    np.savez(OUT/'validation_scores.npz',**saved)
    base=baseline_scores(v,va,ROOT/'experiments/round25/selected_bundle');old=round4_gates.apply_gate(v,base,'parse_and_repair',4)
    oldok=vy[np.arange(len(v)),old].astype(bool);assert oldok.sum()==752;hard=metadata(v)['hard']
    records=[dict(name='control',model=None,alpha=0.,correct=752,n=798,hard_correct=309,hard_n=355,repaired=0,damaged=0,net=0)];choices={'control':old}
    original=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original)
    for name,s in saved.items():
        for alpha in [.1,.25,.5]:
            key=f'{name}_a{alpha}';c=round4_gates.apply_gate(v,base+alpha*normalized(s),'parse_and_repair',4);ok=vy[np.arange(len(v)),c].astype(bool);choices[key]=c
            rec=dict(name=key,model=name,alpha=alpha,correct=int(ok.sum()),n=len(v),hard_correct=int(ok[hard].sum()),hard_n=int(hard.sum()),repaired=int((ok&~oldok).sum()),damaged=int((~ok&oldok).sum()),net=int(ok.sum()-752))
            records.append(rec);print('VAL',rec,flush=True)
    round4_gates.trace=original;winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['alpha']))
    dump(OUT/'selection.json',dict(selected=winner,records=records,test_opened=False));np.savez(OUT/'validation_choices.npz',**choices);print('SELECTED',winner,flush=True)
    if winner['model'] is None:return
    bundle=OUT/'selected_bundle';bundle.mkdir();shutil.copytree(ROOT/'experiments/round25/selected_bundle',bundle/'baseline');dest=bundle/'semantic';dest.mkdir()
    for f in ['config.json','model.npz','vocabulary.json']:shutil.copyfile(OUT/'models'/winner['model']/f,dest/f)
    dump(bundle/'config.json',dict(alpha=winner['alpha']))
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2,size
    dump(OUT/'asset_manifest.json',dict(bytes=size,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets}))


if __name__=='__main__':main()
