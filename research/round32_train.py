from data_access import ROOT
from round22_common import read,dump,sha
from round29_programs import encode_questions
from round29_model import trim,save
from round32_graph import GraphModel,encode_graphs,predict_encoded
import numpy as np
import torch
from torch import nn
import json,sys,time,datetime

OUT=ROOT/'experiments/round32'


def main(stage,anonymize=False):
    OUT.mkdir(exist_ok=True)
    if not (OUT/'plan.json').exists():dump(OUT/'plan.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_n=4743,validation_n=798,test_opened=False,seed=32032,epochs=[10,25,50],
        method='Discriminative contextual roles/LCA operations; encoder initialized only from locally scratch-trained official TRAIN semantic model',
        auxiliary='Reference graph classes plus actual candidate final-answer supervision. Unknown roles excluded rather than called unused.',
        diagnostic='Same grouped 3795/948 TRAIN split; initialize from corresponding diagnostic encoder, never full encoder',
        asset='Only word encoder and small relation heads retained; generator decoder removed',
        anonymize_entities=anonymize,source_hash=sha(ROOT/'experiments/round29/training_programs.jsonl')))
    torch.set_num_threads(4);torch.manual_seed(32032);rng=np.random.default_rng(32032)
    rows=[json.loads(s) for s in open(ROOT/'experiments/round29/training_programs.jsonl',encoding='utf-8')]
    y=np.asarray([r['labels'] for r in rows],np.float32);ids=np.arange(len(rows));held=None
    if stage=='diagnostic':
        with np.load(ROOT/'experiments/round29/diagnostic_split.npz') as z:ids=z['fit'];held=z['held']
    source=ROOT/f'experiments/round29/{stage}/reference_only/epoch15';vocab=read(source/'vocabulary.json')
    encoder=encode_questions
    if anonymize:
        from round33_text import encode_questions as encoder
        for token in [f'entity{i}' for i in range(16)]+['possessive']:
            if token not in vocab:vocab[token]=len(vocab)
    q=encoder(rows,vocab,max_length=256);questions=[r['question'] for r in rows]
    rr,rp=encode_graphs([r['reference'] for r in rows],questions,strict=True)
    cr,cp=encode_graphs([c for r in rows for c in r['candidates']],[r['question'] for r in rows for c in r['candidates']])
    cr=cr.reshape(len(rows),4,24);cp=cp.reshape(len(rows),4,24,24)
    model=GraphModel(len(vocab));state=model.state_dict()
    with np.load(source/'model.npz') as z:
        for k in list(state):
            if k=='words.weight':state[k][:len(z[k])]=torch.from_numpy(z[k].copy())
            elif k.startswith('encoder.'):state[k]=torch.from_numpy(z[k].copy())
    for name,target,n in [('role_prior',rr[ids],3),('pair_prior',rp[ids],6)]:
        count=np.bincount(target[target>=0],minlength=n)+1;state[name]=torch.from_numpy((count/count.sum()).astype(np.float32))
    model.load_state_dict(state);device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
    path=OUT/stage;path.mkdir();dump(path/'training_ids.json',[rows[i]['train_index'] for i in ids]);dump(path/'vocabulary.json',vocab)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03);history=[];tick=time.perf_counter()
    def supervised(logp,target):
        mask=target.ge(0);loss=-logp.gather(-1,target.clamp_min(0)[...,None]).squeeze(-1)*mask
        dims=tuple(range(1,target.ndim));per=loss.sum(dims)/mask.sum(dims).clamp_min(1);valid=mask.reshape(len(mask),-1).any(1)
        return per[valid].mean() if valid.any() else logp.sum()*0
    for epoch in range(1,51):
        model.train();total=0.
        for ix in np.array_split(rng.permutation(ids),int(np.ceil(len(ids)/48))):
            qt=torch.as_tensor(trim(q[ix]),device=device);logp=model(qt)
            gr=torch.as_tensor(rr[ix],device=device);gp=torch.as_tensor(rp[ix],device=device)
            loss=supervised(logp[0],gr)+supervised(logp[1],gp)
            sr=torch.as_tensor(cr[ix],device=device);sp=torch.as_tensor(cp[ix],device=device);s=model.score(logp,sr,sp)
            target=torch.as_tensor(y[ix],device=device)
            group=(torch.logsumexp(s,1)-torch.logsumexp(s.masked_fill(target<.5,-1e4),1)).mean()
            loss+=.5*group+.1*nn.functional.binary_cross_entropy_with_logits(s,target)
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();total+=float(loss.detach())*len(ix)
        rec=dict(epoch=epoch,loss=total/len(ids),seconds=time.perf_counter()-tick)
        if epoch in [10,25,50]:
            dest=path/f'epoch{epoch}';dest.mkdir();save(model,dest/'model.npz');dump(dest/'vocabulary.json',vocab)
            dump(dest/'config.json',dict(epochs=epoch,training_n=len(ids),seed=32032,initialization=str(source.relative_to(ROOT))))
            if held is not None:
                s=predict_encoded(model,q[held],cr[held],cp[held],device);choice=s.argmax(1)
                available=(cr[held]>=0).any(2)|(cp[held]>=0).any((2,3))
                eligible=(available.sum(1)>=2)&((y[held]*available).sum(1)>0)&(y[held].sum(1)<4)
                rec['diagnostic']=dict(correct=int(y[held,np.asarray(choice)].sum()),n=len(held),
                    eligible_n=int(eligible.sum()),eligible_correct=int(y[held,np.asarray(choice)][eligible].sum()))
        if epoch%5==0:print(stage,rec,flush=True)
        history.append(rec);dump(path/'history.json',history)
    dump(path/'complete.json',dict(n=len(ids),seconds=time.perf_counter()-tick,parameters=sum(p.numel() for p in model.parameters()),test_opened=False))


if __name__=='__main__':main(sys.argv[1])
