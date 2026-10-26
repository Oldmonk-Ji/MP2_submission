"""Matched pointer model plus verified wrong-program contrast supervision."""
from data_access import ROOT
from round22_common import read,dump
from round29_programs import vocabulary,encode_questions,encode_programs
from round29_model import trim,save
from round30_model import ProgramModel
from round29_train import evaluate
import numpy as np
import torch
from torch import nn
import json,sys,time

OUT=ROOT/'experiments/round31'


def main(stage):
    torch.set_num_threads(4);torch.manual_seed(29029);rng=np.random.default_rng(29029)
    rows=[json.loads(s) for s in open(OUT/'training_programs.jsonl',encoding='utf-8')];y=np.asarray([r['labels'] for r in rows],np.float32)
    ids=np.arange(len(rows));held=None
    if stage=='diagnostic':
        with np.load(OUT/'diagnostic_split.npz') as z:ids=z['fit'];held=z['held']
    path=OUT/stage/'semantic_contrast';path.mkdir(parents=True);assert not (path/'complete.json').exists()
    bank=[dict(question=r['question']) for r in rows];vocab=vocabulary([bank[i] for i in ids])
    q=encode_questions(bank,vocab,max_length=256);ref=encode_programs([r['reference'] for r in rows])
    programs=encode_programs([c for r in rows for c in r['candidates']]).reshape(len(rows),4,-1)
    negatives=encode_programs([c for r in rows for c in r['contrast_negatives']]).reshape(len(rows),3,-1)
    cm=np.asarray([r['contrast_mask'] for r in rows],np.float32)
    usable=np.asarray([[c['clean'] for c in r['candidates']] for r in rows])
    weights=np.asarray([1. if r['reference']['clean'] else (.25 if r['reference']['informative'] else .05) for r in rows],np.float32)
    dump(path/'vocabulary.json',vocab);dump(path/'training_ids.json',[rows[i]['train_index'] for i in ids])
    device='cuda' if torch.cuda.is_available() else 'cpu';model=ProgramModel(len(vocab)).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01);history=[];tick=time.perf_counter()
    for epoch in range(1,(40 if stage=='diagnostic' else 80)+1):
        model.train();total=0.;count=0
        for ix in np.array_split(rng.permutation(ids),int(np.ceil(len(ids)/64))):
            qt=torch.as_tensor(trim(q[ix]),device=device);rt=torch.as_tensor(trim(ref[ix]),device=device);memory=model.encode(qt)
            nll,positive,_=model(qt,rt,memory);w=torch.as_tensor(weights[ix],device=device)
            loss=(nll*w).sum()/w.sum()
            cp=programs[ix];cp=torch.as_tensor(trim(cp.reshape(-1,cp.shape[-1])),device=device)
            _,cs,_=model(qt,cp,memory);cs=cs.reshape(-1,4);target=torch.as_tensor(y[ix],device=device)
            loss+=.5*(torch.logsumexp(cs,1)-torch.logsumexp(cs.masked_fill(target<.5,-1e4),1)).mean()
            neg=negatives[ix];neg=torch.as_tensor(trim(neg.reshape(-1,neg.shape[-1])),device=device)
            _,ns,_=model(qt,neg,memory);ns=ns.reshape(-1,3);mask=torch.as_tensor(cm[ix],device=device)
            contrast=(nn.functional.softplus(.5+ns-positive[:,None])*mask).sum()/mask.sum().clamp_min(1)
            loss+=.5*contrast
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            total+=float(loss.detach())*len(ix);count+=len(ix)
        rec=dict(epoch=epoch,training_loss=total/count,seconds=time.perf_counter()-tick)
        if held is not None and epoch in (5,15,25,40):rec['diagnostic']=evaluate(model,q[held],ref[held],programs[held],y[held],usable[held],weights[held],device)
        if epoch%5==0:print(stage,rec,flush=True)
        history.append(rec);dump(path/'history.json',history)
        if epoch in (15,40,80):
            dest=path/f'epoch{epoch}';dest.mkdir();save(model,dest/'model.npz');dump(dest/'vocabulary.json',vocab)
            dump(dest/'config.json',dict(variant='semantic_contrast',epochs=epoch,training_n=len(ids),seed=29029))
    dump(path/'complete.json',dict(n=len(ids),seconds=time.perf_counter()-tick,device=device,parameters=sum(p.numel() for p in model.parameters()),test_opened=False))


if __name__=='__main__':main(sys.argv[1])
