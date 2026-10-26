"""Fit two preregistered comparators; select on TRAIN development only."""
from round18_common import OUT, ROOT, load_part, read_plan, normalized, paired, sha
import json, time, functools
import numpy as np
import torch
from round18_model import (Comparator, vocabulary, encode_bank, tensor_batch,
                           objective, predict_data, save_model)
from features import metadata
import round4_gates
round4_gates.trace = functools.lru_cache(maxsize=8192)(round4_gates.trace)


def candidates(scores, bank, y, meta, baseline, base_choice, alphas):
    result=[]
    for alpha in alphas:
        # Preserve the historical gate's score scale and margin limit.
        fused=baseline+alpha*normalized(scores)
        choice=round4_gates.apply_gate(bank,fused,'parse_and_repair',4)
        result.append(dict(alpha=alpha,**paired(choice,base_choice,y,meta['hard'])))
    return result


def main():
    assert not (OUT/'frozen.json').exists(), 'Cannot tune after audit unsealing.'
    plan=read_plan(); cfg=plan['comparator']
    assert (OUT/'baseline/done.json').exists()
    b,a,y,ids=load_part('fit'); d,da,dy,dev=load_part('development'); dm=metadata(d)
    vpath=OUT/'vocabulary.json'
    vocab=vocabulary(b,cfg['vocab_limit'])
    if vpath.exists(): assert json.loads(vpath.read_text(encoding='utf-8'))==vocab
    else: vpath.write_text(json.dumps(vocab,ensure_ascii=False),encoding='utf-8')
    x,coverage=encode_bank(b,a,vocab,cfg); dx,devcoverage=encode_bank(d,da,vocab,cfg)
    with np.load(OUT/'baseline/development.npz') as z:bs=z['scores'];bc=z['choices']
    mixed=np.flatnonzero((y.sum(1)>0)&(y.sum(1)<4))
    mean=a['embeddings'].reshape(-1,1024).mean(0)
    scale=a['embeddings'].reshape(-1,1024).std(0).clip(.001)
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    specs=dict(plan_sha=sha(OUT/'plan.json'),fit_questions=len(b),development_questions=len(d),
               mixed_fit_questions=len(mixed),positive_negative_fit_pairs=int(sum(int(r.sum())*int(4-r.sum()) for r in y)),
               tokenizer='fit-only words and punctuation; numeric value features; fixed head/tail truncation',
               architecture='shared random word/numeric/position encoder + convolution; Q-to-candidate attention; candidate-to-candidate attention; shared antisymmetric pair utilities',
               loss='within-question opposite-label logistic + 0.25 correct-set logsumexp; all six pairs; no positive-positive negative labels',
               ablation='same layers and seed, set other-candidate contexts to zero',
               official_embeddings='input features; trainable projection and all network weights start random',
               epoch_selection=cfg['selection'],fit_coverage=coverage,development_coverage=devcoverage,
               device=device,gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
               reference_steps_used=False,synthetic_used=False,audit_used=False,validation_used=False,test_used=False)
    specpath=OUT/'architecture.json'
    if specpath.exists():assert json.loads(specpath.read_text())==specs
    else:specpath.write_text(json.dumps(specs,indent=2))
    print(json.dumps(specs),flush=True)
    for mode in cfg['modes']:
        path=OUT/mode;path.mkdir(exist_ok=True)
        if (path/'selected.json').exists():print('SKIP COMPLETE',mode,flush=True);continue
        torch.manual_seed(plan['seed']);rng=np.random.default_rng(plan['seed'])
        cc=dict(cfg,mode=mode,seed=plan['seed'],plan_sha=sha(OUT/'plan.json'))
        model=Comparator(cc,len(vocab),mean,scale).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        (path/'config.json').write_text(json.dumps(cc,indent=2))
        history=[];best=None;best_key=None;last_best=0;start=time.perf_counter()
        for epoch in range(1,cfg['epochs']+1):
            model.train();total=0.;n=0;tick=time.perf_counter()
            order=rng.permutation(mixed)
            for startidx in range(0,len(order),cfg['batch']):
                ix=order[startidx:startidx+cfg['batch']]
                target=torch.as_tensor(y[ix],device=device)
                scores,margin=model(**tensor_batch(x,ix,device))
                loss=objective(scores,margin,target)
                opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
                total+=float(loss.detach())*len(ix);n+=len(ix)
            scores,margin=predict_data(model,dx,device,cfg['batch'])
            with torch.no_grad():
                loss_dev=float(objective(torch.tensor(scores),torch.tensor(margin),torch.tensor(dy)))
            variants=candidates(scores,d,dy,dm,bs,bc,cfg['alpha'])
            sel=max(variants,key=lambda v:(v['correct'],-v['changed'],-v['alpha']))
            key=(sel['correct'],-sel['changed'],-sel['alpha'],-loss_dev)
            standalone=round4_gates.apply_gate(d,scores,'parse_and_repair',4)
            rec=dict(epoch=epoch,train_loss=total/n,development_loss=loss_dev,selected=sel,
                     standalone=paired(standalone,bc,dy,dm['hard']),variants=variants,seconds=time.perf_counter()-tick)
            history.append(rec)
            if best_key is None or key>best_key:
                best_key=key;last_best=epoch;best=rec
                save_model(model,path);np.savez(path/'development.npz',scores=scores,margins=margin)
            (path/'history.json').write_text(json.dumps(history,indent=2))
            print(mode,'epoch',epoch,'loss',round(total/n,4),'dev',sel['correct'],
                  'repair/harm/net',sel['repaired'],sel['damaged'],sel['net'],'alpha',sel['alpha'],
                  'standalone',rec['standalone']['correct'],'sec',round(rec['seconds'],1),flush=True)
            if epoch-last_best>=cfg['patience']:break
        selected=dict(best=best,epochs_run=len(history),fit_seconds=time.perf_counter()-start,
                      parameters=sum(p.numel() for p in model.parameters()),model_bytes=(path/'model.npz').stat().st_size,
                      plan_sha=sha(OUT/'plan.json'),vocabulary_sha=sha(vpath),no_audit_selection=True)
        (path/'selected.json').write_text(json.dumps(selected,indent=2))
        print('SELECTED',mode,best['epoch'],best['selected'],flush=True)


if __name__=='__main__':main()
