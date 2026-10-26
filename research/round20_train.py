"""Matched full-data, quality-filtered and random-removal training runs."""
from round20_common import OUT,plan,load_part,subset,read,dump,sha,normalized,paired
from round20_models import fit_baseline,fit_neural,fit_kernel
from round20_quality import random_matched
from features import metadata
import round4_gates
import argparse,json,time
import numpy as np


def train_variant(name):
    assert not (OUT/'frozen.json').exists()
    path=OUT/'runs'/name;path.mkdir(parents=True,exist_ok=True)
    if (path/'done.json').exists():return read(path/'done.json')
    dump(path/'fusion.json',dict(alpha=plan()['alpha'],gate='parse_and_repair',margin_limit=4))
    b,a,y,ids=load_part('fit');d,da,dy,dids=load_part('development')
    mask=read(OUT/f'quality/{name}.json') if name!='full' else dict(name='full',fit_ids=ids.tolist(),removed_ids=[],removed_fraction=0.)
    keep=np.flatnonzero(np.isin(ids,mask['fit_ids']))
    assert set(mask['fit_ids'])|set(mask['removed_ids'])==set(ids.tolist())
    assert not set(mask['fit_ids'])&set(mask['removed_ids'])
    assert len(ids)-len(keep)<=int(len(ids)*.1)
    stamp=dict(name=name,fit_ids=ids[keep].tolist(),removed_ids=mask['removed_ids'],plan_sha=sha(OUT/'plan.json'))
    if (path/'training_subset.json').exists():assert read(path/'training_subset.json')==stamp
    else:dump(path/'training_subset.json',stamp)
    b,a,y=subset(b,a,y,keep);start=time.perf_counter()
    base=fit_baseline(b,a,y,d,da,dy,path/'baseline')
    long=fit_neural(b,a,y,d,da,dy,path/'long')
    kernel=fit_kernel(b,a,y,d,da,path/'kernel')
    with np.load(path/'baseline/development.npz') as z:bs=z['scores']
    with np.load(path/'long/development.npz') as z:ls=z['scores']
    with np.load(path/'kernel/development.npz') as z:ks=z['scores']
    scores=dict(baseline=bs,long=bs+plan()['alpha']*normalized(ls),kernel=bs+plan()['alpha']*normalized(ks))
    saved=dict(ids=dids,labels=dy);result={};meta=metadata(d)
    for head,s in scores.items():
        c=round4_gates.apply_gate(d,s,'parse_and_repair',4);saved[head+'_scores']=s;saved[head+'_choice']=c
        result[head]=paired(c,c,dy,meta['hard'])
    np.savez(path/'development.npz',**saved)
    outcome=dict(name=name,n_fit=len(b),removed=len(mask['removed_ids']),removed_fraction=len(mask['removed_ids'])/len(ids),
        development=result,models=dict(baseline=base,long=long,kernel=kernel),seconds=time.perf_counter()-start,
        total_inference_bytes=dict(long=base['inference_asset_bytes']+long['inference_asset_bytes']+(path/'fusion.json').stat().st_size,
                                   kernel=base['inference_asset_bytes']+kernel['inference_asset_bytes']+(path/'fusion.json').stat().st_size))
    dump(path/'done.json',outcome)
    print('TRAINED',name,'retained',len(b),'development',{k:v['correct'] for k,v in result.items()},flush=True)
    return outcome


def select():
    assert not (OUT/'selection.json').exists()
    d,_,dy,_=load_part('development',embeddings=False);hard=metadata(d)['hard'];records=[]
    with np.load(OUT/'runs/full/development.npz') as z:full=z['long_choice']
    for name in ('evidence_only','quality_05','quality_10'):
        r=read(OUT/f'runs/{name}/done.json')
        with np.load(OUT/f'runs/{name}/development.npz') as z:c=z['long_choice']
        records.append(dict(name=name,removed=r['removed'],**paired(c,full,dy,hard)))
    admissible=[r for r in records if r['removed']>0]
    assert admissible
    chosen=max(admissible,key=lambda r:(r['correct'],-r['damaged'],-r['removed']))
    result=dict(primary=chosen,records=records,head='long',alpha=plan()['alpha'],
        primary_selected_before_random_control=True,no_audit_access=True,
        rule='Development correct, fewer harms, smaller deletion count; retain all variants; audit result cannot switch primary.')
    dump(OUT/'selection.json',result)
    random_matched(chosen['name'])
    print('SELECTED',json.dumps(chosen),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['full','quality','select','random']);a=p.parse_args()
    if a.stage=='full':train_variant('full')
    elif a.stage=='quality':
        for name in ('evidence_only','quality_05','quality_10'):train_variant(name)
    elif a.stage=='random':train_variant('random_matched')
    else:select()
