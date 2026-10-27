"""Coordinate-invariant learned attention grounding from the existing text net."""
from pathlib import Path
import json
import numpy as np
from round241_text import encode
from round241_numpy import forward,softmax

NAMES=['question_same_quantity_mass','question_min_same_quantity_mass','question_wrong_quantity_mass',
       'question_quantity_support','question_quantity_top_match','candidate_same_quantity_mass',
       'candidate_min_same_quantity_mass','candidate_wrong_quantity_mass','candidate_quantity_support',
       'candidate_quantity_top_match','question_quantity_entropy','candidate_quantity_entropy',
       'target_to_conclusion_mass','target_entropy','new_number_to_question_quantity_mass',
       'new_number_entropy','question_quantity_attention_peak','candidate_quantity_attention_peak',
       'question_word_match_mass','target_word_match_mass']

def attention_features(data,w):
    def local(ids,numbers):
        x=w['embedding.weight'][ids]+numbers@w['number.weight'].T+w['number.bias']
        padded=np.pad(x,((0,0),(1,1),(0,0)));h=x[:,::2]+w['conv.bias']
        for k in range(3):h=h+padded[:,k:k+x.shape[1]:2]@w['conv.weight'][:,:,k].T
        mean=h.mean(-1,keepdims=True);var=np.square(h-mean).mean(-1,keepdims=True)
        h=(h-mean)/np.sqrt(var+1e-5)*w['norm.weight']+w['norm.bias']
        return np.maximum(h,0),ids[:,::2]!=0
    def variables(ids):
        padded=np.pad(ids,((0,0),(1,1)))
        windows=np.stack([padded[:,k:k+ids.shape[1]:2] for k in range(3)],2)
        known=(windows[:,:,:,None]==np.arange(4,20)[None,None,None,:]).any(2)
        new=(windows==2).any(2)
        return known,new
    def avg(a,mask):return (a*mask).sum(1)/np.maximum(mask.sum(1),1).astype(np.float32)
    def minimum(a,mask):return np.where(mask.any(1),np.where(mask,a,np.inf).min(1),0)
    def ent(p,keys):return -(p*np.log(np.maximum(p,1e-9))).sum(2)/np.log(np.maximum(keys.sum(1),2).astype(np.float32))[:,None]
    n=len(data['q']);qids=data['q'];cids=data['c'].reshape(n*4,-1)
    qh,qm=local(qids,data['qn']);ch,cm=local(cids,data['cn'].reshape(n*4,-1,4))
    qvar,_=variables(qids);cvar,cnew=variables(cids)
    qh=np.repeat(qh,4,0);qm=np.repeat(qm,4,0);qvar=np.repeat(qvar,4,0)
    qknown=qvar.any(2)&qm;cknown=cvar.any(2)&cm;cnew&=cm
    same=(qvar.astype(np.float32)@np.swapaxes(cvar.astype(np.float32),1,2))>0
    align=(qh@w['attention.weight'].T)@np.swapaxes(ch@w['attention.weight'].T,1,2)/4
    qc=softmax(np.where(cm[:,None,:],align,-1e4));cq=softmax(np.where(qm[:,None,:],align.swapaxes(1,2),-1e4))
    qs=(qc*same).sum(2);cs=(cq*same.swapaxes(1,2)).sum(2)
    qw=(qc*(cknown[:,None,:]&~same)).sum(2);cw=(cq*(qknown[:,None,:]&~same.swapaxes(1,2))).sum(2)
    qt=np.take_along_axis(same,qc.argmax(2)[:,:,None],2)[:,:,0]
    ct=np.take_along_axis(same.swapaxes(1,2),cq.argmax(2)[:,:,None],2)[:,:,0]
    qe,ce=ent(qc,cm),ent(cq,qm)
    qtail=qm&(np.arange(qm.shape[1])[None,:]>=qm.sum(1)[:,None]-6)
    ctail=cm&(np.arange(cm.shape[1])[None,:]>=cm.sum(1)[:,None]-8)
    qcenter=np.repeat(qids[:,::2],4,0);ccenter=cids[:,::2]
    words=(qcenter[:,:,None]==ccenter[:,None,:])&(qcenter[:,:,None]>=20)&(ccenter[:,None,:]>=20)
    word_mass=(qc*words).sum(2);qword=(qcenter>=20)&qm
    values=[avg(qs,qknown),minimum(qs,qknown),avg(qw,qknown),avg(qs-qw,qknown),avg(qt,qknown),
            avg(cs,cknown),minimum(cs,cknown),avg(cw,cknown),avg(cs-cw,cknown),avg(ct,cknown),
            avg(qe,qknown),avg(ce,cknown),avg((qc*ctail[:,None,:]).sum(2),qtail),avg(qe,qtail),
            avg((cq*qknown[:,None,:]).sum(2),cnew),avg(ce,cnew),avg(qc.max(2),qknown),avg(cq.max(2),cknown),
            avg(word_mass,qword),avg(word_mass,qword&qtail)]
    result=np.stack(values,1).reshape(n,4,len(NAMES)).astype(np.float32)
    assert np.isfinite(result).all()
    return result

def scores_and_features(bank,path):
    if not bank:return np.empty((0,4),np.float32),np.empty((0,4,len(NAMES)),np.float32)
    path=Path(path);vocab=json.loads((path/'vocab.json').read_text(encoding='utf-8'))
    with np.load(path/'model.npz') as archive:w={k:archive[k].astype(np.float32) for k in archive.files}
    data=encode(bank,vocab);raw=[];features=[]
    for start in range(0,len(bank),32):
        batch={k:v[start:start+32] for k,v in data.items()}
        raw.append(forward(batch,w));features.append(attention_features(batch,w))
    return np.concatenate(raw).astype(np.float32),np.concatenate(features)
