"""CPU inference for the scratch text comparator; no Torch dependency."""
from pathlib import Path
import json
import numpy as np
from round241_text import encode

def softmax(x):
    z=np.exp(x-x.max(-1,keepdims=True))
    return z/z.sum(-1,keepdims=True)

def forward(data,w):
    def linear(x,name):return x@w[name+'.weight'].T+w[name+'.bias']
    def local(ids,numbers):
        x=w['embedding.weight'][ids]+linear(numbers,'number')
        padded=np.pad(x,((0,0),(1,1),(0,0)))
        h=x[:,::2]+w['conv.bias']
        for k in range(3):h=h+padded[:,k:k+len(x[0]):2]@w['conv.weight'][:,:,k].T
        mean=h.mean(-1,keepdims=True);variance=np.square(h-mean).mean(-1,keepdims=True)
        h=(h-mean)/np.sqrt(variance+1e-5)
        h=h*w['norm.weight']+w['norm.bias']
        return np.maximum(h,0),ids[:,::2]!=0
    def pool(h,mask):
        mean=(h*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1).astype(np.float32)
        maximum=np.where(mask[:,:,None],h,-1e4).max(1)
        return np.concatenate([mean,maximum],-1)
    n=len(data['q'])
    qh,qm=local(data['q'],data['qn'])
    ch,cm=local(data['c'].reshape(n*4,-1),data['cn'].reshape(n*4,-1,4))
    qh=np.repeat(qh,4,axis=0);qm=np.repeat(qm,4,axis=0)
    alignment=(qh@w['attention.weight'].T)@np.swapaxes(ch@w['attention.weight'].T,1,2)/4
    qc=softmax(np.where(cm[:,None,:],alignment,-1e4))@ch
    cq=softmax(np.where(qm[:,None,:],np.swapaxes(alignment,1,2),-1e4))@qh
    qcompare=np.maximum(linear(np.concatenate([qh,qc,qh-qc,qh*qc],-1),'compare'),0)
    ccompare=np.maximum(linear(np.concatenate([ch,cq,ch-cq,ch*cq],-1),'compare'),0)
    h=np.concatenate([pool(qcompare,qm),pool(ccompare,cm)],-1).reshape(n,4,-1)
    h=np.concatenate([h,h-h.mean(1,keepdims=True)],-1)
    return linear(np.maximum(linear(h,'head.0'),0),'head.3')[:,:,0]

def scores(bank,path):
    if not bank:return np.empty((0,4),np.float32)
    path=Path(path)
    vocab=json.loads((path/'vocab.json').read_text(encoding='utf-8'))
    with np.load(path/'model.npz') as archive:w={k:archive[k].astype(np.float32) for k in archive.files}
    data=encode(bank,vocab)
    result=[]
    for start in range(0,len(bank),32):
        result.append(forward({k:v[start:start+32] for k,v in data.items()},w))
    return np.concatenate(result).astype(np.float32)
