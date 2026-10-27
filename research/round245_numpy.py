"""NumPy-only inference for the scratch two-layer text Transformer."""
from pathlib import Path
import json
import numpy as np
from round245_text import encode,positions

def softmax(x):
    z=np.exp(x-x.max(-1,keepdims=True));return z/z.sum(-1,keepdims=True)

def forward(data,w):
    def linear(x,name):return x@w[name+'.weight'].T+w[name+'.bias']
    def norm(x,name):
        mean=x.mean(-1,keepdims=True);variance=np.square(x-mean).mean(-1,keepdims=True)
        return (x-mean)/np.sqrt(variance+1e-5)*w[name+'.weight']+w[name+'.bias']
    def local(ids,numbers):
        x=w['embedding.weight'][ids]+linear(numbers,'number')
        padded=np.pad(x,((0,0),(3,3),(0,0)))
        h=x[:,::4]+w['conv.bias']
        for k in range(7):h=h+padded[:,k:k+x.shape[1]:4]@w['conv.weight'][:,:,k].T
        return np.maximum(norm(h,'local_norm'),0),ids[:,::4]!=0
    n=len(data['q']);qh,qm=local(data['q'],data['qn'])
    ch,cm=local(data['c'].reshape(n*4,-1),data['cn'].reshape(n*4,-1,4))
    qh=np.repeat(qh,4,axis=0);qm=np.repeat(qm,4,axis=0)
    x=np.concatenate([qh+w['segment.weight'][0],ch+w['segment.weight'][1]],1)+positions()
    mask=np.concatenate([qm,cm],1)
    for layer in range(2):
        name=f'layers.{layer}'
        h=norm(x,name+'.norm1')
        qkv=h@w[name+'.self_attn.in_proj_weight'].T+w[name+'.self_attn.in_proj_bias']
        q,k,v=qkv.reshape(n*4,-1,3,4,12).transpose(2,0,3,1,4)
        logits=(q@np.swapaxes(k,-2,-1))/np.float32(np.sqrt(12))
        probabilities=softmax(np.where(mask[:,None,None,:],logits,-1e4))
        attended=(probabilities@v).transpose(0,2,1,3).reshape(n*4,-1,48)
        x=x+linear(attended,name+'.self_attn.out_proj')
        x=x+linear(np.maximum(linear(norm(x,name+'.norm2'),name+'.linear1'),0),name+'.linear2')
    x=norm(x,'final_norm')[:,qm.shape[1]:]
    mean=(x*cm[:,:,None]).sum(1)/np.maximum(cm.sum(1,keepdims=True),1).astype(np.float32)
    maximum=np.where(cm[:,:,None],x,-1e4).max(1)
    h=np.concatenate([mean,maximum],-1).reshape(n,4,96)
    h=np.concatenate([h,h-h.mean(1,keepdims=True)],-1)
    return linear(np.maximum(linear(h,'head.0'),0),'head.3')[:,:,0]

def weights(path):
    with np.load(Path(path)/'model.npz') as archive:w={k:archive[k].astype(np.float32) for k in archive.files}
    w['embedding.weight']*=w.pop('embedding_scale')[:,None]
    return w

def scores(bank,path):
    if not bank:return np.empty((0,4),np.float32)
    path=Path(path);w=weights(path)
    vocab=json.loads((path/'vocab.json').read_text(encoding='utf-8'));data=encode(bank,vocab)
    result=[]
    for start in range(0,len(bank),16):result.append(forward({k:v[start:start+16] for k,v in data.items()},w))
    return np.concatenate(result).astype(np.float32)
