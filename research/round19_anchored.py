"""Input-only surface identity guides learned local comparison; not a semantic rule."""
from round19_model import Comparator, encode_bank, tokens, NUM, pool
import hashlib, math
import numpy as np
import torch

STOP=set('a an the we is are was were be been being and or to of in on at with let s as it this that these those i you they he she his her their our its have has had do does did will would can could should may might'.split())


def keys(text,length):
    ts=tokens(text)
    if len(ts)>length:ts=ts[:length//2]+ts[-(length-length//2):]
    out=np.zeros(length,np.int64)
    for i,t in enumerate(ts):
        if NUM.fullmatch(t):
            from decimal import Decimal
            key='number:'+str(Decimal(t.replace(',','')).normalize())
        elif len(t)>1 and t.isalpha() and t not in STOP:key='word:'+t
        else:continue
        out[i]=int.from_bytes(hashlib.blake2b(key.encode(),digest_size=8).digest(),'little')&((1<<63)-1)
    return out


def encode_anchored(bank,arrays,vocab,cfg):
    x,coverage=encode_bank(bank,arrays,vocab,cfg)
    x['qk']=np.asarray([keys(r['question'],cfg['question_tokens']) for r in bank])
    x['ck']=np.asarray([[keys(c['text'],cfg['candidate_tokens']) for c in r['candidates']] for r in bank])
    return x,coverage


class AnchoredComparator(Comparator):
    def forward(self,q,qn,c,cn,e,qk,ck):
        n,_,length=c.shape;qx,qm=self.encode(q,qn)
        cx,cm=self.encode(c.reshape(n*4,length),cn.reshape(n*4,length,5))
        qx=qx.repeat_interleave(4,0);qm=qm.repeat_interleave(4,0);qkey=qk.repeat_interleave(4,0)
        ckey=ck.reshape(n*4,length)
        exact_q=(ckey[:,:,None]==qkey[:,None,:])&ckey[:,:,None].ne(0)
        aq=self.align_q(cx)@self.align_q(qx).transpose(-1,-2)/math.sqrt(self.cfg['width'])
        aq=aq+self.cfg['anchor_bias']*exact_q
        q_context=aq.masked_fill(~qm[:,None,:],-1e4).softmax(-1)@qx
        conditioned=self.question_compare(torch.cat([cx,q_context,cx-q_context,cx*q_context],-1))
        own=torch.cat([pool(cx,cm),pool(conditioned,cm),self.emb_project(((e-self.mean)/self.scale).clamp(-10,10)).reshape(n*4,-1)],-1).reshape(n,4,-1)
        h=conditioned.reshape(n,4,length,-1);cm=cm.reshape(n,4,length)
        a,b=h[:,self.left],h[:,self.right];ma,mb=cm[:,self.left],cm[:,self.right]
        if self.cfg['mode']=='joint':
            ak,bk=ck[:,self.left],ck[:,self.right]
            exact=(ak[...,None]==bk[...,None,:])&ak[...,None].ne(0)
            alignment=self.align_c(a)@self.align_c(b).transpose(-1,-2)/math.sqrt(self.cfg['width'])
            alignment=alignment+self.cfg['anchor_bias']*exact
            ab=alignment.masked_fill(~mb[...,None,:],-1e4).softmax(-1)@b
            ba=alignment.transpose(-1,-2).masked_fill(~ma[...,None,:],-1e4).softmax(-1)@a
        else:ab,ba=torch.zeros_like(a),torch.zeros_like(b)
        fa=self.other_compare(torch.cat([a,ab,a-ab,a*ab],-1));fb=self.other_compare(torch.cat([b,ba,b-ba,b*ba],-1))
        ua=self.utility(torch.cat([own[:,self.left],pool(fa,ma)],-1)).squeeze(-1)
        ub=self.utility(torch.cat([own[:,self.right],pool(fb,mb)],-1)).squeeze(-1)
        margin=ua-ub
        return margin@self.incidence,margin
