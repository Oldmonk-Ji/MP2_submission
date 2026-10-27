"""Use learned neural quantity alignment to qualify complementary corrections."""
import numpy as np
from round249_features import build as original
from round253_attention import NAMES

def build(bank,s,strict,text,attention,old,arm):
    x,eligible,names=original(bank,s,strict,text,old,'top2')
    keep=['base_normalized_delta','text_normalized_delta','strict_normalized_delta','text_group_mean_delta']
    core=x[:,:,[names.index(k) for k in keep]]
    delta=attention-attention[np.arange(len(bank)),old,None,:]
    if arm=='alignment_utility':
        result=np.concatenate([core,delta],2);labels=keep+['alignment_delta_'+k for k in NAMES]
    elif arm=='alignment_text_anchored':
        strength=np.tanh(np.maximum(core[:,:,1],0)/2)[:,:,None]
        result=strength*np.concatenate([np.ones_like(core[:,:,:1]),core[:,:,[0,2,3]],delta],2)
        labels=['text_strength']+['text_strength_x_'+k for k in [keep[0],keep[2],keep[3]]+NAMES]
    else:raise ValueError(arm)
    return result.astype(np.float32),eligible,labels

def choose(bank,s,strict,text,attention,old,w,arm):
    if not bank:return np.empty(0,np.int64)
    x,eligible,_=build(bank,s,strict,text,attention,old,arm)
    z=np.clip((x-w['mean'])/w['scale'],-6,6)
    utility=np.where(eligible,z@w['coef']+w['intercept'],-np.inf)
    row=np.arange(len(bank));best=utility.argmax(1);result=np.asarray(old).copy()
    use=utility[row,best]>0;result[use]=best[use]
    return result
