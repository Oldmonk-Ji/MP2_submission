"""Monotonic, dimensionless text/old-model corroboration features."""
import numpy as np
from round249_features import build as original

def build(bank,s,strict,text,old,arm):
    x,eligible,names=original(bank,s,strict,text,old,'top2')
    f={k:x[:,:,i] for i,k in enumerate(names)}
    nn=f['text_normalized_delta'];base=f['base_normalized_delta'];verification=f['strict_normalized_delta']
    group=f['text_group_mean_delta']
    if arm=='monotone_linear':
        values=[base,nn,verification,group]
        labels=['base_relative_advantage','text_relative_support','strict_relative_support','text_group_support']
    elif arm=='monotone_text_gate':
        strength=np.tanh(np.maximum(nn,0)/2)
        values=[strength,strength*base,strength*verification,strength*group]
        labels=['text_strength','text_x_base','text_x_strict','text_x_group']
    else:raise ValueError(arm)
    return np.stack(values,2).astype(np.float32),eligible,labels

def choose(bank,s,strict,text,old,w,arm):
    if not bank:return np.empty(0,np.int64)
    x,eligible,_=build(bank,s,strict,text,old,arm)
    utility=(x/w['scale'])@w['coef']+w['intercept']
    utility=np.where(eligible,utility,-np.inf)
    row=np.arange(len(bank));best=utility.argmax(1);result=np.asarray(old).copy()
    use=utility[row,best]>0;result[use]=best[use]
    return result
