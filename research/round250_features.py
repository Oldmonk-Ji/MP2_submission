"""Scale-invariant and text-anchored correction evidence."""
import numpy as np
from round249_features import build as original,normalized,entropy

def build(bank,s,strict,text,old,arm):
    x,eligible,names=original(bank,s,strict,text,old,'top2')
    f={name:x[:,:,i] for i,name in enumerate(names)}
    n=len(bank)
    def repeat(a):return np.broadcast_to(a[:,None],(n,4))
    be=repeat(entropy(normalized(s)));ne=repeat(entropy(normalized(text)))
    if arm=='scale_free':
        keep=['base_normalized_delta','text_normalized_delta','strict_normalized_delta',
            'text_group_mean_delta','text_group_max_delta','base_group_mean_delta','answer_group_size_delta',
            'base_rank','old_base_rank','base_top','text_top','strict_top','old_not_raw_top',
            'base_top_gap','text_top_gap','text_strict_agree','text_strict_product','text_base_product',
            'gate_raw_alternative','text_group_strict_product','old_strict_top','old_parse_valid']
        return np.stack([f[k] for k in keep]+[be,ne],2).astype(np.float32),eligible,keep+['base_normalized_entropy','text_normalized_entropy']
    if arm!='text_anchored':raise ValueError(arm)
    strength=np.tanh(np.maximum(f['text_normalized_delta'],0)/2)
    contexts=[np.ones_like(strength),np.tanh(f['base_normalized_delta']),np.tanh(f['strict_normalized_delta']),
        f['text_top'],f['strict_top'],f['old_strict_top'],np.clip(f['base_top_gap']/3,0,1),
        np.clip(f['text_top_gap']/3,0,1),be,ne,f['answer_group_size_delta'],np.tanh(f['text_group_mean_delta'])]
    labels=['constant','base_relative_support','strict_relative_support','text_top','strict_top','old_strict_top',
            'base_gap','text_gap','base_entropy','text_entropy','answer_size_delta','text_group_support']
    return np.stack([strength*a for a in contexts],2).astype(np.float32),eligible,['text_strength_x_'+k for k in labels]

def choose(bank,s,strict,text,old,w,arm):
    if not bank:return np.empty(0,np.int64)
    x,eligible,_=build(bank,s,strict,text,old,arm)
    z=np.clip((x-w['mean'])/w['scale'],-6,6)
    utility=z@w['coef']+w['intercept']
    utility=np.where(eligible,utility,-np.inf)
    row=np.arange(len(bank));best=utility.argmax(1);result=np.asarray(old).copy()
    use=utility[row,best]>0;result[use]=best[use]
    return result
