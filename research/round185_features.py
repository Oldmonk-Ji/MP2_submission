"""Pure strict trace root features, no training cache or references."""
import numpy as np
from round180_features import programs
from round107_latent import root_features,predict
from round103_features import matrix

def build(bank):
    features=[];counts=[]
    for row in bank:
        ff,cc=root_features(row,programs(row));features.extend(ff);counts.extend(cc)
    return matrix(features),np.r_[0,np.cumsum(counts)].astype(np.int64)
