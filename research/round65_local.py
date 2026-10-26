"""Whole program context under exactly the same relation supervision."""
import numpy as np
from round46_graph import graph,features,csr,LIMIT,DIM
from round48_local import node_scores

def focus(pair):
    q=pair['question'];qs=graph(q)['quantities']
    return features(q,qs,pair['positive']),features(q,qs,pair['negative']),[]

def packet_matrix(rows,packets):
    ds=[];ptr=[0]
    for r,p in zip(rows,packets):
        for j in range(LIMIT):
            if j<len(p['hypotheses']):ds.append(features(r['question'],p['quantities'],p['hypotheses'][j]['program']))
            ptr.append(len(ds))
    return csr(ds),np.asarray(ptr,np.int64)
