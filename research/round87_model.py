"""Local quantity language plus shared supplied-candidate semantic context."""
import numpy as np
from scipy import sparse
from scipy.special import softmax
DIM=8192

def local_matrix(x):
    z=x.tocoo();y=sparse.csr_matrix((z.data,(z.row,z.col%DIM)),shape=(x.shape[0],DIM),dtype=np.float32);y.sum_duplicates()
    norm=np.sqrt(np.asarray(y.multiply(y).sum(1)).ravel()).clip(1e-6);return sparse.diags(8/norm).dot(y).tocsr()

def context(arrays):
    e=np.asarray(arrays['embeddings'],np.float32);e=e/np.linalg.norm(e,axis=2,keepdims=True).clip(1e-8);q=e.mean(1);return q/np.linalg.norm(q,axis=1,keepdims=True).clip(1e-8)

def forward(x,ctx,w):
    def layernorm(a,prefix):return (a-a.mean(1,keepdims=True))/np.sqrt(a.var(1,keepdims=True)+1e-5)*w[prefix+'weight']+w[prefix+'bias']
    a=np.asarray(x@w['local_weight'].T)+w['local_bias'];a=layernorm(a,'local_norm_')
    c=np.clip((ctx-w['mean'])/w['scale'],-8,8) if bool(w['use_context']) else np.zeros_like(ctx)
    b=layernorm(c@w['context_weight'].T+w['context_bias'],'context_norm_')
    h=np.concatenate([a,b],1);h=np.maximum(h@w['hidden_weight'].T+w['hidden_bias'],0)
    return h@w['out_weight'].T+w['out_bias']

def probabilities(x,ctx,w):return softmax(forward(x,ctx,w),axis=1)
