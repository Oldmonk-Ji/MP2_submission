"""Bounded low-rank pair/context interaction in the same linear relation head."""
import numpy as np
from round133_context import question_context

def interaction(x,context,owners,w):
    left=(np.asarray(x@w['pair_pc'].T)-w['pair_mean'])/w['pair_std']
    right=((context-w['context_mean'])@w['context_pc'].T)/w['context_std']
    return np.einsum('ni,nj->nij',left,right[owners]).reshape(len(owners),w['pair_pc'].shape[0]*w['context_pc'].shape[0])*float(w['interaction_scale'])

def probabilities(x,context,owners,w):
    prior=((context-w['context_mean'])*float(w['context_scale']))@w['context_weight'].T
    cross=interaction(x,context,owners,w)@w['interaction_weight'].T
    s=np.asarray(x@w['weight'].T)+w['bias']+prior[owners]+cross
    s-=s.max(axis=1,keepdims=True);p=np.exp(s);return p/p.sum(axis=1,keepdims=True)

def checks():
    from scipy import sparse
    rng=np.random.default_rng(135135);x=sparse.csr_matrix(rng.normal(size=(3,7)));q=rng.normal(size=(2,5));owners=np.array([0,0,1]);w=dict(pair_pc=rng.normal(size=(2,7)),pair_mean=np.zeros(2),pair_std=np.ones(2),context_pc=rng.normal(size=(2,5)),context_std=np.ones(2),context_mean=np.zeros(5),context_scale=np.array(.5),interaction_scale=np.array(.3),weight=rng.normal(size=(6,7)),bias=rng.normal(size=6),context_weight=rng.normal(size=(6,5)),interaction_weight=rng.normal(size=(6,4)))
    z=interaction(x,q,owners,w);expected=np.array([np.outer(x[i].toarray()[0]@w['pair_pc'].T,q[o]@w['context_pc'].T).ravel()*.3 for i,o in enumerate(owners)])
    assert np.allclose(z,expected);p=probabilities(x,q,owners,w);assert np.allclose(p[:2],probabilities(x[:2],q[:1],np.array([0,0]),w));assert np.allclose(p.sum(1),1)
    assert probabilities(x[:0],q,np.array([],dtype=int),w).shape==(0,6)
    return dict(interaction_matches_explicit_outer_product=True,batch_invariance=True,probability_normalization=True,zero_quantity_pairs_supported=True)
