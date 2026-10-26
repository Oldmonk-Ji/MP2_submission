"""Supplied semantic context for quantity-pair relation probabilities."""
import numpy as np

def question_context(arrays):
    # Sum float32 inputs in float64 so candidate reordering cannot change the
    # effective context through float32 accumulation noise.
    return np.asarray(arrays['embeddings']).mean(axis=1,dtype=np.float64)

def probabilities(x,context,owners,w):
    semantic=(context-w['context_mean'])*float(w['context_scale'])
    prior=semantic@w['context_weight'].T
    s=np.asarray(x@w['weight'].T)+w['bias']+prior[owners]
    s-=s.max(axis=1,keepdims=True);p=np.exp(s);return p/p.sum(axis=1,keepdims=True)

def checks():
    from scipy import sparse
    rng=np.random.default_rng(133133);e=rng.normal(size=(2,4,5)).astype(np.float32);q=question_context(dict(embeddings=e));qq=question_context(dict(embeddings=e[:,[3,1,0,2]]));assert np.array_equal(q,qq)
    x=sparse.csr_matrix(rng.normal(size=(3,7)));owners=np.array([0,0,1]);w=dict(weight=rng.normal(size=(6,7)),bias=rng.normal(size=6),context_mean=rng.normal(size=5),context_scale=np.array(2.),context_weight=rng.normal(size=(6,5)))
    p=probabilities(x,q,owners,w);assert np.allclose(p.sum(1),1)
    joined=np.concatenate([x.toarray(),((q-w['context_mean'])*2)[owners]],1);s=joined@np.concatenate([w['weight'],w['context_weight']],1).T+w['bias'];s-=s.max(1,keepdims=True);expected=np.exp(s)/np.exp(s).sum(1,keepdims=True);assert np.allclose(p,expected)
    assert np.allclose(p[:2],probabilities(x[:2],q[:1],np.array([0,0]),w))
    return dict(candidate_permutation_exact=True,split_linear_matches_concatenated=True,probabilities_normalized=True,batch_invariance=True)
