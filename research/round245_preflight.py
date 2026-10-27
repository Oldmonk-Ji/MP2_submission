"""Untrained TRAIN-input numerical/inference preflight, no selection labels."""
from round22_common import ROOT,load,dump
from round245_text import fit_vocab,encode,save_vocab
from round245_network import Comparator,export,predict
from round245_numpy import scores
from threadpoolctl import threadpool_limits
import numpy as np
import torch

def main():
    threadpool_limits(4);torch.set_num_threads(4)
    bank,_,_,_=load('train');vocab=fit_vocab(bank,np.arange(64))
    out=ROOT/'experiments/round245_preflight';out.mkdir(exist_ok=False)
    save_vocab(out/'vocab.json',vocab);model=Comparator(len(vocab));export(model,out/'model.npz')
    original=predict(model,encode(bank[:3],vocab));actual=scores(bank[:3],out)
    difference=float(np.abs(original-actual).max())
    assert np.allclose(original,actual,atol=2e-4,rtol=2e-5),difference
    permutation=[2,0,3,1]
    shuffled=[dict(question=r['question'],candidates=[r['candidates'][j] for j in permutation]) for r in bank[:3]]
    assert np.allclose(scores(shuffled,out),actual[:,permutation],atol=2e-4)
    assert np.allclose(scores(bank[:1],out),actual[:1],atol=2e-4)
    result=dict(torch_numpy_max_difference=difference,candidate_permutation=True,batch_invariance=True,training_only_inputs=True,test_opened=False)
    dump(out/'result.json',result);print(result,flush=True)

if __name__=='__main__':main()
