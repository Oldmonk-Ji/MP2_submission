"""Raw-text model learns corrections to cross-fitted TRAIN-only rankings."""
import numpy as np
from round22_common import ROOT
from round241_network import TextComparator
import round242_pretrain as language_training

class ResidualComparator(TextComparator):
    def forward(self,q,c,qn,cn,baseline=None):
        raw,qmean,cmean=super().forward(q,c,qn,cn)
        return raw if baseline is None else raw+baseline,qmean,cmean

def train(data,y,rows,vocab,path):
    with np.load(ROOT/'experiments/round216/cache.npz') as z:
        held=z['folds']==2;held_s1=z['held_s1']
    offset=np.load(ROOT/'experiments/round225/fit_only_s1.npz')['scores'].copy()
    quadratic=np.load(ROOT/'experiments/round227/scores.npz')['design']
    offset+=.25*(quadratic-quadratic.mean(1,keepdims=True))
    hs=np.load(ROOT/'experiments/round234/context_ground_held.npz')['scores']
    offset[held]=held_s1+.25*(hs-hs.mean(1,keepdims=True))
    language_training.TextComparator=ResidualComparator
    return language_training.train(dict(data,baseline=offset.astype(np.float32)),y,rows,vocab,path)
