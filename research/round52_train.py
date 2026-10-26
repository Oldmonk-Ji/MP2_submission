"""Matched learning on identical legal programs; abstract relation features added."""
from data_access import ROOT
from round22_common import read,dump,sha
import round48_train as training
from round50_selector import select
import shutil

OUT=ROOT/'experiments/round52'

if __name__=='__main__':
    OUT.mkdir();old=ROOT/'experiments/round51'
    for p in old.iterdir():
        if p.is_file() and p.name not in ('prepared_freeze.json','train_proposals.jsonl','plan.json'):shutil.copy2(p,OUT/p.name)
    plan=read(old/'plan.json');plan.update(objective='Source-identity-independent relation features plus same SFT/exact/GRPO objectives.',training_n=4743,
        updates_per_rollout=2,learning_rate=.0003,weight_decay=.01,kl=.02,relation_weight=.2,group_size=16,clip=.2,
        warm_start='Scratch trained, no external weights',validation='Three fixed training arms plus one frozen-weight coverage-only ablation, no parameter sweep',
        final_test='Best NEW method by complete VAL, then fixed TEST comparison; incumbent selection remains VAL-only')
    dump(OUT/'plan.json',plan);dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}))
    training.OUT=OUT;training.DIM=2048;training.select=select;training.main()
