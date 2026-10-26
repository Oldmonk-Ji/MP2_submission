"""Matched SFT ablation: remove ONLY the extra abstract relation feature block."""
from data_access import ROOT
from round22_common import read,dump,sha
from scipy import sparse
import round48_train as training
from round50_selector import select
import shutil

OUT=ROOT/'experiments/round53'

if __name__=='__main__':
    OUT.mkdir();old=ROOT/'experiments/round51'
    for name in ('train_targets.npz','program_node_ptr.npy','constraint_pairs.json','constraint_parent_ids.npy','held_source_review_before_reference.json','held_execution_checks.json'):
        shutil.copy2(old/name,OUT/name)
    for name in ('node_features.npz','constraint_positive.npz','constraint_negative.npz'):
        sparse.save_npz(OUT/name,sparse.load_npz(old/name)[:,:1024])
    plan=read(old/'plan.json');plan.update(arms=['sft'],objective='Matched lexical-only SFT: same programs, seeds, examples, optimizer and epochs as round52 SFT.',
        source_hashes={**plan['source_hashes'],str(__import__('pathlib').Path(__file__).resolve().relative_to(ROOT)):sha(__file__)},test_opened=False)
    dump(OUT/'plan.json',plan);dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file()}))
    training.OUT=OUT;training.select=select;training.main()
