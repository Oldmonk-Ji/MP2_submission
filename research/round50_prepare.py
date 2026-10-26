"""Correct legal action space and freeze a single missing-evidence safeguard."""
from data_access import ROOT
from round22_common import read,dump,sha
import numpy as np,shutil,datetime

OUT=ROOT/'experiments/round50'

def main():
    OUT.mkdir();source=ROOT/'experiments/round49';plan=read(source/'plan.json')
    plan.update(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        objective='Learn only selectable executed programs; complete answer coverage required for neural intervention. No confidence threshold or alpha tuning.',
        valid_actions='Program mask AND at least one original candidate final answer matches execution. No reference consulted for action legality.',
        unknown_guard='If any valid numeric answer lacks a represented executable hypothesis, return baseline choice. Consensus rows also keep baseline.',
        no_action_rows='Retained in every training epoch. A dummy action has zero optimization signal; zero stored reward denotes no executable policy action, not a false candidate label.',
        warm_start='Frozen round48 SFT20 warm checkpoints, reference distribution recomputed under corrected action mask.',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research'/f'round50_{n}.py' for n in ('prepare','train','selector','evaluate')]},
        reference_rule='Actual selectable program terminal answer compared with TRAIN reference. The deployment also uses a fixed baseline residual and input-only unknown fallback; RL does not directly optimize that ensemble argmax.')
    dump(OUT/'plan.json',plan)
    for name in ('node_features.npz','program_node_ptr.npy','constraint_pairs.json','constraint_parent_ids.npy','constraint_positive.npz','constraint_negative.npz','reviewed_programs.json','held_source_review_before_reference.json','held_execution_checks.json','preparation_result.json'):
        shutil.copy2(source/name,OUT/name)
    with np.load(source/'train_targets.npz') as z:data={k:z[k].copy() for k in z.files}
    raw=data['mask'].copy();data['mask'] &= data['matches'].any(2)
    assert not np.any((data['rewards']>.5)&~data['mask'])
    data['raw_program_mask']=raw;np.savez(OUT/'train_targets.npz',**data)
    dump(OUT/'action_mask_audit.json',dict(total_programs=int(raw.sum()),selectable_programs=int(data['mask'].sum()),
        rows_without_selectable_program=int((~data['mask'].any(1)).sum()),training_n=len(data['ids']),
        correct_program_coverage_unchanged=True,reference_free_legality=True,test_opened=False))
    dump(source/'coverage_phase_complete.json',dict(training_run=False,validation_run=False,successor='round50',
        reason='Do not train another known-misaligned action space; combine repaired evidence with legal terminal actions before the next matched ablation.',test_opened=False))
    dump(OUT/'prepared_freeze.json',dict(hashes={str(p.relative_to(ROOT)):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='prepared_freeze.json'}))
    print(read(OUT/'action_mask_audit.json'),flush=True)

if __name__=='__main__':main()
