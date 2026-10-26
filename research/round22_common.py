"""All eligible official TRAIN rows fit final experts; official VAL selects method."""
from data_access import ROOT, CODE, load as official_load
from pathlib import Path
import datetime, hashlib, json
import numpy as np

OUT = ROOT / 'experiments/round22'
INCUMBENT = ROOT / 'experiments/round4/selected_bundle'
BEST_INTERNAL = ROOT / 'experiments/round20/runs/quality_05'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def plan(): return read(OUT/'plan.json')


def load(name):
    assert name in ('train','validation')
    b,a,y=official_load(name);ids=np.arange(len(b))
    if name=='train':
        ids=np.asarray(plan()['training_ids'])
        b=[b[i] for i in ids];a={'embeddings':a['embeddings'][ids]};y=y[ids]
        assert len(b)==4743
    else:assert len(b)==798
    b=[dict(question=r['question'],candidates=[dict(index=j,text=c['text']) for j,c in enumerate(r['candidates'])]) for r in b]
    return b,a,y,ids


def subset(b,a,y,ix):
    return [b[i] for i in ix],{'embeddings':a['embeddings'][ix].copy()},y[ix]


def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'plan.json').exists()
    excluded=read(ROOT/'experiments/round21_audit_quality/masks.json')['defects_and_ambiguities']['removed_ids']
    assert len(set(excluded))==12
    fit=[i for i in range(4755) if i not in excluded]
    p=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seed=22022,
        latest_user_instruction='Final models train on ALL official TRAIN except the agreed 12. Official VAL (798) selects methods; TEST remains sealed. No permanent internal audit split and no evaluation deletion.',
        training_ids=fit,excluded_training_ids=excluded,training_n=4743,validation_n=798,
        evaluation_exclusions=[],normal_hard_training_examples_restored=True,
        teacher_recipe='Same five scratch-trained models as incumbent plus PCA96/256-anchor RBF comparator.',
        training='Every final expert fits all 4743 eligible questions. Grouped 3-fold cross-fitting is used ONLY to obtain honest training predictions for fitting readout; no question is permanently held out of final experts.',
        oof_folds=3,kernel=dict(pca=96,anchors=256,gamma_factor=3.,C=.1,seed=19019),
        readout_families=['linear','conditional','prob_candidate','prob_group_mean','prob_group_max'],
        linear_C=[.01,.1,1.],residual_alpha=[.1,.25,.5,1.],probability_temperature_range=[.25,4.],probability_penalty=.005,
        gate='parse_and_repair',gate_margin=4,
        baseline='Frozen round4 incumbent, official validation 750/798; reproduce on full validation.',
        baseline_weights='Original round4 full-training fusion; round20 quality_05 weights also retained as fixed historical-method control.',
        selection='Official 798-question validation accuracy, then fewer harms versus incumbent, smaller alpha, smaller parameter count. All predeclared variants reported. No test access.',
        validation_role='Reused development evidence, not an untouched generalization estimate. Complete official test is final submission metric.',
        training_exclusion_caveat='12 IDs from train-side review explicitly approved for training exclusion. No other rows filtered.',
        goal_95_validation_correct=759,
        official_metric=dict(name='official score_predictions.py accuracy percent',population='complete 467-question test bank',
            choice='one candidate 0..3 per question; numeric final answer correctness',exclude_or_abstain=False,
            all_test_questions_have_answer_disagreement=True,accuracy95_minimum_correct=444,accuracy99_minimum_correct=463,
            inference_assets_max_bytes=16*1024**2,cpu_seconds_max=30,total_threads=4,reference_cpu='Xeon Platinum 8457C',
            source='https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md',verified_client_date='2026-10-15'),
        prior_938_950_scores_not_selection_target=True,test_opened=False,
        source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [ROOT/'experiments/round21_audit_quality/masks.json',
            INCUMBENT/'base/config.json',CODE/'data/candidates/gsm8k_train.jsonl',CODE/'data/gsm8k_train.jsonl']})
    dump(OUT/'plan.json',p)
    dump(OUT/'training_exclusion_policy.json',dict(forbidden_training_ids=excluded,training_ids=fit,
        training_n=4743,validation_n=798,evaluation_exclusions=[],test_opened=False))
    print(dict(final_training=4743,official_validation=798,excluded_training_ids=excluded,test='SEALED'))


if __name__=='__main__':prepare()
