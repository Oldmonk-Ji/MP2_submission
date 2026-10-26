"""Score immutable audit masks against cached, immutable predictions only."""
from round21_audit_quality import ROOT, CODE, OUT, PREVIOUS, NAMES, HEADS, read, dump, sha, source_fingerprints
from round20_common import no_validation
from data_access import forbid_test, read_jsonl, parse_answer
from collections import Counter
import datetime
import json
import math
import numpy as np

VIEW_ZH = dict(full='完整审查集', defects_only='仅排除明确缺陷',
    defects_and_ambiguities='排除明确缺陷＋实质歧义', uncertainty_05='仅排除最不确定的约 5%',
    uncertainty_10='仅排除最不确定的 10%', quality_then_uncertainty_10='质量问题优先，再排除不确定题，共 10%')
MODEL_ZH = {'full_long':'原完整训练＋联合比较器',
            'evidence_only_long':'上一轮预选主方案（18 道训练剔除）',
            'quality_05_kernel':'上一轮探索性最高方案（5% 训练筛选＋核比较器）'}


def integrity():
    f = read(OUT / 'frozen.json')
    assert all(sha(ROOT / k) == v for k, v in f['hashes'].items()), 'Frozen review/masks changed'
    assert source_fingerprints() == f['source_hashes']
    checked = {}
    for n in (18, 19, 20):
        previous = read(ROOT / f'experiments/round{n}/frozen.json')
        assert all(sha(ROOT / k) == v for k, v in previous['hashes'].items()), f'round{n} changed'
        checked[f'round{n}_frozen_unchanged'] = True
    manifest = read(ROOT / 'experiments/round4/asset_manifest.json')
    assert all(sha(ROOT / 'experiments/round4/selected_bundle' / k) == v['sha256'] for k, v in manifest.items())
    # Test audit-hook logic by invocation, never by opening the forbidden files.
    for guard, name in ((forbid_test, 'gsm8k_test.npz'), (no_validation, 'gsm8k_validation.jsonl')):
        try: guard('open', (name,))
        except PermissionError: pass
        else: raise AssertionError('Data access guard failure')
    return dict(**checked, frozen_subsets_unchanged=True, incumbent_unchanged=True,
                original_data_unchanged=True, forbidden_file_guards_pass=True)


def main():
    assert not (OUT / 'results.json').exists(), 'One fixed descriptive evaluation; do not overwrite'
    verified = integrity()
    p = read(OUT / 'plan.json')
    masks = read(OUT / 'masks.json')
    original = read(PREVIOUS / 'audit_result.json')
    reviewed = read(OUT / 'adjudication.json')
    with np.load(PREVIOUS / 'audit_predictions.npz', allow_pickle=False) as z:
        ids = z['ids'].astype(int)
        y = z['labels'].astype(bool)
        choices = {f'{n}_{h}': z[f'{n}_{h}_choice'].astype(int) for n in NAMES for h in HEADS}
    assert ids.tolist() == p['audit_ids']
    idx = np.arange(len(ids))
    correct = {k: y[idx, c] for k, c in choices.items()}
    for name in NAMES:
        for head in HEADS:
            assert int(correct[f'{name}_{head}'].sum()) == original['results'][name][head]['correct']
    bank = read_jsonl(CODE / 'data/candidates/gsm8k_train.jsonl')
    hard = np.array([len(set(parse_answer(c['text']) for c in bank[i]['candidates'] if parse_answer(c['text']) is not None)) >= 2 for i in ids])
    views = {}
    for view, mask in masks.items():
        keep = np.isin(ids, mask['kept_ids'])
        assert len(set(mask['kept_ids']) | set(mask['removed_ids'])) == len(ids)
        assert not set(mask['kept_ids']) & set(mask['removed_ids'])
        assert keep.sum() == mask['kept_n']
        assert (~keep).sum() <= p['maximum_removed']
        results = {}
        for name in NAMES:
            for head in HEADS:
                key = f'{name}_{head}'
                a, b = correct[key], correct[f'full_{head}']
                n = int(keep.sum())
                repairs = int((keep & a & ~b).sum())
                harms = int((keep & ~a & b).sum())
                results[key] = dict(n=n, correct=int(a[keep].sum()), accuracy=float(a[keep].mean()),
                    wrong=int((~a[keep]).sum()),
                    original_n=len(ids), original_correct=int(a.sum()), original_accuracy=float(a.mean()),
                    excluded_correct=int(a[~keep].sum()), excluded_wrong=int((~a[~keep]).sum()),
                    repaired_vs_same_head_full_training=repairs, damaged_vs_same_head_full_training=harms,
                    net_vs_same_head_full_training=repairs-harms,
                    fixed_denominator_correct_fraction=float(a[keep].sum()/len(ids)),
                    hard_n=int((keep & hard).sum()), hard_correct=int((a & keep & hard).sum()),
                    retained_accuracy_at_least_95=bool(a[keep].mean() >= .95))
                assert results[key]['correct'] + results[key]['excluded_correct'] == results[key]['original_correct']
                assert results[key]['excluded_correct'] + results[key]['excluded_wrong'] == mask['removed_n']
        views[view] = dict(**mask, label_zh=VIEW_ZH[view], results=results)
    result = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        n=950, audit_is_subset_of_official_training=True, quality_review_n=len(reviewed),
        quality_review_categories=dict(Counter(r['decision'] for r in reviewed)),
        remaining_question_quality_not_certified=True, uncertainty_rank_uses_correctness=False,
        masks_fixed_before_this_subgroup_evaluation=True, prior_full_audit_already_seen=True,
        results_descriptive_not_new_validation=True, views=views,
        original_full_results_reproduced_exactly=True, shared_masks_for_all_models=True,
        reference_labels_unchanged=True, model_training=False, no_audit_examples_enter_training=True,
        incumbent_replaced=False, validation_opened=False, test_opened=False, integrity=verified)
    dump(OUT / 'results.json', result)
    write_reports(result, reviewed)
    # Snapshot all derived artifacts separately from the already frozen masks.
    paths = [f for f in OUT.iterdir() if f.is_file() and f.name != 'completion_manifest.json']
    paths.append(ROOT / 'research/round21_evaluate.py')
    dump(OUT / 'completion_manifest.json', dict(created_utc=result['created_utc'],
        status='completed_posthoc_audit_sensitivity_only', hashes={str(f.relative_to(ROOT)):sha(f) for f in paths},
        original_artifacts_verified=verified, validation_opened=False, test_opened=False))
    for view, v in views.items():
        print(view, 'coverage', v['coverage'], {k:(v['results'][k]['correct'], v['results'][k]['n'], round(v['results'][k]['accuracy']*100,4)) for k in MODEL_ZH})


def write_reports(result, reviewed):
    views = result['views']
    lines = [
        '第 21 轮：审查集质量与不确定性筛选（事后敏感性分析）',
        '',
        '范围纠正：用户所说“最多筛掉 10%”针对审查集，不是训练部分。',
        '本轮复用冻结模型和预测；审查题从未加入训练。没有因改审查范围重训模型。',
        '第 20 轮完整 950 题结果、原始数据与所有模型保持不变。本次原始成绩逐项复算一致。',
        '官方验证集与测试集均未打开。',
        '',
        '筛选程序',
        '1. 950 题全部按冻结 full_long 的不同最终答案组之间的分数间隔排序；不使用正确标签。',
        '   同数值候选先合并，避免将“答案相同、写法不同”误当成不确定。间隔是排序信号，不是校准概率。',
        '2. 复核 175 题的题干与官方最终数值：最不确定 95 题、按语义提示选 40 题、开头 25 题、固定哈希抽查 20 题，去重后 175。',
        '   对其中 17 道疑点题进一步逐条查看参考步骤与全部四个真实候选。未做外部人工复核。',
        '3. 认定 5 道明确缺陷、7 道实质歧义；5 道弱疑点暂保留，158 道未见足够排除依据。其余 775 题未完成逐题语义复核。',
        '4. 先封存规则、逐题理由和名单，再计算本次子集成绩。此前已看过完整审查结果，因此仍是事后分析，不能称新的盲审。',
        '5. 六种固定视图使用同一名单比较所有 15 个模型／预测头，最多排除 95/950=10%。',
        '   仅质量筛选不凑满额度。高不确定性筛选可以排除完全有效的难题，不能把这些题称作错标。',
        '',
        '核心成绩（准确率只针对保留题；括号为答对数/保留数）',
    ]
    for key, label in MODEL_ZH.items():
        lines += ['', label + ' [' + key + ']']
        for view, v in views.items():
            m = v['results'][key]
            lines.append(f"  {VIEW_ZH[view]}：{m['accuracy']:.2%}（{m['correct']}/{m['n']}），覆盖率 {v['coverage']:.2%}；排除 {v['removed_n']} 题中，原来答对 {m['excluded_correct']}、答错 {m['excluded_wrong']}。")
    key = 'quality_05_kernel'
    base = views['full']['results'][key]
    quality = views['defects_and_ambiguities']['results'][key]
    combined = views['quality_then_uncertainty_10']['results'][key]
    lines += ['', '如何解读',
        f"上一轮探索性最高模型：完整成绩仍是 {base['correct']}/{base['n']}={base['accuracy']:.2%}。",
        f"仅排除 12 道质量疑点：{quality['correct']}/{quality['n']}={quality['accuracy']:.2%}。",
        f"质量优先＋不确定性合计排除 95 道：{combined['correct']}/{combined['n']}={combined['accuracy']:.2%}，覆盖 90%。",
        f"这 95 道中模型原来答对 {combined['excluded_correct']} 道、答错 {combined['excluded_wrong']} 道；删题没有修复任何一次原有预测。",
        f"如把未覆盖的 95 题视为未作答，按原始 950 题计，已答对比例是 {combined['fixed_denominator_correct_fraction']:.2%}；不能把保留子集分数当成全量准确率。",
        '95% 若只在筛选后子集出现，含义是“覆盖 90% 的选择性准确率达到 95%”，不是全量审查或官方验证达到 95%。',
        '质量复核与模型置信度针对不同问题。真正提高全量能力仍需修复保留和排除题中的有效难题，并在新固定数据划分验证。',
        '本次标签和理由仅供审查诊断，不进入训练；未来若把它们用于训练，必须弃用这批题作为独立评估依据。',
        '', '所有固定对照（便于复核修复、误伤、净增；参照同预测头的 full 训练模型）']
    for view, v in views.items():
        lines += ['', VIEW_ZH[view], 'model | correct/n | accuracy | repairs | harms | net']
        for key, m in v['results'].items():
            lines.append(f"{key} | {m['correct']}/{m['n']} | {m['accuracy']:.6%} | {m['repaired_vs_same_head_full_training']} | {m['damaged_vs_same_head_full_training']} | {m['net_vs_same_head_full_training']:+d}")
    lines += ['', '复现（工作目录为项目 Project2）',
        '已有冻结结果：读取 experiments/round21_audit_quality/results.json 与 masks.json 即可逐项校验。',
        '原始执行顺序：python research/round21_audit_quality.py prepare；python research/round21_adjudicate.py；python research/round21_evaluate.py。',
        '脚本禁止覆盖已存在的冻结与结果文件。不要删除已有结果来重跑；若新实验需要变更规则，另建一轮。',
        '所有明细、原题、参考答案、替代方程、名单、源文件哈希均在本目录；模型使用上一轮缓存预测，不发生训练。']
    (OUT / '审查集筛选结果.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    notes = ['审查集质量复核清单（AI 复核，未改官方标签）',
        '题号是官方训练文件的 0-based 行号。仅用于事后诊断，不注入训练复核页面。',
        '以下 17 题查看了原始四候选；不展示任何模型选择或该模型答对答错。', '']
    for r in reviewed:
        if r['decision'] == 'keep': continue
        notes += [f"题号 {r['train_index']} | {r['decision']} | {r['category_zh']}",
            '原题：'+r['question'], '原参考最终答案：'+r['reference_answer'],
            '判断理由：'+r['reason_zh'],
            '方程：'+'；'.join(f"{x['expression']} = {x['value']:g}" for x in r['independent_interpretations']),
            '质量筛选处理：'+('暂保留' if r['decision']=='uncertain_keep' else '仅在诊断子集中排除'), '']
    notes += ['其余已复核题的独立算式与说明见 adjudication.json；未复核题没有被自动认证为高质量。']
    (OUT / '审查集质量复核清单.txt').write_text('\n'.join(notes)+'\n', encoding='utf-8')


if __name__ == '__main__': main()
