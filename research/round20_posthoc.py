"""Descriptive summary only. Does not select, modify or refit any model."""
from round20_common import OUT,ROOT,read,dump,sha
from collections import Counter
import datetime,importlib.metadata as md,platform,sys
import numpy as np


def main():
    audit=read(OUT/'audit_result.json');frozen=read(OUT/'frozen.json')
    assert all(sha(ROOT/k)==v for k,v in frozen['hashes'].items())
    names=('full','evidence_only','quality_05','quality_10','random_matched')
    results=audit['results'];runs={n:read(OUT/f'runs/{n}/done.json') for n in names}
    with np.load(OUT/'audit_predictions.npz') as z:
        y=z['labels'];ids=z['ids'];base=y[np.arange(len(y)),z['full_long_choice']].astype(bool)
        repaired={};damaged={}
        for n in names:
            now=y[np.arange(len(y)),z[n+'_long_choice']].astype(bool)
            repaired[n]=ids[now&~base].astype(int).tolist();damaged[n]=ids[~now&base].astype(int).tolist()
    overlap=dict(quality_repaired_also_random=sorted(set(repaired['evidence_only'])&set(repaired['random_matched'])),
        quality_unique_repairs=sorted(set(repaired['evidence_only'])-set(repaired['random_matched'])),
        random_unique_repairs=sorted(set(repaired['random_matched'])-set(repaired['evidence_only'])))
    env=dict(python=sys.version,platform=platform.platform(),
        versions={k:md.version(k) for k in ('numpy','scipy','scikit-learn','torch')},
        cpu='11th Gen Intel Core i7-11800H @ 2.30GHz; 8 cores, 16 logical processors',
        gpu='NVIDIA RTX A2000 Laptop GPU, 4096 MiB',nvidia_driver='595.95',inference_threads=4,
        hardware_source='Win32_Processor and nvidia-smi queried in this session',
        gpu_training_only=True,official_validation_opened=False,test_opened=False)
    dump(OUT/'environment.json',env)
    observed_max=max((results[n][h]['correct'],n,h) for n in names for h in ('baseline','long','kernel'))
    diagnostics=dict(descriptive_only=True,no_new_predictions=True,no_model_or_selection_changes=True,
        head_long_repaired_ids=repaired,head_long_damaged_ids=damaged,**overlap,
        maximum_observed_audit=dict(correct=observed_max[0],n=950,variant=observed_max[1],head=observed_max[2],
            is_primary=observed_max[1]==frozen['primary'] and observed_max[2]=='long',
            must_not_be_interpreted_as_unbiased_winner_selection=True),
        limitations=['One training seed combination per arm.',
          'Only the development-selected 18-question removal has a size-matched random control; 157/314 deletion quality-specific effects are not identified.',
          'Secondary p-values are unadjusted for multiple comparisons and not confirmatory.',
          'Development and audit have opposite directions for several variants.',
          'The TRAIN audit is not official validation and not a pristine external sample.'])
    dump(OUT/'posthoc_diagnostics.json',diagnostics)
    lines=[]
    labels=dict(full='不筛选',evidence_only='有证据问题/歧义18题',quality_05='最多5%，实际157题',quality_10='最多10%，实际314题',random_matched='匹配随机删除18题')
    for n in names:
        v=results[n]['long']
        lines.append(f"{labels[n]}：开发 {runs[n]['development']['long']['correct']}/650；审查 {v['correct']}/950（{100*v['accuracy']:.2f}%）；修复 {v['repaired']}、误伤 {v['damaged']}、净增 {v['net']:+d}。")
    text='''训练数据质量筛选：结果快览（第20轮）

已完成筛选和重训。训练侧审查集上确实有小幅提升；所有版本已保留。
当前观察到的最高成绩是固定核模型融合、删除157道拟合题：896/950（94.32%），还未达到95%。
这组相对完整数据的同方法886/950，修复14题、误伤4题、净增10题；它属于次要探索对照，不能看完审查分数再把它说成预先选定主实验。
本轮训练/开发/审查题仍全部来自官方训练集，官方验证集与测试集均未使用。旧主模型和原始数据未改。

同一长文本融合方法的公平比较
'''+ '\n'.join(lines)+'''

如何理解
预先选定的18题清理版本，审查从887提升到891，修复6、误伤2，净增4；这是这批题上的真实改善，已保留。
但匹配随机删除18题也达到891，所以目前不能把这4题净增完全归因于筛选出了坏数据。
157题和314题版本的审查结果也更高，但它们在开发区分别净减8和11题，表明不同样本上结论尚不一致。
157题核模型融合在开发区是586/650，低于完整数据同方法的593/650；审查却净增10。两侧数据都应报告。
157题和314题版本没有各自等量随机对照；不能拿随机删除18题来证明删除157题的质量优势。
次要比较的单侧p值未经多重比较校正，只用于描述，不能据此宣布普遍可靠提升。
这950道审查题达到95%需要903题；当前探索最高896题，相差7题。该差距不能换算为官方验证或测试上的保证。

清理了什么
交叉预测只在3155道拟合题内部完成；最多可删315题，实际最多314题（9.95%），未为凑数量继续删除。
最高风险60题经助手审阅，结合拟合区中既有人工复核7题，共18题有具体缺陷/歧义证据，46题保护保留，3题未决。
例如参考解把6个班算成4个班，或唯一数值正确候选把周末当5天、通过错误抵消偶然得到正确终值。
这些推理缺陷与“官方数值标签错误”不是同一件事，官方数据与标签均未改。
其余被筛掉的样本只是模型不确定，未被认定为错标签。仅删模型不会的题会丢掉有效难例，因此设了保留保护和随机对照。

主实验修复的例子
train 1386：3台相机的利润差从只算每台5美元，改为总差15美元。
train 1661：把当前欠款60美元，改为4个月后剩余20美元。
train 1059：把取款前余额44美元，改为取尽5美元纸币后余下4美元。
也有误伤：train 4540从5件裙子共15周误改为单件3周。全部案例在audit_changes.jsonl中，未注入训练复核或重新调参。

建议保留状态
原主模型继续保留；18题清理版本、157题探索版本、314题版本和随机对照的权重全部保留。
后续固定配置复验应优先区分“质量筛选收益”和“训练路径/随机删样本收益”，再做一次明确选型的官方验证比较。
本轮没有因统计证据不足而删除有提升的候选，也没有事后改动审查前的主实验选择。

详见同目录：训练数据质量筛选_第20轮报告.txt、quality/adjudication.json、quality/removed_training_review.json、audit_result.json、candidate_status.json。
'''
    quick=OUT/'筛选重训效果快览.txt';quick.write_text(text,encoding='utf-8')
    manifest=read(OUT/'completion_manifest.json')
    for path in (quick,OUT/'posthoc_diagnostics.json',OUT/'environment.json'):
        manifest['artifacts'][str(path.relative_to(OUT))]=sha(path)
    manifest['posthoc_summary_only_no_new_fitting']=True
    dump(OUT/'completion_manifest.json',manifest)
    assert all(sha(ROOT/k)==v for k,v in frozen['hashes'].items())
    assert all(sha(OUT/k)==v for k,v in manifest['artifacts'].items())
    print(quick);print('Verified frozen models and completed artifact hashes.');print(overlap)


if __name__=='__main__':main()
