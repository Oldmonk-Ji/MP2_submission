"""Preserve evidence, validate frozen sources, and report negative results."""
from data_access import ROOT,CODE,read_jsonl
from round22_common import read,dump,sha,load
from round40_evaluate import metrics
from round29_37_report import closure
import numpy as np,datetime,sys

def export():
    bank=read_jsonl(CODE/'data/candidates/gsm8k_validation.jsonl');b,a,y,ids=load('validation')
    for number in (40,42):
        out=ROOT/f'experiments/round{number}'
        with np.load(out/'validation_predictions.npz') as z:choices=z['choices'];base=z['baseline']
        assert len(choices)==798 and all(0<=int(j)<4 for j in choices)
        dump(out/'official_validation_predictions.json',dict(protocol='dase7506-mp2-open-hard-verifier-v5',split='validation',ids=[r['id'] for r in bank],selected=choices.astype(int).tolist()))
        pp=[dict(answer=None) for _ in b]
        for r in read(out/'validation_proofs.json'):pp[r['validation_index']]=r['extraction']
        report=read(out/'validation_result.json');report.update(metrics(b,y,base,choices,pp));dump(out/'validation_result.json',report)
    # Correct diagnostic conflict denominator: None is not a valid answer.
    out=ROOT/'experiments/round40';b,a,y,ids=load('train');rr=read(out/'train_effect.json')
    from data_access import parse_answer
    mask=np.asarray([len({parse_answer(c['text']) for c in r['candidates']}-{None})>1 for r in b])
    rr['conflict_metric_correction']='Only distinct valid parsed answers count as conflicts; None excluded. Prediction choices unchanged.'
    # Derive the train subset metrics without running inference again.
    from round40_evaluate import apply
    with np.load(ROOT/'experiments/round38/oof_baseline.npz') as z:base=z['choices'];s=z['scores'];fold=z['folds']
    xx=read(out/'train_extractions.json');new,pp=apply(b,base,s,[r['extraction'] for r in xx]);disc=set(read(out/'freeze.json')['discovery_ids'])
    held=np.asarray([k for k,i in enumerate(ids) if fold[k]!=0 and int(i) not in disc])
    rr['all']=metrics(b,y,base,new,pp);rr['uninspected_held']=metrics([b[k] for k in held],y[held],base[held],new[held],[pp[k] for k in held]);dump(out/'train_effect.json',rr)
    print('Exported full-validation predictions; corrected conflict-only diagnostics.')

def finish():
    old={};mismatches=[];seen={}
    for number in range(22,38):
        manifest=ROOT/f'experiments/round{number}/completion_manifest.json'
        if not manifest.exists():continue
        hashes=read(manifest)['hashes']
        for path,h in hashes.items():
            if path not in seen:seen[path]=sha(ROOT/path)
            if seen[path]!=h:mismatches.append(dict(round=number,path=path))
        old[str(number)]=len(hashes)
    assert not mismatches,mismatches
    frozen={}
    for number in (39,40,41,42):
        hashes=read(ROOT/f'experiments/round{number}/freeze.json')['source_hashes']
        for path,h in hashes.items():assert sha(ROOT/path)==h,path
        frozen[str(number)]=len(hashes)
    current={}
    for number in (40,42):
        out=ROOT/f'experiments/round{number}';r=read(out/'validation_result.json');official=read(out/'official_validation_score.json')
        assert r['correct']==official['correct']==752 and official['n']==798
        current[str(number)]=r
    asset=read(ROOT/'experiments/round25/asset_manifest.json');assert asset['bytes']<=16*1024**2
    schemas=[read(ROOT/f'experiments/round{n}/relation_supervision.json') for n in (40,42)]
    total_facts=sum(len(r['relation_labels']) for r in schemas[0])+sum(len(r['labels']['facts']) for r in schemas[1])
    summary=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rounds=[38,39,40,41,42],
        root_error_population=352,root_adjudications=read(ROOT/'experiments/round38/root_cause_summary.json'),
        relation_supervised_questions=len(schemas[0])+len(schemas[1]),relation_supervised_facts=total_facts,
        relation_label_limit='Assistant source-adjudicated labels with span/execution/contrast checks; no independent human agreement or claim of globally reliable extraction.',
        rejected_semantic_audits={str(n):read(ROOT/f'experiments/round{n}/result.json') for n in (39,41)},
        validation=current,selected_round=25,validation_correct=752,validation_n=798,validation_accuracy=752/798*100,
        conflict_correct=309,conflict_n=355,test_opened=False,test_accuracy=None,goal_97_achieved=False,
        goal_97_test_min_correct=453,training_n=4743,validation_exclusions=[],additional_pretrained_weights=False,new_neural_training_runs=0,
        selected_asset_bytes=asset['bytes'],old_completion_manifests_verified=old,distinct_old_files_verified=len(seen),new_pre_audit_freezes_verified=frozen,
        runtime_note='round40 encountered unsupported mixed-number literal during first validation pass before result selection. Frozen parser unchanged; evaluation wrapper returns unknown on parsing exceptions. No validation-specific semantic rule was added.',
        rules_checked_date='2026-10-16',rules_url='https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md',
        next_research_constraint='Do not call a few successful hand-built grammars a semantic solution. Expand source-adjudicated relation and target supervision across unseen TRAIN syntax, retain fold-isolated evaluation and measure natural error correction. Existing training discovery/audit IDs are now seen; no reusing them as fresh evidence.')
    dump(ROOT/'experiments/semantic_root_cause_20261016.json',summary)
    lines=['真实错题、关系监督与语义路线验证：第38—42轮','日期：2026-10-16','',
      '结论：最终测试超过97%的目标尚未完成。没有新测试成绩；测试输入和答案未用于本轮工作。',
      '当前仍保留第25轮：完整官方验证752/798 = 94.235589%；不同有效数值答案的冲突题309/355 = 87.042254%。',
      '这两项均不是测试成绩。467题测试超过97%需至少453题正确，当前没有证据宣称已经做到。','',
      '一、先查真实错误，而不是继续增加模型或扫参数',
      '从同题/规范化模板隔离的训练OOF预测中找到352个真实错误。使用第22轮历史固定融合及算术门控，不使用全量训练模型的样本内错题冒充泛化错误。',
      '按预先固定哈希顺序选取fold0的24题，逐题读取原问题、实际错误候选和训练参考。标注错误主张、原文证据位置、正确关系和可执行目标。',
      '24题中20题是明确语义关系错误，2题只有算术/代数错误，2题两者兼有。不能把所有错误候选的所有步骤都当负标签。',
      '根因包括：物品/人物绑定；相等关系的范围；分数所作用的原始量或剩余量；速度和用时方向；多层每份关系；所求单价、数量与总价；时间范围；状态终点。',
      '例：训练3243漏掉每栋12层；3259把25元大杯总价当5个大杯的数量；3635把21元蓝本总价当3元单价。','',
      '二、建立可检查的关系监督',
      f'最终保留26题、{total_facts}条关系事实监督，分别保存在round40和round42；与24题根因标注有交集，不应相加宣称50个不同题目。',
      '事实区分物品数量、该物品的单价、总支出、可用金额、余额、层级每份关系、单位倍率、时间基准；目标类型单独标注。',
      '每条原文事实保存精确字符区间。日历/时钟换算是明示固定常数，单独标记，不伪造为题干原文证据。',
      '这些标签是AI辅助逐条审查，不是独立人工金标；执行得到正确数值只是附加检查，不等于语义标签天然正确。',
      '反事实检查覆盖换人名、数量修改、重复相同数值、预算不等于花光、总量/单价/差额目标、缺失层级、名词中心词、时间换算和未知子集限定。',
      '没有外部题库、额外预训练权重或新增神经模型训练。没有给改写题沿用原官方embedding。','',
      '三、冻结规则后的自然训练审查与完整验证',
      '第39轮金额关系：独立训练折6个完整证明中4个成立、2个错误。事实抽对但把平均价格和两人花费差当总花费，否决该版，没有用验证挑结果。',
      '第40轮加入目标监督：训练共覆盖19/4743；OOF总修复3、误伤0，但收益不属于新的未见审查证据。新未见训练折没有覆盖到完整证明。',
      '第40轮完整验证覆盖2/798，得分752/798，修复0、误伤0；未替换第25轮。',
      '第41轮层级关系：独立训练折8个证明中6个成立、2个错误。遗漏一打的12倍和每月到每年的时间换算，否决该版，没有用验证挑结果。',
      '第42轮补入单位和时间范围：训练覆盖14/4743，OOF总修复2、误伤0；收益涉及已经进入开发的数据，不宣称独立收益。',
      '第42轮完整验证覆盖0/798，得分752/798，修复0、误伤0；未替换第25轮。',
      '旧审查失败进入新版开发后，明确从“未见审查”分母移除；不能用修好旧审查题后的高正确率宣称泛化。',
      '第40轮初次验证遇到不支持的带分数文本，计算在选型前中断；只给评估包装层加解析异常时退回基线的处理，冻结语义规则未改变。',
      '最终两版均由原官方score_predictions.py在新进程复核完整798题。记录中的约5—6秒是进程内开发计时，不是包含启动、导入及独立评分的合规计时。','',
      '四、目前明确的瓶颈',
      '局部手写语法能够解释并修复发现集，却未覆盖新的自然表达。当前瓶颈是关系抽取的覆盖与完整性，不是再调整融合权重。',
      '继续手写少数题型补丁没有证据能把最终测试推到97%；本轮没有证明新的路线已经具有广泛语义理解能力。',
      '下一步监督需要覆盖更多真实训练表达，尤其目标、数量所属对象、时间和状态绑定，并以未见训练错误的实际修复量验证迁移。已看过的题号必须登记为开发数据。',
      '保留本轮正负例和失败证据，后续可用于关系抽取器监督；不要把自动执行一致的参考算式重新当作可靠关系金标。','',
      '五、边界与可复现性',
      '训练约定仍是4743题（仅排除约定12题）；最终基线资产未改动。验证798题全部参与评分，输出每题一个原候选，未知关系只在内部回退，不拒答、不删题。',
      '已在线核对官方GUIDE：仅所供训练数据、验证开发选型、测试最终评分、无额外预训练权重、离线选择、16MiB、参考CPU四线程30秒等要求未变。',
      '官方来源：https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md',
      f'第22—37轮完成清单已逐项核对，去重后{len(seen)}个旧文件哈希一致；本轮四个审查前冻结清单也均一致。',
      f'保留第25轮推理资产{asset["bytes"]}字节。未执行测试、未提交榜单、未改写旧轮次。','',
      '主要文件：',
      'experiments/round38/root_cause_annotations.json',
      'experiments/round40/relation_supervision.json',
      'experiments/round42/relation_supervision.json',
      'experiments/round39/natural_relation_adjudications.json',
      'experiments/round41/natural_relation_adjudications.json',
      'experiments/round40/official_validation_score.json',
      'experiments/round42/official_validation_score.json',
      'experiments/semantic_root_cause_20261016.json','',
      '复跑完整验证（PowerShell，项目根目录）：',
      "$env:PYTHONPATH='E:\\New Life\\HKU\\7506\\Project2\\research;E:\\New Life\\HKU\\7506\\Project2\\.venv\\Lib\\site-packages'",
      "& 'D:\\anaconda3\\envs\\mp1\\python.exe' research/round40_evaluate.py",
      "& 'D:\\anaconda3\\envs\\mp1\\python.exe' research/round42_evaluate.py",
      '注意：prepare/study脚本按设计拒绝覆盖已有freeze；重建应在单独工作副本中进行，保留原记录。']
    report=ROOT/'真实错题与关系监督验证_2026-10-16_第38至42轮.txt';report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=set()
    for n in range(38,43):
        files.update(p for p in (ROOT/f'experiments/round{n}').rglob('*') if p.is_file())
        files.update((ROOT/'research').glob(f'round{n}_*.py'))
    files.update(closure([ROOT/'research/round40_evaluate.py',ROOT/'research/round42_evaluate.py']))
    files.update([ROOT/'research/round38_42_report.py',report,ROOT/'experiments/semantic_root_cause_20261016.json'])
    dump(ROOT/'experiments/round38_42_completion_manifest.json',dict(experiment_batch_completed=True,goal_97_achieved=False,test_opened=False,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    print(dict(report=str(report),relation_facts=total_facts,prior_files_verified=len(seen),new_files_frozen=len(files),goal_97_achieved=False))

if __name__=='__main__':globals()[sys.argv[1]]()
