"""Write the descriptive report from frozen, already evaluated artifacts."""
from round18_common import OUT, ROOT, sha
import datetime, json


def read(name):return json.loads((OUT/name).read_text(encoding='utf-8'))


def main():
    r=read('audit_result.json');f=read('frozen.json');d=read('posthoc_diagnostics.json');arch=read('architecture.json')
    assert all(sha(ROOT/k)==v for k,v in f['hashes'].items())
    results=r['results'];rows=[]
    names={'baseline':'同划分重训的五成员基线','joint':'基线＋联合比较器（主实验）','independent':'基线＋关闭候选交互的对照','joint_standalone':'联合比较器单独选答案＋相同算术门控','independent_standalone':'独立对照单独选答案＋相同算术门控'}
    for key,name in names.items():
        v=results[key]
        rows.append(f"{name}：{v['correct']}/{v['n']}（{100*v['accuracy']:.2f}%）；修复 {v['repaired']}，误伤 {v['damaged']}，净增 {v['net']:+d}；冲突子集 {v['hard_correct']}/{v['hard_n']}。")
    j=results['joint'];ci=j['cluster_bootstrap_95_ci_percentage_points'];jd=d['results']['joint']
    text=f'''第18轮：真实训练候选的从零初始化联合比较器
时间：2026-10-15（UTC+8）

结论
本轮训练与封存审查已完成，未通过进入官方验证集的门槛，保留原选定基线。
主实验在950道训练侧封存题上修复3题、误伤2题、净增1题。联合结构的可靠收益尚未证实。
这不是官方验证集成绩，不能与历史750/798直接相减。99%目标未达到。
本轮未读取官方validation或test的候选、嵌入或答案，未加入参考解步骤监督或生成题。

一、数据与隔离
4755道官方训练题固定划分：拟合3105、开发调参700、封存审查950。
同一道题四个候选不能跨区；人物名及数值归一化后的相同题干模板也不跨区。
已知人工标注过的题目组排除在新审查区之外。模板规则无法保证发现全部近义改写。
全部词表、标准化参数和模型权重只在拟合区学习。融合系数与早停轮数只在开发区选择。
模型、配置、代码、训练输入及判定规则先做SHA256封存，再读取本轮审查标签。
项目历史探索曾接触训练数据，因此这里是本轮拟合与评估隔离，不是完全未经历史探索的外部测试。
本轮审查已消耗；不允许根据这些结果继续在同一审查区挑模型、阈值或融合系数。

二、本轮确实改了什么
新写一个197089参数的神经比较器。网络、词表及投影权重从零初始化；官方提供的固定1024维候选嵌入仍作为允许的输入特征。
输入是题目文字、候选文字及对应的官方嵌入，不输入题目ID、候选位置ID、官方答案或参考解正文。
词级表示与数值特征经过共享卷积编码，候选先对照题目，再让两个候选互相对照。
同一对候选共用打分网络，互换顺序时差值反号；每题六对差值聚合为四个候选分数。
监督仅由官方最终数值标签导出。只比较同题正确与错误候选，多份正确候选不会互相当负例。
拟合区有12420份真实候选，1660道题同时包含正负候选，形成5464对正误比较；没有受控样本或生成文本。
损失为题内平均的正误配对logistic损失，加0.25倍多正确答案集合损失。
独立对照使用相同结构、随机种子、题目输入和选型政策，只将“另一候选的上下文”置零。
这是明确的结构消融，但只有一个随机种子，且两版分别早停；不应把细小差异解释为结构的普遍因果效果。

三、公平基线
重新训练现有五个成员：原MLP、数量特征MLP、答案组特征MLP、文字与嵌入联合模型、词组线性模型。
旧的全训练集权重、词表、标准化及融合系数都没有直接拿来评分新审查区。
五个成员在相同3105题拟合；五项融合系数在相同700题开发区重新估计。
已有算术修正门控固定保留，所有比较共享它。旧模型架构和门控仍包含历史开发经验。
联合模型的融合为 baseline_scores + alpha × 题内标准化后的比较器分数，再经过相同门控。
alpha候选在开审查前限定为0.1、0.25、0.5、1.0；两版最后都选0.1。
联合版选第14轮，最多实际运行22轮；独立版选第4轮，实际运行12轮。最大24轮、耐心8轮。
开发区：基线649/700；联合融合651/700（修复3、误伤1）；独立融合649/700（修复2、误伤2）。

四、封存审查结果
{chr(10).join(rows)}

主实验共改变44道题的候选选择：3次错改对、2次对改错、1次错改错、38次对改对。
“改变候选”不能当作“修复错误”，相同正确答案之间切换不增加正确数。
主实验准确率差 {j['accuracy_gain_percentage_points']:.3f} 个百分点；题目模板分组bootstrap的95%区间 [{ci[0]:.3f}, {ci[1]:.3f}] 个百分点。
精确单侧配对二项检验p={j['paired_one_sided_p']:.3f}。区间跨零，无法排除抽样波动。
进入下一阶段的预设条件是：开发净增>0、审查净增至少5、配对p≤0.05、bootstrap区间下界>0。
本轮只有开发净增条件通过，其余三项失败。
独立对照的审查分数更高，但它是事先固定的次要诊断，不能开审查后改选它并宣布成功。

五、过拟合与覆盖诊断（开审查后的描述性分析，不用于选型）
联合模型拟合区正误对准确率：{jd['fit']['pair_quality']['correct']}/{jd['fit']['pair_quality']['n']} = {100*jd['fit']['pair_quality']['accuracy']:.2f}%。
联合模型开发区正误对准确率：{100*jd['development']['pair_quality']['accuracy']:.2f}%；审查区：{100*jd['audit']['pair_quality']['accuracy']:.2f}%。
同一损失在拟合区评估为{jd['fit']['eval_loss']:.4f}，开发区{jd['development']['eval_loss']:.4f}，审查区{jd['audit']['eval_loss']:.4f}，显示明显过拟合。
词表最多4096，题目上限96 token，候选上限160 token；超过长度时固定保留头尾，不使用标签裁剪。
审查3800份候选中1372份发生截断，保留原始token总量的85.14%；题目保留99.51%。
截断是必须检验的覆盖风险，本轮没有证明它是性能失败的唯一原因。
本轮只用最终答案监督，不能把选到正确数字解释为逐步推理正确或人物、价格、时间关系均已学会。

六、实际修复与误伤案例
以下均是已经封存并评估过的审查题，只作分析，不追加到拟合标注或训练复核页面。
1. train_index=689，修复：分糖后比较Andy与Caleb；错误解只比较新增17和11，正确解比较现有26和22，答案6改为4。独立对照也修复。
2. train_index=728，修复：全年早餐鸡蛋；错误解漏乘每周5天，答案676改为3380。独立对照也修复。
3. train_index=3505，修复：问年龄却答练击剑年数；答案16改为30。独立对照也修复。
4. train_index=3077，误伤：网络月费把“每月25日”读成“每25天”，又重复累计，答案190改为800。两版都误伤。
   此题另有标注歧义：题干写before the 25th却实际在25日付款，严格按字面应不享折扣、合计200；官方按190计分，四候选也没有200。
   本轮仍保留官方标签与全部题目，未删掉此例来抬高成绩。这是人工阅读发现的题意问题，不是重新训练标签；未据此估计全数据歧义比例。
5. train_index=4298，误伤：3盒纸巾总价6元，问每盒价格。联合版把2改为6，遗漏除以3；独立版保留2。
6. train_index=1412，错改错：两人每天各吃一个苹果，题目问周数；21天改为6周仍错，正确为3周。
三项修复与独立对照完全重合，联合版额外误伤单价题。这批案例不支持“候选互相对照已可靠解决关系绑定”的说法。
逐题索引、官方标签、分数及全部改变记录保存在audit_decisions.jsonl和audit_changes.jsonl。

七、实现检查与资源
按题及模板隔离、候选重排、相同候选零差值、去掉ID后输入不变、CPU导出复现均通过；详见checks.json。
冻结原有基线的16个资产SHA256全部保持不变。
联合模型额外推理资产（权重、配置、词表）867984 bytes；与本轮基线合计16528042 bytes，小于16MiB；仍须正式打包后重新计量。
本机4线程、950题、进程已启动且输入已加载：基线{r['timing']['baseline_local_warm_seconds']:.2f}秒；联合模型加载、编码、预测额外{r['timing']['joint_load_encode_predict_local_warm_seconds']:.2f}秒。
这些是局部热运行诊断，不包含完整冷启动官方评分流程，也不是参考硬件认证。
训练使用RTX A2000 Laptop GPU；评估使用CPU。完整版本号在posthoc_diagnostics.json。

八、复现与冻结纪律
当前目录已封存，train/freeze/audit脚本有重复运行保护，避免覆盖结果或重复挑选。
要完整复现实验，应使用独立项目副本，保留历史依赖、官方TRAIN文件和本轮plan.json，副本的experiments/round18只放原plan.json。
不要删除或移动现有封存目录。不要把审查样本混回拟合区再宣称同一审查结果独立。
在副本Project2目录，使用已有PyTorch解释器：
  $env:PYTHONIOENCODING='utf-8'
  $env:PYTHONPATH='<副本Project2>\\.venv\\Lib\\site-packages'
  & 'D:\\anaconda3\\envs\\mp1\\python.exe' '.\\research\\round18_baseline.py'
  & 'D:\\anaconda3\\envs\\mp1\\python.exe' '.\\research\\round18_train.py'
  & 'D:\\anaconda3\\envs\\mp1\\python.exe' '.\\research\\round18_evaluate.py' freeze
  & 'D:\\anaconda3\\envs\\mp1\\python.exe' '.\\research\\round18_evaluate.py' audit
  & 'D:\\anaconda3\\envs\\mp1\\python.exe' '.\\research\\round18_diagnostics.py'
  & 'D:\\anaconda3\\envs\\mp1\\python.exe' '.\\research\\round18_write_report.py'
上述入口只加载TRAIN数据，显式阻断validation/test。不要运行旧的全流程训练脚本作为本轮入口。
GPU底层实现可能造成细小数值差异；数据划分、种子、选型政策与冻结产物均已保存。
从零生成划分的逻辑也保留在round18_common.py，但复现本次比较应直接使用原plan.json。

九、当前决策
不晋级、不替换现有选定模型、不比较官方验证集；参考解监督与生成题仍保持关闭。
下一次实现改动应先在拟合/开发区检验过拟合和长推理覆盖，再重新设计审查；不能在本次950题上反复选最好结果。
源代码由Codex协助实现，使用官方数值解析与提供的嵌入，训练依赖PyTorch、NumPy、SciPy、scikit-learn；没有下载新模型或外部训练数据。
本报告是实验记录，不是完成提交的十页课程报告。
'''
    report=OUT/'真实候选联合比较器_第18轮报告.txt'
    report.write_text(text,encoding='utf-8',newline='\n')
    assets=[p for p in OUT.rglob('*') if p.is_file() and p.name!='completion_manifest.json']
    sources=list((ROOT/'research').glob('round18_*.py'))
    completion=dict(status='completed_not_promoted',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_side_gate_passed=False,validation_opened=False,test_opened=False,
        report=str(report.relative_to(ROOT)),frozen_inputs_and_code_verified=True,
        sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(assets+sources)})
    (OUT/'completion_manifest.json').write_text(json.dumps(completion,ensure_ascii=False,indent=2),encoding='utf-8')
    print(str(report));print(json.dumps({k:v for k,v in completion.items() if k!='sha256'},ensure_ascii=False))


if __name__=='__main__':main()
