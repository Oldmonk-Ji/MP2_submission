"""Summarize this turn without claiming a gain where none was measured."""
from data_access import ROOT
import json,hashlib
OUT=ROOT/'experiments/round12'

def main():
    def read(r,name):return json.loads((ROOT/'experiments'/r/name).read_text())
    r9=read('round9','selection.json');r10=read('round10','selection.json');r11=read('round11','selection.json');r12=read('round12','decision.json');r13=read('round13','selection.json');ev=read('round12','validation_result.json')
    original=ROOT/'experiments/round4/selected_bundle';manifest=read('round4','asset_manifest.json')
    assert all(hashlib.sha256((original/name).read_bytes()).hexdigest()==v['sha256'] for name,v in manifest.items())
    report=f'''本轮优化记录（训练集与验证集；测试集未访问）

最终结果：没有找到超过现有750/798（93.98496%）验证成绩的新方案。
保留选择器 research/round4_selector.py 和 experiments/round4/selected_bundle。
99%要求至少791/798，仍相差41题；本轮没有缩小差距。

本轮已实际运行的路线
1. 答案分组：把官方解析器判为同一数值答案的候选合为一组，结合五个原有评分模块的最大/平均分、概率支持、分歧和算术证据，用很小的LightGBM学习修正。避免把同一个答案的多个写法当作互不相关的结论。
   初筛18种组合，修正后的训练留出结果最佳{r9['selected']['groups']['internal']['correct']}/951，基线{r9['baseline_internal']['correct']}/951。按预先固定的三折复查流程再训练，4755题上基线4393题、新方案{r12['summary']['group']['correct']}题，修正{r12['summary']['group']['fixed']}、损失{r12['summary']['group']['lost']}，净增3题。两个折净改善、一个折净退步。
2. 教师式概率融合：对原有模块分别学习温度和融合权重，对照概率平均/对数概率平均、按题标准化/保留原始置信度、同答案概率相加/平均，共8种预先固定形式。最佳训练留出{r10['selected']['probability']['correct']}/951；三折复查修正36题、损失36题，整体没有提升。MP1中概率融合的经验没有在本数据上获得相同收益。
3. 题意词与算式交互：把题目局部词组、数量依赖符号和算式结构作为交叉特征，学习剩余/总量/折扣等措辞与推理是否匹配。使用固定哈希空间和线性偏移，不引入预训练模型。最佳有上下文训练留出{r11['selected']['True']['internal']['correct']}/951；去掉上下文的消融{r11['selected']['False']['internal']['correct']}/951。没有过初筛，未拿验证集调参。
4. 置信度修正：给当前按题标准化的分数补充全局定标原始分数、概率和置信度交互，允许有正有负的小修正。最佳训练留出{r13['selected']['confidence']['correct']}/951；只有标准化分数的消融{r13['selected']['norm_only']['correct']}/951，没有通过初筛。

验证与资源
只有经过训练三折复查的答案分组路线进入验证比较；逐候选版本作为固定消融。两者均750/798，冲突题307/355。正确数没有改善。
分组方案资产{ev['variants']['answer_group']['asset_bytes']}字节（{ev['variants']['answer_group']['asset_bytes']/2**20:.3f}MiB）。本机i7-11800H、四线程，官方独立预测进程加评分进程耗时{ev['fresh_predict_and_score_seconds']:.2f}秒。当地验证规模的资源测量不能替代参考机器上的最终资格测量。
已有模型增加一个113KB左右的修正器并未改善成绩，因此未替换现有方案。

实现检查及纠错
初版答案分组在同一答案包含多份推理时，会随候选顺序选中不同推理文本，再影响算术复核。第一次候选打乱检查失败，该初版被否决；before_order_fix保存了当时记录，不把它算作有效成绩。
修复采用每个数值答案组内原模型评分最高的推理作为稳定代表，必要时用文本排序处理完全相同分数。没有根据某道验证题的标签修改规则或阈值。
修复后重跑训练筛选、三折复查及验证工程回归，正式预测和评分均完成。已检查未知ID、不同批量、空输入、候选排列、同答案分组稳定性、概率归一化、LightGBM分组边界、旧资产校验以及测试文件访问防护。

实验可信度
训练筛选复用了先前的基模型OOF分数，后续meta折与原基模型折不完全嵌套；早期951题筛选与三折复查也有重叠。因此这些是开发证据，不应宣称为独立未见数据上的无偏估计。纠错后的三折净增3题也不构成稳定提升证明。
验证集在历史多轮开发中已经被复用。此次没有根据验证结果继续搜索分组权重、树数或融合温度；相同方案的再次运行是修复顺序依赖后的必要复测。
所有测试输入和测试参考均未访问。旧模型资产全部校验一致。失败的关系提取器没有接入本轮选择器。没有扩大人工复核题包，因为本轮没有新的人工作答需求。

据此调整下一步方向
现有分数重新混合的空间很小，继续只调融合权重缺少收益证据。需要给选择器新增可靠的信息来源。
优先完成统一的事件关系监督：赠予者/收受者、商品批次、事件时间与状态时间分开、单价分母、所求量。之前的owner和stage混合了多种含义，难以建立有效学习任务。
明确的约束和参考解采用的约定应分别保存，歧义关系保留备选解释；给出“明确”结论需要有审查证据。不能因分类器歧义分数低就自动确认题意。
如引入新的事件模型，应先制定统一标注规范、扩充训练监督并冻结全新审查集，再通过提取质量门槛。当前已打开的审查题只适合回归诊断。

重现（工作目录Project2，Python为.venv/Scripts/python.exe）
research/round9_train.py
research/round10_fusion.py
research/round11_train.py
research/round12_confirm.py
research/round13_confidence.py
research/round12_evaluate.py
research/check_fusion.py
各round目录含plan、selection/decision、frozen哈希和训练权重；round12另含官方验证预测与评分结果。训练特征缓存不属于推理资产。'''
    (OUT/'本轮优化报告.txt').write_text(report,encoding='utf-8')
    print(json.dumps(dict(selected=ev['selected_path'],correct=ev['baseline']['correct'],new_method_selected=ev['new_method_selected'],test_access=False),ensure_ascii=False))
if __name__=='__main__':main()
