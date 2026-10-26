"""Record all development outcomes and the still-sealed final test boundary."""
from data_access import ROOT,CODE
from round22_common import read,dump,sha
from pathlib import Path
import ast,json,platform,sys


def main():
    # Correct all pre-packaging size estimates, keeping the original failure record.
    p=ROOT/'experiments/round25/selection.json';s=read(p)
    if not s.get('all_record_sizes_crlf_corrected'):
        for r in s['records']:
            count=4 if r['name'].startswith('compact') else 5
            if r['name'] not in ('full','compact'):count+=4 if 'ground-answer' in r['name'] else 2
            r['bytes']+=4*count+4;r['eligible']=r['bytes']<=16*1024**2
        s['all_record_sizes_crlf_corrected']=True;dump(p,s)
    summaries=[]
    for n in range(24,29):
        out=ROOT/f'experiments/round{n}';selection=read(out/'selection.json');best=selection['selected']
        summaries.append(dict(round=n,**{k:best[k] for k in ('name','correct','n','hard_correct','hard_n','repaired','damaged')},
            configurations=len(selection['records']),asset_bytes=read(out/'asset_manifest.json')['bytes']))
    ver=read(ROOT/'experiments/round25/verification.json')
    report=ROOT/'继续打榜结果_2026-10-16_第24至28轮.txt'
    lines=['DASE7506 MP2 继续打榜结果与交接（2026-10-16，Asia/Shanghai）','',
        '目标：符合作业规则，在完整467题最终测试集上超过97%，即至少453/467。',
        '状态：目标尚未达成。本次没有打开任何测试输入、测试参考或测试评分；不能把以下验证成绩当测试成绩。',
        '当前推荐开发模型：第25轮 selected_bundle。完整官方验证752/798 = 94.235589%；旧最佳750/798 = 93.984962%。',
        '相对旧最佳修复2题、误伤0题、净增2题。相对第22轮全量训练对照749/798，修复3题、误伤0题。',
        '不同数值答案的冲突题309/355 = 87.042254%。其余443题全部保留。',
        '完整验证超过97%至少775/798，尚差23题。验证分布和测试分布不同；不存在测试97%的证据。','',
        '数据与规则',
        '只用官方训练4755题中排除既定12题后的4743题。第25轮所有最终专家均由这4743题拟合。',
        '最终包保留第22轮全量重训的3个MLP和1个文本线性模型，新增4个从零训练的96宽MLP（ground/answer各2个种子）。',
        '删除的是资产昂贵、旧融合系数为负的joint专家；没有删除训练/验证难题。',
        '新增模型采用uniform-positive交叉熵 + 0.25 BCE，35轮，种子25025/25026；与相同宽度、种子、训练曝光的原mixed损失比较。',
        '已有专家权重固定，新4个专家各加0.125的题内标准化分数；保留原parse_and_repair门控。',
        '没有外部题库、额外预训练权重、预测缓存查表、题号路由或推理时LLM/API。官方数据/解析器/评分器未改。',
        '官方在线指南于本次会话重新核对：https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md','',
        '完整实验结果（每个配置均留档，未隐藏负结果）',
        '轮次 | 本轮最佳/798 | 冲突题/355 | 相对旧最佳修复/误伤 | 配置数 | 结论']
    conclusions={24:'线性残差净增2；显式题目交互未超过线性对照',25:'同分、无误伤、资产更小，选为当前开发方案',
        26:'题目算式假设分支无额外收益，不加入推荐模型',27:'2048锚点多尺度核未优于256锚点，无额外收益',28:'最佳alpha=0，不加入第24轮残差'}
    for r in summaries:
        lines.append(f"{r['round']} | {r['correct']}/798 | {r['hard_correct']}/355 | {r['repaired']}/{r['damaged']} | {r['configurations']} | {conclusions[r['round']]}")
    lines+=['',
        '第24轮修复旧模型3题、误伤1题；第25轮修复2题且无误伤。组合二者反而下降，不能把各自修复题简单相加。',
        '第24轮二次特征消融只移除了显式题目文本SVD通道；固定候选表示和grounding仍含题目信息，不能宣称严格去题目消融。',
        '第26轮同数值不同出现位置分别保留，但有界算式枚举仍不是语义证明；未据此强制改答。',
        '第27轮修复一次NumPy float64/PyTorch float32不匹配后按原计划重跑；失败日志保留。',
        '第25轮修复Windows CRLF造成的配置字节估算差异；最终按全部实际文件计数，预测及选中方案不变。','',
        '官方复现与资源（第25轮）',
        f"额外推理资产：{ver['asset_bytes']:,}字节 = {ver['asset_MiB']:.6f} MiB，限制16 MiB。",
        f"本地Intel Core i7-11800H、四线程，新进程完整验证预测{ver['prediction_process_seconds']:.4f}秒，评分{ver['scoring_process_seconds']:.4f}秒，合计{ver['total_fresh_process_seconds']:.4f}秒。",
        '这是本机完整验证计时，不能冒充Xeon Platinum 8457C上的完整测试资格认证。',
        '官方评分与训练缓存逐项一致；抽查24个随机/变更样本的不同批量、未知ID、候选排列（按最终数值比较）、空输入、全相同候选均通过。',
        '第22/23轮冻结完成清单逐项哈希通过，历史模型未覆盖。','',
        '路径（相对于Project2）',
        '推理入口：research/round25_selector.py',
        '训练入口：research/round25_run.py',
        '推荐资产：experiments/round25/selected_bundle',
        '正式验证评分：experiments/round25/official_validation_score.json',
        '检查：experiments/round25/verification.json',
        '全部配置：experiments/round24至round28各目录的selection.json、plan.json',
        '最终数值错题：experiments/round25/validation_remaining_errors.jsonl（46题）',
        '修复清单：experiments/round25/validation_correctness_changes.jsonl',
        '源码依赖清单：experiments/round25/inference_source_manifest.json；不能只复制单个selector文件。','',
        '无需重训的复现（PowerShell，从Project2目录执行；换输出文件避免覆盖记录）',
        "$env:PYTHONPATH = 'E:\\New Life\\HKU\\7506\\Project2\\.venv\\Lib\\site-packages'",
        "$env:OMP_NUM_THREADS='4'; $env:OPENBLAS_NUM_THREADS='4'; $env:MKL_NUM_THREADS='4'; $env:NUMEXPR_NUM_THREADS='4'",
        "& 'D:\\anaconda3\\envs\\mp1\\python.exe' -X utf8 DASE7506-MP2-main/code/predict.py --method research/round25_selector.py --checkpoint experiments/round25/selected_bundle --split validation --output experiments/round25/reproduced_predictions.json",
        "& 'D:\\anaconda3\\envs\\mp1\\python.exe' -X utf8 DASE7506-MP2-main/code/score_predictions.py --split validation --predictions experiments/round25/reproduced_predictions.json --output experiments/round25/reproduced_score.json",'',
        '重新训练：保留第22轮基线与第4轮历史配置，使用同一环境运行round25_run；为避免覆盖冻结结果，应在新的输出目录运行：',
        "python -c \"import sys; sys.path.insert(0,'research'); import round25_run as r; r.OUT=r.ROOT/'experiments/round25_reproduction'; r.main()\"",
        '旧专家可从research/round22_train.py的全量fit_experts配方复现；它们不是950题留出模型。','',
        '解释与后续边界',
        '本次共比较162个验证配置（含对照），验证集又一次被用于方法选择。净增2题很小，不能推断未见测试提升幅度。',
        '修复题206的选中候选仍有错误叙述，只是最终数值正确；不能把修复记作完整语义推理成功。',
        '当前复核页仍是第三批1题；未覆盖页面、未将未返回的第四/第五批当成人工金标准。',
        '最终测试继续封存。需要继续研究时，应新增可信语义监督或更可靠的题目/候选局部对齐，避免继续扩大同类融合网格。',
        '没有提交GitHub issue，没有发布成绩，也没有声称达到97%。',
        'AI辅助披露：Codex编写实验代码、检查、分析和本交接。需在最终课程报告中披露。']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Freeze all local source files reachable by Python imports (conservative,
    # includes imports inside unused training/analysis functions).
    search=[ROOT/'research',ROOT/'starter',CODE];seen=set();todo=[ROOT/'research/round25_selector.py']
    while todo:
        f=todo.pop()
        if f in seen:continue
        seen.add(f)
        tree=ast.parse(f.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else ([node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
            for name in names:
                found=next((d/(name+'.py') for d in search if (d/(name+'.py')).exists()),None)
                if found and found not in seen:todo.append(found)
    dump(ROOT/'experiments/round25/inference_source_manifest.json',dict(hashes={str(f.relative_to(ROOT)):sha(f) for f in sorted(seen)}))
    import numpy,scipy,sklearn,torch
    dump(ROOT/'experiments/round25/environment.json',dict(python=sys.version,platform=platform.platform(),numpy=numpy.__version__,
        scipy=scipy.__version__,sklearn=sklearn.__version__,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,seed=[25025,25026]))
    dump(ROOT/'experiments/continuation_20261016.json',dict(target_test_minimum_correct=453,test_n=467,test_opened=False,
        target_achieved=False,recommended_round=25,validation_correct=752,validation_n=798,rounds=summaries,
        report=str(report.relative_to(ROOT))))
    for n in range(24,29):
        out=ROOT/f'experiments/round{n}';files=[f for f in out.rglob('*') if f.is_file() and f.name!='completion_manifest.json']
        source=list(seen)+[ROOT/f'research/round{n}_run.py',ROOT/f'research/round{n}_selector.py',Path(__file__),report]
        if n==26:source.append(ROOT/'research/round26_programs.py')
        dump(out/'completion_manifest.json',dict(completed=True,goal_97_achieved=False,test_opened=False,
            hashes={str(f.relative_to(ROOT)):sha(f) for f in files+source}))
    print(report,flush=True)


if __name__=='__main__':main()
