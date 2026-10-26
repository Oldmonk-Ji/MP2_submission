"""Validation-only reliability, fresh-process timing, and round-3 research report."""
from data_access import ROOT,CODE,load
import hashlib,html,json,shutil,subprocess,time
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from features import metadata
from confidence import confidence_features
from reliability import calibration_stats
from round3_selector import scores,predict,predict_with_confidence
from train_models import SEED,metrics

OUT=ROOT/'experiments/round3'


def paired(a,b):
    rng=np.random.default_rng(SEED);diff=[]
    for _ in range(5000):
        ids=rng.integers(0,len(a),len(a));diff.append(float((a[ids]-b[ids]).mean()))
    return dict(fixed_old_errors=int(((a==1)&(b==0)).sum()),lost_old_correct=int(((a==0)&(b==1)).sum()),
                gain_pp=float(100*(a-b).mean()),gain_95ci_pp=(100*np.quantile(diff,[.025,.975])).tolist())


def main():
    b,_,y=load('train');bv,av,yv=load('validation');mt=metadata(b);mv=metadata(bv)
    result=json.loads((OUT/'stack_results.json').read_text());z=np.load(OUT/'stack_scores.npz')
    bundle=OUT/'stack_bundle'
    st=z['meta_oof'];sv=z['validation'];xt,ct=confidence_features(st,mt);xv,cv=confidence_features(sv,mv)
    sc=StandardScaler().fit(xt);correct_t=y[np.arange(len(y)),ct];correct_v=yv[np.arange(len(yv)),cv]
    cal=LogisticRegression(C=.1,max_iter=1000,random_state=SEED).fit(sc.transform(xt),correct_t)
    w=cal.coef_[0]/sc.scale_;bias=cal.intercept_[0]-sc.mean_@w
    np.savez(bundle/'confidence_calibrator.npz',weight=w,bias=bias)
    conf=cal.predict_proba(sc.transform(xv))[:,1]
    old=np.load(ROOT/'experiments/round2/reliability/crossfit_scores.npz')
    old_correct=old['validation_correct']
    detail=dict(calibration_fit='TRAIN meta-crossfit scores only; base folds reused, not a fully nested estimate',
        calibrated=calibration_stats(conf,correct_v),conflict_calibrated=calibration_stats(conf[mv['hard']],correct_v[mv['hard']]),
        paired_to_previous=paired(correct_v,old_correct),test_access=False)
    oldresult=json.loads((ROOT/'experiments/round2/reliability/results.json').read_text())
    records=json.loads((OUT/'validation_comparison.json').read_text())+json.loads((OUT/'joint_validation.json').read_text())
    alternatives=[]
    for r in records:
        if r['name']=='internal_selected_blend':continue
        s=np.load(OUT/'full'/r['name']/'validation_scores.npz')['scores'];c=np.where(mv['valid'],s,-1e30).argmax(1)
        correct=yv[np.arange(len(yv)),c]
        r['paired_to_previous']=paired(correct,old_correct);alternatives.append(correct)
    alternatives.extend([correct_v,old_correct]);oracle=np.stack(alternatives).max(0)
    detail['diagnostic_any_evaluated_method_correct']=dict(correct=int(oracle.sum()),n=len(oracle),
        note='Uses validation labels; not deployable, not a ceiling for future methods.')
    failures=[]
    for i in np.flatnonzero(oracle==0):
        failures.append(dict(index=int(i),question=bv[i]['question'],candidates=bv[i]['candidates'],
                             correct_candidates=np.flatnonzero(yv[i]).tolist()))
    (OUT/'remaining_shared_errors.json').write_text(json.dumps(failures,ensure_ascii=False,indent=2),encoding='utf-8')
    # Consistency, permutation equivariance, and variable-size contract for the stack.
    s=scores(bv,av,bundle);assert np.allclose(s,sv,atol=2e-4,rtol=1e-4)
    ids=[0,13,52,77,111,250,520];bb=[dict(bv[i],id='new-'+str(i)) for i in ids];aa={'embeddings':av['embeddings'][ids]}
    selected,confidence=predict_with_confidence(bb,aa,bundle)
    assert selected==cv[ids].tolist() and np.allclose(confidence,conf[ids],atol=2e-5)
    perm=np.array([2,0,3,1]);pb=[dict(r,candidates=[dict(r['candidates'][int(j)],index=k) for k,j in enumerate(perm)]) for r in bb]
    assert np.allclose(scores(pb,{'embeddings':aa['embeddings'][:,perm]},bundle),s[ids][:,perm],atol=2e-4,rtol=1e-4)
    for n in (1,3,7):assert predict(bb[:n],{'embeddings':aa['embeddings'][:n]},bundle)==selected[:n]
    assert predict([],{'embeddings':np.empty((0,4,1024))},bundle)==[]
    detail['stack_contract_checks']='passed; export, confidence, unknown IDs, permutation, N=0/1/3/7'
    python=ROOT/'.venv/Scripts/python.exe';timings=[]
    for k in range(3):
        pred=OUT/f'official-validation-predictions-{k}.json';score=OUT/f'official-validation-score-{k}.json';tick=time.perf_counter()
        subprocess.run([str(python),str(CODE/'predict.py'),'--method',str(ROOT/'research/round3_selector.py'),
            '--checkpoint',str(bundle),'--split','validation','--output',str(pred)],check=True,capture_output=True)
        subprocess.run([str(python),str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(pred),
            '--output',str(score)],check=True,capture_output=True)
        timings.append(time.perf_counter()-tick)
        official=json.loads(score.read_text());assert official['correct']==int(correct_v.sum())
        assert json.loads(pred.read_text())['selected']==cv.tolist()
    detail['official_validation']=official;detail['local_fresh_prediction_plus_score_seconds']=timings
    assets={str(f.relative_to(bundle)):dict(bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in bundle.rglob('*') if f.is_file()}
    detail['asset_bytes']=sum(x['bytes'] for x in assets.values());detail['within_asset_limit']=detail['asset_bytes']<=16*2**20
    (OUT/'stack_asset_manifest.json').write_text(json.dumps(assets,indent=2))
    detail['records']=records
    detail['uncertainty_note']='Paired bootstrap intervals condition on selected models; repeated validation selection is not corrected.'
    # Preserve previous winner unless a valid new method actually beats it.
    candidates=[dict(name='previous_best',correct=int(old_correct.sum()),path=str(ROOT/'experiments/round2/selected_bundle'),
                     selector=str(ROOT/'research/selector.py'),asset_bytes=oldresult['asset_bytes'])]
    for r in records:
        if r['name']=='internal_selected_blend':continue
        path=OUT/'full'/r['name'];assetsize=sum((path/f).stat().st_size for f in ('config.json','model.npz','vocabulary.json') if (path/f).exists())
        if assetsize<=16*2**20:
            candidates.append(dict(name=r['name'],correct=r['validation']['correct'],path=str(path),selector=str(ROOT/'research/round3_selector.py'),asset_bytes=assetsize))
    if detail['within_asset_limit']:
        candidates.append(dict(name='train_oof_stack',correct=int(correct_v.sum()),path=str(bundle),selector=str(ROOT/'research/round3_selector.py'),asset_bytes=detail['asset_bytes']))
    best=max(candidates,key=lambda r:(r['correct'],-r['asset_bytes']))
    best.update(n=798,accuracy=100*best['correct']/798,target_correct=791,additional_correct_needed=791-best['correct'],test_access=False)
    (OUT/'selection.json').write_text(json.dumps(best,indent=2));detail['selected']=best
    (OUT/'reliability.json').write_text(json.dumps(detail,indent=2))
    internal=json.loads((OUT/'internal_selection.json').read_text())['records']
    joint=json.loads((OUT/'joint_selection.json').read_text())['records']
    fits=dict(internal_configurations=len(internal)+len(joint),full_refits=len(records)-1,
              additional_oof_fits=len(result['oof_ledger']),
              stack_member_ablations=5,unused_planned_experiment='joint internal scalar fusion replaced by OOF stacking',
              completed_recorded_fit_seconds=sum(r['seconds'] for r in internal+joint+records if 'seconds' in r)+sum(r['seconds'] for r in result['oof_ledger']),
              overhead_note='Excludes parsing/analysis, one interrupted dense fit and two failed export attempts; those attempts were rerun. No hidden validation grid.',
              cpu='Intel Core i7-11800H; four library threads',gpu='one NVIDIA RTX A2000 Laptop 4 GiB for joint training; inference CPU only')
    (OUT/'ledger.json').write_text(json.dumps(fits,indent=2))
    make_html(records,detail,result,oldresult,fits)
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'research').glob('round3_*.py')}
    (OUT/'source_hashes.json').write_text(json.dumps(sources,indent=2))
    print(json.dumps(dict(selected=best,stack=result['validation'],paired=detail['paired_to_previous'],seconds=timings,assets=detail['asset_bytes']),indent=2))


def make_html(records,detail,stack,old,fits):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=OUT/'report';report.mkdir(exist_ok=True)
    selected=detail['selected'];rows=[('Previous best',745,302)]
    for r in records:
        if r['name']=='internal_selected_blend':continue
        rows.append((r['name'],r['validation']['correct'],round(r['validation']['hard_accuracy']/100*355)))
    rows.append(('Train-only OOF stack',stack['validation']['correct'],round(stack['validation']['hard_accuracy']/100*355)))
    fig,ax=plt.subplots(figsize=(10,6));labels=[r[0] for r in rows];values=[100*r[1]/798 for r in rows]
    ax.barh(labels[::-1],values[::-1],color=['#247b98' if x!='Previous best' else '#d48b2f' for x in labels[::-1]])
    ax.set_xlim(78,100);ax.axvline(99,color='#ad3341',ls='--',label='99% target');ax.set_xlabel('Official validation selection accuracy (%)')
    for i,v in enumerate(values[::-1]):ax.text(v+.12,i,f'{v:.2f}',va='center',fontsize=9)
    ax.legend();fig.tight_layout();fig.savefig(report/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,('calibrated','conflict_calibrated'),('All validation questions','Conflicting-answer questions')):
        for obj,label in ((old,'Previous best'),(detail,'OOF stack')):
            bins=obj[key]['bins'];ax.plot([v['confidence'] for v in bins],[v['accuracy'] for v in bins],'o-',label=label)
        ax.plot([.45,1],[.45,1],'--',color='gray');ax.set(xlim=(.45,1.01),ylim=(.45,1.01),xlabel='Mean calibrated confidence',ylabel='Observed accuracy',title=title);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(report/'reliability.png',dpi=160);plt.close(fig)
    table=''.join(f'<tr><td>{html.escape(name)}</td><td>{c}/798 ({100*c/798:.2f}%)</td><td>{h}/355 ({100*h/355:.2f}%)</td></tr>' for name,c,h in rows)
    internal=json.loads((OUT/'joint_selection.json').read_text())['records']
    training=''.join(f'<tr><td>{r["name"]}</td><td>{r["internal"]["correct"]}/951</td><td>{r["seconds"]:.1f}s</td></tr>' for r in internal)
    ablations=json.loads((OUT/'stack_ablations.json').read_text())
    pruning=''.join(f'<tr><td>{r["name"]}</td><td>{r["validation"]["correct"]}/798</td><td>{r["member_asset_bytes"]/2**20:.2f}</td></tr>' for r in ablations)
    d=detail['paired_to_previous'];cr=detail['conflict_calibrated'];weights=', '.join(f'{m["name"]}: {m["weight"]:.3f}' for m in stack['members'])
    reliability_rows=''.join(f'<tr><td>{scope}</td><td>{metric}</td><td>{old[key][field]:.4f}</td><td>{detail[key][field]:.4f}</td></tr>'
        for scope,key in [('全部题','calibrated'),('冲突题','conflict_calibrated')]
        for metric,field in [('Brier（低更好）','brier'),('ECE（低更好）','ece_equal_count_10'),('AUROC（高更好）','auroc')])
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>MP2 第三轮实验</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#1b2938}}h1,h2{{color:#17556a}}table{{border-collapse:collapse;width:100%;margin:18px 0}}td,th{{padding:8px 12px;border-bottom:1px solid #d6e1e7;text-align:left}}.card{{padding:20px;background:#edf5f7;border-radius:12px}}img{{width:100%}}code{{font-size:.9em}}small{{color:#596574}}</style>
<h1>MP2 第三轮：新证据是否真正有用？</h1><p>2026-10-14 · 只使用训练集和验证集；未读取测试输入或参考答案。</p>
<div class="card"><b>当前选择：{html.escape(selected['name'])}，{selected['correct']}/798 = {selected['accuracy']:.2f}%。</b>
<p>99% 目标要求至少 791/798，目前还需净增 {selected['additional_correct_needed']} 题。验证集包含 443 道无有效答案冲突的容易题；355 道冲突题单独报告。总体验证准确率不能直接当作测试成绩，官方说明测试题均有答案冲突。</p></div>
<h2>问题与方法</h2><p>本轮检验：按相同最终答案汇总四份候选表示、加入数字来源/算术链局部证据，以及训练一个轻量词组分支，是否补充现有固定向量。所有词表、IDF、标准化和模型权重只从训练部分拟合。数字值替换成题目数字槽位或“新数字”，减少记忆具体数值的机会。</p>
<p>算术检查只核验可安全解析的局部数值等式、数值覆盖与结果复用，不声称理解完整题意。变量表达式不能截成数值后缀；官方答案解析器保持原样。词组分支为从零训练的 TF-IDF 投影和 MLP，使用一张 RTX A2000；没有外部预训练权重或生成 API。</p>
<h2>验证集比较</h2><img src="comparison.png" alt="各方法验证集准确率"><table><tr><th>方法</th><th>全部验证题</th><th>答案冲突题</th></tr>{table}</table>
<p>四个 35 轮 MLP 使用同一训练配置，分别检验新增算术证据、按答案汇总，以及两者组合。文本独立排序器使用题内正负候选差分训练。joint_answer_only 只移除了<strong>新增文本分支</strong>的推理文本；仍保留官方全文表示，因此不能解释成“整个系统不读推理”。</p>
<h2>选择流程与训练过拟合</h2><p>先固定训练集内 3804/951 题划分，比较 8 个 MLP 配置、8 个线性文本配置。文本融合系数在训练留出集选择为 0。神经文本分支在初始留出实验中过拟合，因此在查看该分支官方验证结果之前增加 10 轮和显式题目交互对照，共 9 个配置。内部最优为 joint_reasoning_10。此训练划分沿用了此前开发划分，并非从未接触的独立审计集。</p>
<table><tr><th>神经文本配置</th><th>训练内部留出集正确数</th><th>训练与导出时间</th></tr>{training}</table>
<h2>训练集折外融合</h2><p>较弱的单模型仍可能修正不同错误，因此用 3 折训练集折外分数拟合一个固定 C=0.1 的配对逻辑回归。没有用验证标签拟合权重。成员分数按每题标准化，权重为：{html.escape(weights)}。</p>
<p>相对上一轮最优，融合修正 {d['fixed_old_errors']} 道旧错题，同时错掉 {d['lost_old_correct']} 道旧对题，净变化 {d['gain_pp']:.2f} 个百分点；配对 bootstrap 95% 区间 [{d['gain_95ci_pp'][0]:.2f}, {d['gain_95ci_pp'][1]:.2f}]。区间没有修正反复使用验证集选择方案的偏差，不代表独立泛化保证。</p>
<p>再分别移除一个成员，并仅用训练折外分数重新拟合其余权重。完整融合仍最高；去掉较大的神经文本分支只少对 1 题，资产约减半，是可保留的紧凑备选。下面五个消融均计入验证集选择次数，不能当作新的独立验证。</p><table><tr><th>移除成员</th><th>验证正确数</th><th>成员资产 MiB（另加少量配置）</th></tr>{pruning}</table>
<h2>可靠性与资源</h2><p>融合置信度校准只用训练集元模型交叉预测。基模型折被复用，这不是完整嵌套的无偏性能估计。验证集 Brier={detail['calibrated']['brier']:.4f}，十个等样本量分箱 ECE={detail['calibrated']['ece_equal_count_10']:.4f}；冲突题 Brier={cr['brier']:.4f}，ECE={cr['ece_equal_count_10']:.4f}。高置信子集准确率不等于全部题目准确率。</p>
<table><tr><th>范围</th><th>指标</th><th>上一轮最优</th><th>本轮融合</th></tr>{reliability_rows}</table><img src="reliability.png" alt="上一轮与本轮校准可靠性对比"><p>准确率小幅增加，但可靠性指标并非全面改善：冲突题 ECE 变差，识别错题的 AUROC 也下降。不能仅凭更高准确率宣称系统更善于认识自己的错误。</p>
<p>融合资产 {detail['asset_bytes']/2**20:.2f} MiB（上限 16 MiB）；本机四线程全验证集新进程预测＋独立评分耗时 {', '.join(f'{v:.2f}' for v in detail['local_fresh_prediction_plus_score_seconds'])} 秒。该计时未使用测试集，不能冒充参考 Xeon 上的正式资格认证。CPU：i7-11800H；GPU 仅用于训练。</p>
<p>所有新方法均通过未知 ID、不同批量大小、候选换序、数值有限性和权重导出一致性检查。官方原始数据、解析器及评分器未修改。</p>
<h2>从 MP1 迁移的结论</h2><p>本轮实际检验了更丰富的候选比较、额外文本能力、训练时长与基于训练折外预测的融合。增加分支和训练时长不能自动形成更强教师；训练损失接近零同时留出集下降，应按过拟合处理。重复候选不增加真实语义覆盖，本轮也未把候选重采样称为新的合成问题。未使用外部教师、测试答案或按题查表。</p>
<p>优先继续改善能跨题泛化的题意条件验证，而非继续堆同类模型。当前局部算术核验不能区分“算式本身正确但漏了条件”的解法；本轮数据不支持已达到 99% 的说法。</p>
<h2>复现与披露</h2><p>运行方法见 research/第三轮运行说明.txt。配置与权重保存在 experiments/round3，包含内部选择、失败方法、折外训练记录、官方验证评分、资产及源码哈希。AI 协助了特征设计、实现、实验运行和分析；使用 NumPy、SciPy、scikit-learn、PyTorch 与原项目官方数据/解析器。</p>
<small>完成记录：{fits['internal_configurations']} 个内部配置、{fits['full_refits']} 个全训练重拟合、{fits['additional_oof_fits']} 个新增折外模型。累计已记录拟合耗时 {fits['completed_recorded_fit_seconds']:.1f} 秒，不含预处理、一次中断及两次导出失败重跑开销。任何方法至少选对一题的标签辅助并集仅作诊断，不可部署，也不是未来方法上限。</small></html>'''
    (report/'实验报告.html').write_text(text,encoding='utf-8')


if __name__=='__main__':main()
