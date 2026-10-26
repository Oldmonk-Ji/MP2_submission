"""Nonlinear readout ablation; final training and evaluation population unchanged."""
from round22_common import ROOT,CODE,OUT as R22,INCUMBENT,read,dump,sha,load
from round22_fusion import evaluate
from round23_selector import design
from round22_selector import side_features,component_scores
from features import metadata
from pathlib import Path
import lightgbm as lgb
import numpy as np
import datetime,functools,json,os,shutil,subprocess,sys,time
import round4_gates

OUT=ROOT/'experiments/round23'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'plan.json').exists()
    weights=[m['weight'] for m in read(INCUMBENT/'base/config.json')['members']]
    p=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        training_n=4743,validation_n=798,excluded_training_ids=read(R22/'plan.json')['excluded_training_ids'],
        evaluation_exclusions=[],test_opened=False,
        method='Residual LightGBM LambdaRank on honest TRAIN cross-predictions; final experts already fitted on all 4743. init_score is fixed baseline. Predict adds tree output only; no new large expert.',
        features=['scores','conditional'],leaves=[7,15],rounds=[64,128],scales=[.1,.25,.5,1.],
        params=dict(objective='lambdarank',metric='ndcg',ndcg_eval_at=[1],lambdarank_truncation_level=1,
            label_gain=[0,1],learning_rate=.03,min_data_in_leaf=100,lambda_l2=10.,min_gain_to_split=.01,
            max_bin=63,verbosity=-1,num_threads=4,seed=22023,deterministic=True,force_col_wise=True),
        selection='All 798 official validation questions. Most correct, fewer harms, smaller scale, smaller asset. Full record of all trials.',
        limitations='Validation selection is not an independent test. No selection by old 938/950 internal scores.',
        source_hashes={str(f.relative_to(ROOT)):sha(f) for f in [R22/'frozen.json',R22/'oof.npz',R22/'full/eval_components.npz']})
    dump(OUT/'plan.json',p)
    tb,_,ty,tids=load('train');vb,va,vy,vids=load('validation')
    with np.load(R22/'oof.npz') as z:
        tc=z['components'];assert np.array_equal(z['ids'],tids) and np.array_equal(z['labels'],ty)
    with np.load(R22/'full/eval_components.npz') as z:vc=z['scores']
    with np.load(R22/'train_side_features.npz') as z:tq,te=z['cues'],z['evidence']
    vq,ve=side_features(vb)
    with np.load(R22/'validation_comparison.npz') as z:old=z['incumbent_choice'];base_choice=z['refit_original_choice']
    hard=metadata(vb)['hard'];records=[];trained=[];choices={'control':base_choice}
    configs={'control':dict(base_weights=weights,features='scores',scale=0.,gate='parse_and_repair',gate_margin=4)}
    records.append(dict(name='control',tree=None,scale=0.,tree_bytes=0,**evaluate(base_choice,old,vy,hard)))
    original_trace=round4_gates.trace;round4_gates.trace=functools.lru_cache(maxsize=8000)(original_trace)
    try:
        for kind in p['features']:
            tx,tbase=design(tb,tc,weights,kind,tq,te);vx,vbase=design(vb,vc,weights,kind,vq,ve)
            for leaves in p['leaves']:
                for rounds in p['rounds']:
                    name=f'{kind}_l{leaves}_r{rounds}';path=OUT/'models'/name;path.mkdir(parents=True,exist_ok=True)
                    start=time.perf_counter()
                    dataset=lgb.Dataset(tx.reshape(-1,tx.shape[-1]),label=ty.astype(int).reshape(-1),group=np.full(len(tb),4),init_score=tbase.reshape(-1))
                    model=lgb.train(dict(p['params'],num_leaves=leaves),dataset,num_boost_round=rounds)
                    model.save_model(str(path/'model.txt'));delta=model.predict(vx.reshape(-1,vx.shape[-1]),raw_score=True,num_threads=4).reshape(-1,4)
                    tree_bytes=(path/'model.txt').stat().st_size
                    trained.append(dict(name=name,seconds=time.perf_counter()-start,features=tx.shape[-1],trees=model.num_trees(),bytes=tree_bytes))
                    for scale in p['scales']:
                        key=f'{name}_s{scale}';c=round4_gates.apply_gate(vb,vbase+scale*delta,'parse_and_repair',4)
                        choices[key]=c;configs[key]=dict(base_weights=weights,features=kind,scale=scale,gate='parse_and_repair',gate_margin=4)
                        r=dict(name=key,tree=name,scale=scale,tree_bytes=tree_bytes,**evaluate(c,old,vy,hard));records.append(r)
                        print('VAL TREE',key,r['correct'],'fix/harm',r['repaired'],r['damaged'],flush=True)
    finally:round4_gates.trace=original_trace
    winner=max(records,key=lambda r:(r['correct'],-r['damaged'],-r['scale'],-r['tree_bytes']))
    bundle=OUT/'selected_bundle';bundle.mkdir(exist_ok=True)
    for src in (R22/'selected_bundle').rglob('*'):
        if src.is_file():
            dst=bundle/src.relative_to(R22/'selected_bundle');dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    if winner['tree']:shutil.copyfile(OUT/'models'/winner['tree']/'model.txt',bundle/'readout_tree.txt')
    dump(bundle/'tree_fusion.json',configs[winner['name']])
    assets=[f for f in bundle.rglob('*') if f.is_file()];size=sum(f.stat().st_size for f in assets);assert size<=16*1024**2,size
    np.savez(OUT/'validation_comparison.npz',ids=vids,labels=vy,incumbent=old,**choices)
    result=dict(selected=winner,records=records,training=trained,training_n=4743,validation_n=798,
        inference_asset_bytes=size,incumbent_replaced=False,test_opened=False,lightgbm=lgb.__version__)
    dump(OUT/'selection.json',result)
    dump(OUT/'frozen.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        selected=winner['name'],test_opened=False,hashes={str(f.relative_to(ROOT)):sha(f) for f in assets+[OUT/'plan.json',OUT/'selection.json',ROOT/'research/round23_selector.py']}))
    # Reproduce all predictions in a fresh official process; complete population.
    env=dict(os.environ)
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    pred=OUT/'official_validation_predictions.json';score=OUT/'official_validation_score.json';timing=[]
    for args in [[str(CODE/'predict.py'),'--method',str(ROOT/'research/round23_selector.py'),'--checkpoint',str(bundle),'--split','validation','--output',str(pred)],
                 [str(CODE/'score_predictions.py'),'--split','validation','--predictions',str(pred),'--output',str(score)]]:
        start=time.perf_counter();cp=subprocess.run([sys.executable]+args,cwd=CODE,env=env,capture_output=True,text=True,encoding='utf-8');timing.append(time.perf_counter()-start)
        assert cp.returncode==0,(cp.stdout,cp.stderr);print(cp.stdout.strip(),flush=True)
    actual=read(score);assert actual['correct']==winner['correct'] and actual['n']==798
    assert np.array_equal(read(pred)['selected'],choices[winner['name']])
    checks=dict(fresh_official_predictions_equal_cached=True,no_evaluation_deletion=True,all_final_experts_train_4743=True,
        asset_bytes=size,asset_MiB=size/1024**2,local_cpu='Intel Core i7-11800H',threads=4,
        fresh_validation_prediction_seconds=timing[0],fresh_validation_score_seconds=timing[1],total_seconds=sum(timing),
        not_reference_machine_test_timing=True,test_opened=False)
    prior=read(R22/'completion_manifest.json');assert all(sha(ROOT/k)==v for k,v in prior['hashes'].items());checks['round22_unchanged']=True
    dump(OUT/'verification.json',checks)
    changes=[];errors=[]
    for i,(row,c,o) in enumerate(zip(vb,choices[winner['name']],old)):
        newok,oldok=bool(vy[i,c]),bool(vy[i,o])
        r=dict(validation_index=i,question=row['question'],chosen_index=int(c),previous_index=int(o),chosen_correct=newok,previous_correct=oldok,
            kind='repaired' if newok and not oldok else 'damaged' if oldok and not newok else 'same_correctness',
            candidate_texts=[q['text'] for q in row['candidates']])
        if not newok:errors.append(r)
        if c!=o:changes.append(r)
    for name,rows in [('validation_changes',changes),('validation_remaining_errors',errors)]:
        (OUT/f'{name}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    lines=['全量训练后的非线性融合对照（第 23 轮）','',
        '最终模型用官方训练集 4755 道中除约定 12 道之外的全部 4743 道训练。不存在永久内部审查留出。',
        '官方验证 798 道一题不删。官方测试 467 道仍未打开。之前 938/950 的子集成绩不参与本轮选择。',
        '只增加小型树融合层，专家模型仍是第 22 轮全量训练的六个。树使用训练集分组交叉预测拟合，最终专家仍看全量 4743 道。',
        '目标为残差排序：学习原模型在哪些证据组合下需要改判；对照仅分数特征与加入题目／算术证据的条件特征。',
        '',f"旧方案：750/798 = {750/798*100:.4f}%。",
        f"第 22 轮最佳：749/798。第 23 轮 {winner['name']}：{actual['correct']}/798 = {actual['accuracy']:.4f}%。",
        f"相对旧方案修复 {winner['repaired']}、误伤 {winner['damaged']}、净增 {winner['net']:+d}。",
        f"有不同数值答案的题：{winner['hard_correct']}/{winner['hard_n']} = {winner['hard_accuracy']:.4f}%。",
        f"验证集超过 95% 至少需要 759/798，当前差 {max(0,759-actual['correct'])} 道。",
        f"推理资产 {size:,} 字节，{size/1024**2:.4f} MiB。本地 i7-11800H 四线程，完整 798 题新进程预测＋评分 {sum(timing):.3f} 秒。",
        '本地验证时间不等于参考机器上的正式测试资格认证。',
        '所有验证配置公开如下；这是用验证选方案，不是独立测试成绩。旧方案保留，所有候选保留，没有因门槛删除结果。',
        '', 'name | correct/798 | repairs | harms | net']
    lines += [f"{r['name']} | {r['correct']}/798 | {r['repaired']} | {r['damaged']} | {r['net']:+d}" for r in records]
    lines += ['', '复现（Project2 工作目录，先将 OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS 设为 4）：',
        'python DASE7506-MP2-main/code/predict.py --method research/round23_selector.py --checkpoint experiments/round23/selected_bundle --split validation --output experiments/round23/reproduced_predictions.json',
        'python DASE7506-MP2-main/code/score_predictions.py --split validation --predictions experiments/round23/reproduced_predictions.json --output experiments/round23/reproduced_score.json',
        '正式提交指标仍为官方完整 467 题测试的 accuracy（0–100）；不能提交内部或验证成绩作为测试成绩。',
        '源：https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md （2026-10-15 核对）',
        'AI 辅助：Codex 编写实验与分析；无外部训练数据、附加预训练权重或推理时 LLM/API。']
    (OUT/'全量训练优化对照_第23轮报告.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    products=[q for q in OUT.iterdir() if q.is_file()]+assets+[ROOT/'research/round23_selector.py',ROOT/'research/round23_run.py']
    dump(OUT/'completion_manifest.json',dict(completed=True,validation_opened=True,test_opened=False,
        hashes={str(q.relative_to(ROOT)):sha(q) for q in products}))
    print('SELECTED TREE',winner,'RESOURCE',checks,flush=True)


if __name__=='__main__':main()
