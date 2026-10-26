"""Standard frozen feature evaluation: VAL reproduction, interfaces, then TEST.

TEST is optional, final-score-only, never imported into training/data_access.
Once initialized the output cannot be silently reused. No test-error analysis.
"""
from pathlib import Path
import argparse,ast,datetime,hashlib,json,os,shutil,subprocess,sys,time,copy

ROOT=Path(__file__).resolve().parents[1];CODE=ROOT/'DASE7506-MP2-main/code'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--selection',required=True);parser.add_argument('--method',required=True);parser.add_argument('--output',required=True);parser.add_argument('--with-test',action='store_true');args=parser.parse_args()
    selection_path=ROOT/args.selection;selection=read(selection_path);chosen=selection['best_new'];method_source=ROOT/args.method;out=ROOT/args.output
    assert not out.exists(),'Frozen evaluation already initialized; do not repeat TEST silently.'
    assert selection['test_target_fixed_before_test'] and not selection['test_used_for_selection']
    # Prior freezes and new frozen feature preparation must remain unchanged.
    checked={}
    for manifest in [ROOT/'experiments/round46_50_completion_manifest.json']+[ROOT/f'experiments/round{i}/prepared_freeze.json' for i in (51,52,53,54)]:
        for p,h in read(manifest)['hashes'].items():
            if p not in checked:checked[p]=sha(ROOT/p)
            assert checked[p]==h,p
    out.mkdir();bundle=out/'bundle';methoddir=out/'method';methoddir.mkdir()
    shutil.copytree(ROOT/chosen['bundle'],bundle)
    queue=[method_source,ROOT/'starter/method.py',CODE/'predict.py',CODE/'score_predictions.py'];sources=set()
    while queue:
        path=queue.pop()
        if path in sources:continue
        sources.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                for directory in (ROOT/'research',ROOT/'starter',CODE):
                    p=directory/f'{name}.py'
                    if p.exists():queue.append(p);break
    # Conservative closure includes imports inside unused training helpers.
    # Keep their exact source, but reject research-data imports on the actual
    # prediction import path in the wrapper below.
    for p in sources:
        if p.parent!=CODE:shutil.copy2(p,methoddir/p.name)
    # Guard prediction against all reference files, allowing official candidate inputs.
    wrapper=f'''import os,sys\ndef guard(event,args):\n    if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):\n        p=os.fsdecode(args[0]).replace(chr(92),"/").lower()\n        if any(p.endswith("/data/gsm8k_"+s+".jsonl") for s in ("train","validation","test")):\n            raise PermissionError("Prediction must not open reference answers")\nsys.addaudithook(guard)\n_before=set(sys.modules)\nfrom {method_source.stem} import predict as model_predict\nif "data_access" in set(sys.modules)-_before:\n    raise ImportError("Inference imported research data access")\ndef predict(bank,arrays,checkpoint):\n    return model_predict(bank,arrays,checkpoint)\n'''
    (methoddir/'entrypoint.py').write_text(wrapper,encoding='utf-8');method=methoddir/'entrypoint.py'
    assets=[p for p in bundle.rglob('*') if p.is_file()];size=sum(p.stat().st_size for p in assets);assert size<=16*1024**2
    frozen=assets+list(methoddir.glob('*.py'))+[p for p in sources if p.parent==CODE]+[selection_path,Path(__file__).resolve()]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in frozen}
    commands={split:[[str(CODE/'predict.py'),'--method',str(method),'--checkpoint',str(bundle),'--split',split,'--output',str(out/f'{split}_predictions.json')],
        [str(CODE/'score_predictions.py'),'--split',split,'--predictions',str(out/f'{split}_predictions.json'),'--output',str(out/f'{split}_score.json')]] for split in ('validation','test')}
    dump(out/'freeze_before_evaluation.json',dict(created_utc=now(),selected_new_method=chosen,incumbent_selection=selection['selected'],
        selection_basis='Complete official VAL, before opening TEST; best NEW feature method is comparison target even if incumbent remains better.',
        user_authorization='用测试集验证下新方法的效果（纳入标准feature验证流程）',test_enabled=args.with_test,
        test_used_for_tuning_or_selection=False,prior_files_verified=len(checked),hashes=hashes,asset_bytes=size,commands=commands))
    env=dict(os.environ);env['PYTHONPATH']=str(methoddir)+os.pathsep+str(ROOT/'.venv/Lib/site-packages');env['PYTHONIOENCODING']='utf-8'
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]='4'
    def run(arguments,label):
        tick=time.perf_counter();p=subprocess.run([sys.executable,'-X','utf8']+arguments,env=env,cwd=ROOT,capture_output=True,text=True,encoding='utf-8');elapsed=time.perf_counter()-tick
        (out/f'{label}_stdout.txt').write_text(p.stdout,encoding='utf-8');(out/f'{label}_stderr.txt').write_text(p.stderr,encoding='utf-8')
        assert p.returncode==0,(label,p.returncode,p.stdout,p.stderr)
        print(p.stdout.strip(),flush=True);return elapsed
    validation_times=[run(cmd,f'validation{j}') for j,cmd in enumerate(commands['validation'])]
    vr=read(out/'validation_score.json');assert vr['correct']==chosen['correct'] and vr['n']==798
    original=ROOT/chosen['bundle'];cache=original.parent/f"validation_{'sft' if chosen['arm']=='lexical_sft' else chosen['arm']}_predictions.json"
    assert read(cache)['selected']==read(out/'validation_predictions.json')['selected']
    # All interface checks are performed by a separate TRAIN/VAL-only child.
    run([str(ROOT/'research/round52_interface.py'),'--method-dir',str(methoddir),'--bundle',str(bundle),'--predictions',str(out/'validation_predictions.json'),'--output',str(out/'interface_checks.json')],'interface')
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    summary=dict(method=chosen['arm'],validation_correct=vr['correct'],validation_n=vr['n'],validation_accuracy=vr['accuracy'],
        asset_bytes=size,asset_MiB=size/1024**2,validation_fresh_seconds=sum(validation_times),test_evaluated=False,
        test_used_for_tuning_or_selection=False,selected_for_submission=selection['selected'],reference_machine_certified=False,
        local_cpu='Intel Core i7-11800H',threads=4,leaderboard_submitted=False)
    if args.with_test:
        dump(out/'test_started.json',dict(created_utc=now(),frozen_hashes_verified=True))
        times=[run(cmd,f'test{j}') for j,cmd in enumerate(commands['test'])]
        tr=read(out/'test_score.json');assert tr['n']==467 and tr['split']=='test'
        baseline=read(ROOT/'experiments/final_evaluation_20261021/test_score.json')
        summary.update(test_evaluated=True,test_correct=tr['correct'],test_n=tr['n'],test_accuracy=tr['accuracy'],
            historical_baseline_correct=baseline['correct'],historical_baseline_accuracy=baseline['accuracy'],
            correct_difference=tr['correct']-baseline['correct'],test_prediction_seconds=times[0],test_scoring_seconds=times[1],
            test_total_fresh_seconds=sum(times),local_under30=sum(times)<=30,goal_97_achieved=tr['accuracy']>=97,successful_test_evaluations=1)
    for p,h in hashes.items():assert sha(ROOT/p)==h,p
    summary.update(frozen_hashes_unchanged=True,completed_utc=now());dump(out/'summary.json',summary)
    files=[p for p in out.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    dump(out/'completion_manifest.json',dict(completed=True,hashes={str(p.relative_to(ROOT)):sha(p) for p in files}))
    print('FROZEN_FEATURE_RESULT',summary,flush=True)

if __name__=='__main__':main()
