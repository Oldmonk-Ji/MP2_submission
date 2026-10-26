"""Package retained inference path and run a fresh TRAIN-only local smoke test."""
from data_access import ROOT, load, parse_answer
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import numpy as np
from round14_selector import predict
from round4_selector import predict as baseline_predict
from round14_run import verify_baseline

OUT=ROOT/'experiments/round14'
BUNDLE=OUT/'deployed_bundle'


def save(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')


def child():
    # This is a 467-row TRAIN batch. It is deliberately not called an official
    # score, a test-bank run, or a resource-eligibility measurement.
    bank,arrays,y=load('train');bank=bank[:467];arrays={k:v[:467] for k,v in arrays.items()}
    pred=predict(bank,arrays,BUNDLE)
    assert len(pred)==len(bank) and all(type(x) is int and 0<=x<4 for x in pred)
    save('smoke_train_predictions.json',dict(split='train',indices=list(range(len(bank))),selected=pred))
    print(json.dumps(dict(n=len(bank),in_sample_correct=int(y[np.arange(len(bank)),pred].sum()))))


def main():
    decision=json.loads((OUT/'deployment.json').read_text())
    assert decision['mode']=='diagnostic_only' and not decision['override_enabled']
    original=verify_baseline()
    BUNDLE.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'experiments/round4/selected_bundle',BUNDLE/'incumbent',dirs_exist_ok=True)
    config=dict(mode='diagnostic_only',override_enabled=False,incumbent='incumbent')
    (BUNDLE/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    files=[p for p in BUNDLE.rglob('*') if p.is_file()]
    manifest={str(p.relative_to(BUNDLE)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    save('deployed_asset_manifest.json',manifest)
    started=time.perf_counter()
    proc=subprocess.run([sys.executable,str(ROOT/'research/round14_deploy.py'),'--child'],cwd=ROOT,env=os.environ.copy(),text=True,capture_output=True,check=True)
    seconds=time.perf_counter()-started
    bank,arrays,_=load('train');small=bank[:12];xa={k:v[:12] for k,v in arrays.items()}
    original_choices=baseline_predict(small,xa,ROOT/'experiments/round4/selected_bundle')
    wrapped=predict(small,xa,BUNDLE);assert wrapped==original_choices
    saved=json.loads((OUT/'smoke_train_predictions.json').read_text())
    assert saved['selected'][:12]==wrapped
    changed_ids=[dict(r,id='unseen-local-'+str(i)) for i,r in enumerate(small)]
    assert predict(changed_ids,xa,BUNDLE)==wrapped
    assert predict([],{k:v[:0] for k,v in xa.items()},BUNDLE)==[]
    permutation=[2,0,3,1]
    permuted=[dict(r,candidates=[dict(r['candidates'][j],index=n) for n,j in enumerate(permutation)]) for r in small]
    pc=predict(permuted,{k:v[:,permutation] for k,v in xa.items()},BUNDLE)
    assert all(parse_answer(r['candidates'][a]['text'])==parse_answer(s['candidates'][b]['text']) for r,s,a,b in zip(small,permuted,wrapped,pc))
    total=sum(x['bytes'] for x in manifest.values());assert total<16*1024**2
    result=dict(mode='retained baseline; diagnostic checker does not override',rows=467,data_split='train',
                cold_process_seconds=seconds,assets_bytes=total,assets_mib=total/1024**2,
                four_thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')},
                hardware='Local Intel i7-11800H; not reference Xeon 8457C',
                timing_limit='467 TRAIN rows; process also loads the full train input archive. Not the official test timing or resource qualification.',
                quality_limit='In-sample smoke only; do not treat correct counts as generalization.',
                checks=['prediction protocol','baseline parity','variable batch parity','unseen IDs','empty input','candidate permutation numeric invariance','original assets unchanged'],
                retained_baseline=original,validation_access=False,test_access=False)
    save('local_smoke.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--child',action='store_true');a=p.parse_args();child() if a.child else main()
