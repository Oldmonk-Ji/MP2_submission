"""Summarize the bounded round235 continuation using TRAIN/VAL records only."""
from round22_common import ROOT,read,dump

def main():
    rows=[]
    for stage in range(236,241):
        result=read(ROOT/f'experiments/round{stage}/result.json')
        records=result['records']
        val=[r for r in records if 'validation' in r]
        best=max(val,key=lambda r:(r['validation']['correct'],r['held']['correct'],-r['alpha'])) if val else None
        rows.append(dict(round=stage,configurations=len(records),validation_evaluated=len(val),
            best_held=max(r['held']['correct'] for r in records),best_validation=best,
            selected=result['selected'],test_opened=result['test_opened']))
    assert not any(r['test_opened'] for r in rows)
    summary=dict(baseline=dict(round=235,validation=763,held=1479),rounds=rows,
        configurations=sum(r['configurations'] for r in rows),
        validation_evaluations=sum(r['validation_evaluated'] for r in rows),
        improved=any(r['selected'] is not None for r in rows),test_opened=False,
        validation_role='Repeated development evidence, not an untouched statistical test.')
    dump(ROOT/'experiments/round240/continuation_summary.json',summary)
    print(summary,flush=True)

if __name__=='__main__':main()
