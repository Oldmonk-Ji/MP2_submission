"""467-question validation timing proxy; never opens TEST files."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "DASE7506-MP2-main/code"))
from common import load_inputs, load_bank, read_jsonl, parse_answer

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("predict", "score"))
parser.add_argument("--count", type=int, default=467)
parser.add_argument("--round", type=int, choices=(214, 232), default=214)
args = parser.parse_args()
dest = ROOT / f"remote-work/round{args.round}/validation_proxy_predictions.json"
if args.mode == "predict":
    bank, arrays = load_inputs("validation")
    bank = bank[:args.count]
    arrays = {k: v[:args.count] for k, v in arrays.items()}
    sys.path.insert(0, str(ROOT / f"experiments/round{args.round}/frozen/method"))
    from entrypoint import predict
    choices = predict(bank, arrays, ROOT / f"experiments/round{args.round}/bundle")
    dest.parent.mkdir(exist_ok=True, parents=True)
    dest.write_text(json.dumps(dict(count=len(bank), selected=choices)))
else:
    saved = json.loads(dest.read_text())
    bank = load_bank("validation")[:saved["count"]]
    refs = read_jsonl(ROOT / "DASE7506-MP2-main/code/data/gsm8k_validation.jsonl")[:saved["count"]]
    correct = sum(parse_answer(row["candidates"][j]["text"]) == parse_answer(ref["answer"])
                  for row, j, ref in zip(bank, saved["selected"], refs))
    print(json.dumps(dict(split="validation_prefix_timing_proxy", correct=correct,
                          count=len(bank), official_test=False)))
