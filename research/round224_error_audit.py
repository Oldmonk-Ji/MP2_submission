"""Inspect TRAIN held errors without reading validation or TEST references."""
from collections import Counter

import numpy as np

from round22_common import ROOT, dump, load
from round224_target_conclusion import strict_choice
from common import parse_answer


def run():
    with np.load(ROOT / "experiments/round216/cache.npz") as z:
        held = z["folds"] == 2
        score = z["held_s1"]
        strict = z["strict_nested"][held]
    bank, _, y, ids = load("train")
    indices = np.flatnonzero(held)
    rows = [bank[i] for i in indices]
    labels = y[held]
    choice = strict_choice(rows, score, strict)
    result = []
    for local, row in enumerate(rows):
        if labels[local, choice[local]]:
            continue
        parsed = [parse_answer(c["text"]) for c in row["candidates"]]
        counts = sorted(Counter(p for p in parsed if p is not None).values(), reverse=True)
        if counts != [2, 1, 1]:
            continue
        result.append(dict(train_index=int(ids[indices[local]]),
                           question=row["question"],
                           selected=int(choice[local]),
                           parsed=parsed,
                           correct=[int(j) for j in np.flatnonzero(labels[local])],
                           conclusions=[c["text"][-600:] for c in row["candidates"]]))
    out = dict(split="TRAIN held fold only", cluster="2,1,1",
               error_count=len(result), rows=result)
    path = ROOT / "experiments/round224/held_cluster_errors.json"
    path.parent.mkdir(exist_ok=True)
    dump(path, out)
    print(dict(error_count=len(result), output=str(path)), flush=True)


if __name__ == "__main__":
    run()
