"""CPU baseline (no model): bag-of-words with verb x state-token interaction features, same splits as probes.py.
The label is a function of (verb, one setting line), which a plain BoW cannot express but this one can.
python baseline_interaction.py data/sandbox.jsonl data/sandbox_extra.jsonl   ->  results/baseline_interaction.json"""
import json, sys
import numpy as np
from probes import make_task, split, lexical_baseline
from utils import read_jsonl

recs = [r for p in sys.argv[1:] for r in read_jsonl(p)]
out = {}
for task in ("binary", "three"):
    idx, y = make_task(recs, task, "label"); sub = [recs[i] for i in idx]
    texts = [" ".join(f"{r['verb']}__{w}" for w in r["state_text"].replace("\n", " ").split()) + " " + r["action_text"] for r in sub]
    mp = np.array([bool(r["is_minimal"]) for r in sub]); runs = []
    for seed in (0, 1, 2):
        tr, va, te = split(sub, np.arange(len(sub)), seed)
        runs.append(lexical_baseline(texts, y, np.concatenate([tr, va]), te, mp))
    out[task] = {k: [float(r[k]) for r in runs] for k in runs[0]}
    print(task, {k: f"{np.mean(v):.3f} +- {np.std(v):.3f}" for k, v in out[task].items()})
json.dump(out, open("results/baseline_interaction.json", "w"), indent=1)
