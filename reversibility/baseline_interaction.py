"""CPU baselines (no model): bag-of-words with verb x state interaction features, same splits as probes.py.
The label is a function of (verb, one setting line), which a plain BoW cannot express. verb_x_word crosses the verb
with each state word (cannot tell which line a "disabled" came from); verb_x_line crosses it with each whole state line.
python baseline_interaction.py data/sandbox.jsonl data/sandbox_extra.jsonl   ->  results/baseline_interaction.json"""
import json, re, sys
import numpy as np
from probes import make_task, split, lexical_baseline
from utils import read_jsonl

recs = [r for p in sys.argv[1:] for r in read_jsonl(p)]
feats = {"verb_x_word": lambda r: [f"{r['verb']}__{w}" for w in r["state_text"].split()],
         "verb_x_line": lambda r: [f"{r['verb']}__" + re.sub(r"[^A-Za-z]+", "_", l.strip()) for l in r["state_text"].splitlines()]}
out = {}
for task in ("binary", "three"):
    idx, y = make_task(recs, task, "label"); sub = [recs[i] for i in idx]
    mp = np.array([bool(r["is_minimal"]) for r in sub]); out[task] = {}
    for name, f in feats.items():
        texts = [" ".join(f(r)) + " " + r["action_text"] for r in sub]; runs = []
        for seed in (0, 1, 2):
            tr, va, te = split(sub, np.arange(len(sub)), seed)
            runs.append(lexical_baseline(texts, y, np.concatenate([tr, va]), te, mp))
        out[task][name] = {k: [float(r[k]) for r in runs] for k in runs[0]}
        print(task, name, {k: f"{np.mean(v):.3f} +- {np.std(v):.3f}" for k, v in out[task][name].items()})
json.dump(out, open("results/baseline_interaction.json", "w"), indent=1)
