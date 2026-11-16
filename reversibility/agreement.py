"""Section 4.2 quality control: Cohen's kappa between two human annotators and vs the automatic label.
python agreement.py data/manual_annotator1.csv data/manual_annotator2.csv data/sandbox.jsonl data/sciworld.jsonl
(the automatic-label files are optional, e.g. for ToolEmu there is no automatic label)"""
import csv, sys, json
from sklearn.metrics import cohen_kappa_score
from utils import read_jsonl

LABELS = {"reversible", "partially_reversible", "irreversible"}
a1 = {r["id"]: r["your_label"].strip() for r in csv.DictReader(open(sys.argv[1]))}
a2 = {r["id"]: r["your_label"].strip() for r in csv.DictReader(open(sys.argv[2]))}
bad = sorted(i for a in (a1, a2) for i, v in a.items() if v and v not in LABELS)
assert not bad, f"labels must be one of {LABELS}; fix rows {bad}"
auto = {r["id"]: r["label"] for p in sys.argv[3:] for r in read_jsonl(p)}
ids = [i for i in a1 if a1[i] and a2.get(i) and (i in auto or not auto)]
agree = [i for i in ids if a1[i] == a2[i]]
k = lambda a, b, xs: cohen_kappa_score([a[i] for i in xs], [b[i] for i in xs]) if xs else None
binary = lambda a: {i: a[i] == "irreversible" for i in a}          # primary: irreversible vs not ("partial" hinges on mtime)
res = {"n": len(ids), "kappa_annotator1_vs_2": k(a1, a2, ids), "kappa_annotator1_vs_2_binary": k(binary(a1), binary(a2), ids),
       "n_agreed": len(agree), "disagreements": [i for i in ids if a1[i] != a2[i]]}
if auto:
    res.update({"kappa_annotator1_vs_auto": k(a1, auto, ids), "kappa_annotator2_vs_auto": k(a2, auto, ids),
                "kappa_annotator1_vs_auto_binary": k(binary(a1), binary(auto), ids),
                "kappa_annotator2_vs_auto_binary": k(binary(a2), binary(auto), ids),
                # only on rows where both humans agree: higher by construction, report it as such
                "kappa_consensus_vs_auto_agreed_rows_only": k(a1, auto, agree)})
json.dump(res, open("results/label_agreement.json", "w"), indent=1); print(json.dumps(res, indent=1))
