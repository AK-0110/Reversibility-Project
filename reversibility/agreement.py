"""Section 4.2 quality control: Cohen's kappa between two human annotators and vs the automatic label.
python agreement.py data/manual_annotator1.csv data/manual_annotator2.csv data/sandbox.jsonl data/sciworld.jsonl
(the automatic-label files are optional, e.g. for ToolEmu there is no automatic label)"""
import csv, sys, json
from sklearn.metrics import cohen_kappa_score
from utils import read_jsonl

a1 = {r["id"]: r["your_label"].strip() for r in csv.DictReader(open(sys.argv[1]))}
a2 = {r["id"]: r["your_label"].strip() for r in csv.DictReader(open(sys.argv[2]))}
auto = {r["id"]: r["label"] for p in sys.argv[3:] for r in read_jsonl(p)}
ids = [i for i in a1 if a1[i] and a2.get(i) and (i in auto or not auto)]
k12 = cohen_kappa_score([a1[i] for i in ids], [a2[i] for i in ids])
# consensus label: use annotator 1 where they agree, otherwise report disagreements for adjudication
agree = [i for i in ids if a1[i] == a2[i]]
k_auto = cohen_kappa_score([a1[i] for i in agree], [auto[i] for i in agree]) if auto else None
res = {"n": len(ids), "kappa_annotator1_vs_2": k12, "n_agreed": len(agree),
       "kappa_human_consensus_vs_auto": k_auto, "disagreements": [i for i in ids if a1[i] != a2[i]]}
json.dump(res, open("results/label_agreement.json", "w"), indent=1); print(res)
