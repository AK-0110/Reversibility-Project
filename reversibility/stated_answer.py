"""Step 4 (RQ2): ask the model whether each action can be undone; compare true / stated / probe.
python stated_answer.py --model Qwen/Qwen2.5-7B-Instruct --data data/sandbox.jsonl --tag qwen7b \
    --probe-preds results/probe_preds_qwen7b_last_three.csv        (probe preds optional; test split only)"""
import argparse, csv, json, re
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from tqdm import tqdm
from utils import read_jsonl, write_jsonl, seed_all, bootstrap_ci
from prompts import stated_messages

LAB3 = ["reversible", "partially_reversible", "irreversible"]


def parse_stated(text):
    t = text.upper()
    if "IRREVERSIBLE" in t or "NOT REVERSIBLE" in t or "CANNOT BE" in t: return 2
    if "PARTIAL" in t: return 1
    if "REVERSIBLE" in t: return 0
    return None                                   # parse failure: counted, never silently dropped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--probe-preds", default=None)
    ap.add_argument("--test-only", action="store_true")
    a = ap.parse_args(); seed_all()
    from llm import load_model, chat_generate
    recs = [r for d in a.data for r in read_jsonl(d)]
    if a.test_only: recs = [r for r in recs if r.get("split") == "test"]
    model, tok = load_model(a.model); rows = []
    for r in tqdm(recs):
        raw = chat_generate(model, tok, stated_messages(r["state_text"], r["action_text"]), max_new_tokens=12)
        rows.append({"id": r["id"], "prompt": stated_messages(r["state_text"], r["action_text"]), "environment": r["environment"], "true": LAB3.index(r["label"]),
                     "stated": parse_stated(raw), "raw": raw, "is_minimal": r["is_minimal"], "split": r.get("split")})
    write_jsonl(f"results/stated_{a.tag}.jsonl", rows)
    ok = [r for r in rows if r["stated"] is not None]
    y = np.array([r["true"] for r in ok]); p = np.array([r["stated"] for r in ok])
    out = {"n": len(rows), "parse_failures": len(rows) - len(ok),
           "stated_accuracy": float((y == p).mean()), "stated_bal_acc": float(balanced_accuracy_score(y, p)),
           "accuracy_ci95": bootstrap_ci((y == p).astype(float)),
           "confusion_true_by_stated": confusion_matrix(y, p, labels=[0, 1, 2]).tolist()}
    mp = np.array([r["is_minimal"] for r in ok])
    if mp.any(): out["stated_accuracy_minimal_pairs"] = float((y[mp] == p[mp]).mean())
    if a.probe_preds:
        pr = {r["id"]: int(r["pred"]) for r in csv.DictReader(open(a.probe_preds))}
        both = [r for r in ok if r["id"] in pr]
        s = np.array([r["stated"] for r in both]); q = np.array([pr[r["id"]] for r in both])
        t = np.array([r["true"] for r in both])
        out["n_with_probe"] = len(both)
        out["kappa_stated_vs_probe"] = float(cohen_kappa_score(s, q))
        out["probe_accuracy_same_items"] = float((t == q).mean())
        out["stated_accuracy_same_items"] = float((t == s).mean())
        out["agreement_stated_probe"] = float((s == q).mean())
        both_correct = ((s == t) & (q == t)).mean()
        out["both_correct"] = float(both_correct)
        out["probe_right_stated_wrong"] = float(((q == t) & (s != t)).mean())
        out["stated_right_probe_wrong"] = float(((s == t) & (q != t)).mean())
    json.dump(out, open(f"results/stated_{a.tag}.json", "w"), indent=1); print(json.dumps(out, indent=1))
