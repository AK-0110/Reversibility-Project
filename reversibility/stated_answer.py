"""Step 4 (RQ2): ask the model whether each action can be undone; compare true / stated / probe.
python stated_answer.py --model Qwen/Qwen2.5-7B-Instruct --data data/sandbox.jsonl --tag qwen7b --test-only \
    --probe-preds results/probe_preds_qwen7b_last_binary.csv results/probe_preds_qwen7b_last_three.csv
Binary (irreversible vs not, on rows whose true label is reversible/irreversible) is the primary metric: "partial"
hangs on an mtime the state text never shows. Strict scores count parse failures as wrong."""
import argparse, csv, json, os, re
import numpy as np
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, confusion_matrix
from utils import read_jsonl, write_jsonl, seed_all, bootstrap_ci
from prompts import stated_messages

LAB3 = ["reversible", "partially_reversible", "irreversible"]
LABEL_WORD = re.compile(r"IRREVERSIBLE|NOT REVERSIBLE|PARTIAL|REVERSIBLE")


def parse_stated(text):
    """First label word in the reply ("PARTIAL. It cannot be exactly restored" -> partial); None = parse failure."""
    t = text.upper(); m = LABEL_WORD.search(t)
    if m: return 0 if m.group() == "REVERSIBLE" else 1 if m.group() == "PARTIAL" else 2
    return 2 if "CANNOT BE" in t else None            # parse failure: counted, never silently dropped


def scores(y, p, k):
    """p may hold -1 (parse failure): counted as wrong."""
    return {"accuracy": float((y == p).mean()), "bal_acc": float(balanced_accuracy_score(y, p)),
            "accuracy_ci95": bootstrap_ci((y == p).astype(float)),
            "confusion_true_by_stated": confusion_matrix(y, p, labels=list(range(k))).tolist()}


def vs_probe(rows, path):
    """Probe (seed 0 = stored split) vs stated answer on the same test ids."""
    binary = path.endswith("_binary.csv"); conv = (lambda v: int(v == 2)) if binary else (lambda v: v)
    pr = {r["id"]: int(r["pred"]) for r in csv.DictReader(open(path)) if r.get("seed", "0") == "0"}
    both = [r for r in rows if r["id"] in pr and r["stated"] is not None]
    s = np.array([conv(r["stated"]) for r in both]); q = np.array([pr[r["id"]] for r in both])
    t = np.array([conv(r["true"]) for r in both])
    return {"n": len(both), "kappa_stated_vs_probe": float(cohen_kappa_score(s, q)),
            "probe_bal_acc": float(balanced_accuracy_score(t, q)), "stated_bal_acc": float(balanced_accuracy_score(t, s)),
            "probe_accuracy": float((t == q).mean()), "stated_accuracy": float((t == s).mean()),
            "agreement_stated_probe": float((s == q).mean()), "both_correct": float(((s == t) & (q == t)).mean()),
            "probe_right_stated_wrong": float(((q == t) & (s != t)).mean()),
            "stated_right_probe_wrong": float(((s == t) & (q != t)).mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--probe-preds", nargs="*", default=[])
    ap.add_argument("--test-only", action="store_true")
    ap.add_argument("--rescore", action="store_true", help="no model: re-score saved answers against the labels in --data")
    a = ap.parse_args(); seed_all()
    recs = [r for d in a.data for r in read_jsonl(d) if r["label"] in LAB3]   # drops "unverified" (failed re-search)
    if a.test_only: recs = [r for r in recs if r.get("split") == "test"]
    if a.rescore:                                            # e.g. after the ScienceWorld re-search changed labels
        lab = {r["id"]: LAB3.index(r["label"]) for r in recs}
        # keep every saved answer in the file (true = None when unverified); score only labelled rows
        saved = [{**r, "true": lab.get(r["id"]), "stated": parse_stated(r["raw"])} for r in read_jsonl(f"results/stated_{a.tag}.jsonl")]
        rows = [r for r in saved if r["true"] is not None]
    else:
        from tqdm import tqdm
        from llm import load_model, chat_generate
        model, tok = load_model(a.model); rows = []
        for r in tqdm(recs):
            raw = chat_generate(model, tok, stated_messages(r["state_text"], r["action_text"]), max_new_tokens=12)
            rows.append({"id": r["id"], "prompt": stated_messages(r["state_text"], r["action_text"]), "environment": r["environment"], "true": LAB3.index(r["label"]),
                         "stated": parse_stated(raw), "raw": raw, "is_minimal": r["is_minimal"], "split": r.get("split")})
    write_jsonl(f"results/stated_{a.tag}.jsonl", saved if a.rescore else rows)
    y = np.array([r["true"] for r in rows]); p = np.array([-1 if r["stated"] is None else r["stated"] for r in rows])
    ok = p >= 0; b = y != 1                                  # binary rows: true label reversible / irreversible
    yb, pb = (y[b] == 2).astype(int), np.where(p[b] < 0, -1, (p[b] == 2).astype(int))
    out = {"n": len(rows), "parse_failures": int((~ok).sum()),
           "binary_strict": scores(yb, pb, 2), "three_strict": scores(y, p, 3),
           "three_parsed_only": scores(y[ok], p[ok], 3)}
    mp = np.array([r["is_minimal"] for r in rows])
    if mp.any():
        out["binary_strict_minimal_pairs_bal_acc"] = float(balanced_accuracy_score(yb[mp[b]], pb[mp[b]]))
    for env in sorted({r["environment"] for r in rows}):
        e = np.array([r["environment"] == env for r in rows])[b]
        if e.any(): out[f"binary_strict_bal_acc_{env}"] = float(balanced_accuracy_score(yb[e], pb[e]))
    missing = [pp for pp in a.probe_preds if not os.path.exists(pp)]
    if missing: print("WARNING: no probe predictions, skipping the comparison for", missing)
    out["vs_probe"] = {("binary" if pp.endswith("_binary.csv") else "three"): vs_probe(rows, pp)
                       for pp in a.probe_preds if pp not in missing}
    json.dump(out, open(f"results/stated_{a.tag}.json", "w"), indent=1); print(json.dumps(out, indent=1))
