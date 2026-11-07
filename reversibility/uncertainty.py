"""Step 6b (RQ5): is next-state uncertainty related to reversibility?
Input: a JSONL with one row per transition: {id, prompt, target_state, label}  where label in
reversible / partially_reversible / irreversible (label the transitions with the same <=3-step search as Section 4.2;
see prepare_bytesized.py). We report the mean per-token negative log-likelihood (NLL) of the TRUE next state
under the model, and AUROC of NLL for separating irreversible from reversible transitions.
python uncertainty.py --model Qwen/Qwen2.5-7B-Instruct --data data/bytesized_labelled.jsonl --tag qwen7b"""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import mannwhitneyu
from tqdm import tqdm
from utils import read_jsonl, write_jsonl, seed_all, bootstrap_ci

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--max-len", type=int, default=3000)
    a = ap.parse_args(); seed_all()
    import torch
    from llm import load_model
    model, tok = load_model(a.model); recs = read_jsonl(a.data); rows = []
    for r in tqdm(recs):
        msgs = [{"role": "user", "content": r["prompt"] + "\n\nPredict the next game state exactly."},
                {"role": "assistant", "content": r["target_state"]}]
        text = tok.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
        cs = text.rfind(r["target_state"])
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0]
        if enc["input_ids"].shape[1] > a.max_len: continue            # skip over-long prompts (count them below)
        tgt = torch.tensor([i for i in range(len(offs)) if offs[i][0] >= cs and offs[i][1] > offs[i][0]])
        with torch.no_grad():
            logits = model(**enc.to(model.device)).logits[0].float()
        lp = torch.log_softmax(logits[tgt - 1], -1).gather(1, enc["input_ids"][0, tgt.to(model.device)].unsqueeze(1))
        rows.append({"id": r["id"], "label": r["label"], "nll": float(-lp.mean()), "n_tok": int(len(tgt))})
    write_jsonl(f"results/uncertainty_{a.tag}.jsonl", rows)
    rev = np.array([x["nll"] for x in rows if x["label"] == "reversible"])
    irr = np.array([x["nll"] for x in rows if x["label"] == "irreversible"])
    y = np.r_[np.zeros(len(rev)), np.ones(len(irr))]; s = np.r_[rev, irr]
    auc = roc_auc_score(y, s)
    ci = bootstrap_ci(lambda ix: roc_auc_score(y[ix], s[ix]) if len(set(y[ix])) == 2 else .5, size=len(y))
    out = {"n_scored": len(rows), "n_skipped_too_long": len(recs) - len(rows), "n_reversible": len(rev),
           "n_irreversible": len(irr), "mean_nll_rev": float(rev.mean()), "mean_nll_irr": float(irr.mean()),
           "auroc_nll_irr_vs_rev": float(auc), "auroc_ci95": ci,
           "mannwhitney_p": float(mannwhitneyu(irr, rev).pvalue)}
    json.dump(out, open(f"results/uncertainty_{a.tag}.json", "w"), indent=1); print(json.dumps(out, indent=1))
