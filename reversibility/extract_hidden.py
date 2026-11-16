"""Step 2: one forward pass per pair; save hidden states of the last action token, the mean over
action tokens, and the last token BEFORE the action ("pre": sees the state but not the action),
at every layer (index 0 = embeddings).
Usage: python extract_hidden.py --model Qwen/Qwen2.5-7B-Instruct --data data/sandbox.jsonl data/sciworld.jsonl \
           --out data/h_qwen7b_sandbox data/h_qwen7b_sciworld      (one --out prefix per --data file, one model load)"""
import argparse, numpy as np
from tqdm import tqdm
from utils import read_jsonl, seed_all
from prompts import probe_messages
from llm import load_model


def extract(model, tok, recs, out, careful):
    import torch
    last, mean, pre, prompts = [], [], [], []
    for r in tqdm(recs, desc=out):
        text = tok.apply_chat_template(probe_messages(r["state_text"], r["action_text"], careful, r["environment"]),
                                       tokenize=False, continue_final_message=True)
        prompts.append({"id": r["id"], "prompt_text": text})
        cs = text.rfind(r["action_text"])
        assert cs >= 0, "action text not found in prompt"
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0]
        idx = [i for i in range(len(offs)) if offs[i][0] >= cs and offs[i][1] > offs[i][0]] or [len(offs) - 1]
        p0 = max(i for i in range(len(offs)) if offs[i][1] <= cs)       # last token before the action text
        with torch.no_grad():
            hs = model(**enc.to(model.device), output_hidden_states=True).hidden_states
        # .cpu() per layer: with device_map="auto" layers can sit on different GPUs
        for store, f in ((last, lambda h: h[0, idx[-1]]), (mean, lambda h: h[0, idx].mean(0)), (pre, lambda h: h[0, p0])):
            store.append(torch.stack([f(h).float().cpu() for h in hs]).numpy().astype(np.float16))
    for name, arr in (("last", last), ("mean", mean), ("pre", pre)):
        arr = np.stack(arr); bad = int((~np.isfinite(arr)).any(axis=(1, 2)).sum())
        if bad: print(f"WARNING: {bad} rows of {name} overflow float16 (inf/nan); probes on them will fail")
        np.save(f"{out}_{name}.npy", arr)
    from utils import write_jsonl; write_jsonl(out + "_prompts.jsonl", prompts)
    print("saved", np.stack(last).shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", nargs="+", required=True); ap.add_argument("--careful", action="store_true")
    a = ap.parse_args(); seed_all()
    assert len(a.data) == len(a.out), "one --out prefix per --data file"
    model, tok = load_model(a.model)
    for data, out in zip(a.data, a.out):
        extract(model, tok, read_jsonl(data), out, a.careful)
