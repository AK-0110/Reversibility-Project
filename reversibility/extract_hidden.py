"""Step 2: one forward pass per pair; save hidden states of the last action token and the
mean over action tokens, at every layer (index 0 = embeddings).
Usage: python extract_hidden.py --model Qwen/Qwen2.5-7B-Instruct --data data/sandbox.jsonl --out data/h_qwen7b_sandbox"""
import argparse, numpy as np
from tqdm import tqdm
from utils import read_jsonl, seed_all
from prompts import probe_messages
from llm import load_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--careful", action="store_true")
    a = ap.parse_args(); seed_all()
    import torch
    recs = read_jsonl(a.data); model, tok = load_model(a.model)
    last, mean, prompts = [], [], []
    for r in tqdm(recs):
        text = tok.apply_chat_template(probe_messages(r["state_text"], r["action_text"], a.careful, r["environment"]),
                                       tokenize=False, continue_final_message=True)
        prompts.append({"id": r["id"], "prompt_text": text})
        cs = text.rfind(r["action_text"])
        assert cs >= 0, "action text not found in prompt"
        enc = tok(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0]
        idx = [i for i in range(len(offs)) if offs[i][0] >= cs and offs[i][1] > offs[i][0]] or [len(offs) - 1]
        with torch.no_grad():
            hs = model(**enc.to(model.device), output_hidden_states=True).hidden_states
        last.append(torch.stack([h[0, idx[-1]] for h in hs]).float().cpu().numpy().astype(np.float16))
        mean.append(torch.stack([h[0, idx].mean(0) for h in hs]).float().cpu().numpy().astype(np.float16))
    np.save(a.out + "_last.npy", np.stack(last)); np.save(a.out + "_mean.npy", np.stack(mean))
    from utils import write_jsonl; write_jsonl(a.out + "_prompts.jsonl", prompts)
    print("saved", np.stack(last).shape)
