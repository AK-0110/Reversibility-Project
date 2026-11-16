"""Model helpers (torch/transformers imported lazily so CPU-only scripts still run)."""
import os

def load_model(name, device=None):
    import torch
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    if os.environ.get("LOAD_4BIT") and device == "cuda":
        # 4-bit NF4 weights so 8B models fit a 16 GB T4; report this as a deviation (hidden states differ slightly from bf16)
        from transformers import BitsAndBytesConfig
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
        return AutoModelForCausalLM.from_pretrained(name, quantization_config=q, dtype=torch.float16, device_map={"": 0}).eval(), tok
    # bf16 is only emulated before Ampere (T4 = 7.5): use fp16 there; extract_hidden.py warns on fp16 overflow.
    # BF16=1 forces bf16 (slower on a T4) if fp16 output turns to garbage
    dtype = torch.float32 if device == "cpu" else \
        torch.bfloat16 if os.environ.get("BF16") or torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return model, tok


def chat_generate(model, tok, messages, max_new_tokens=200):
    """Greedy decoding (temperature 0)."""
    import torch
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    with torch.no_grad():
        # Qwen's generation_config sets repetition_penalty (1.05-1.1) and sampling params, which
        # generate() still applies with do_sample=False; reset them so decoding is plain greedy
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.0,
                             temperature=None, top_p=None, top_k=None, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class Steer:
    """Add alpha*vec to the residual stream at hidden_states index `probe_layer`
    (= output of decoder block probe_layer-1). Use as a context manager."""
    def __init__(self, model, probe_layer, vec, alpha):
        import torch
        assert probe_layer >= 1, "cannot steer the embedding layer"
        # hidden_states[-1] is the final-normed output, not the residual stream this hook adds to
        assert probe_layer < len(model.model.layers), "last hidden_states index is post-norm; pick an earlier layer"
        self.block = model.model.layers[probe_layer - 1]
        self.delta = (alpha * torch.as_tensor(vec)).to(device=model.device, dtype=model.dtype)

    def __enter__(self):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + self.delta.to(out[0].device),) + tuple(out[1:])
            return out + self.delta.to(out.device)
        self.h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()
