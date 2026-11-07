"""Model helpers (torch/transformers imported lazily so CPU-only scripts still run)."""

def load_model(name, device="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to(device).eval()
    return model, tok


def chat_generate(model, tok, messages, max_new_tokens=200):
    """Greedy decoding (temperature 0)."""
    import torch
    enc = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                  return_dict=True).to(model.device)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


class Steer:
    """Add alpha*vec to the residual stream at hidden_states index `probe_layer`
    (= output of decoder block probe_layer-1). Use as a context manager."""
    def __init__(self, model, probe_layer, vec, alpha):
        import torch
        assert probe_layer >= 1, "cannot steer the embedding layer"
        self.block = model.model.layers[probe_layer - 1]
        self.delta = (alpha * torch.as_tensor(vec)).to(device=model.device, dtype=model.dtype)

    def __enter__(self):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                return (out[0] + self.delta,) + tuple(out[1:])
            return out + self.delta
        self.h = self.block.register_forward_hook(hook)
        return self

    def __exit__(self, *a):
        self.h.remove()
