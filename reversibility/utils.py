import json, os, random
import numpy as np

SEED = 0

def seed_all(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

def read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def bootstrap_ci(values_or_fn, n=1000, seed=0, alpha=0.05, size=None):
    """values_or_fn: 1-D array (CI of the mean) or fn(idx)->float with `size` items to resample."""
    rng = np.random.default_rng(seed)
    if callable(values_or_fn):
        stats = [values_or_fn(rng.integers(0, size, size)) for _ in range(n)]
    else:
        v = np.asarray(values_or_fn, dtype=float)
        stats = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))
