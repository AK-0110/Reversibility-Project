"""Step 6a (RQ4): activation steering along the probe's irreversible direction.
Direction = mean(hidden | irreversible) - mean(hidden | reversible) on the TRAIN split, at the best probe layer.
For each strength we rerun the agent on all paired tasks; a random direction of the same norm is the control.
python steering.py --model Qwen/Qwen2.5-7B-Instruct --data data/sandbox.jsonl data/sciworld.jsonl \
   --hidden data/h_qwen7b_sandbox data/h_qwen7b_sciworld --layer 16 --tag qwen7b
(--layer = best layer from probes.py; index refers to hidden_states, 0 = embeddings; must be >= 1)"""
import argparse, json
import numpy as np
from utils import read_jsonl, seed_all
from tasks import build_tasks
from agent_eval import evaluate

LAB = {"reversible": 0, "irreversible": 1}


def direction(data, hidden, pos, layer):
    recs, H = [], []
    for dp, hp in zip(data, hidden):
        recs += read_jsonl(dp); H.append(np.load(f"{hp}_{pos}.npy", mmap_mode="r")[:, layer, :])
    H = np.concatenate(H).astype(np.float32)
    tr = [i for i, r in enumerate(recs) if r["split"] == "train" and r["label"] in LAB]
    y = np.array([LAB[recs[i]["label"]] for i in tr]); X = H[tr]
    return X[y == 1].mean(0) - X[y == 0].mean(0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--hidden", nargs="+", required=True); ap.add_argument("--pos", default="last")
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--strengths", type=float, nargs="+", default=[-2, -1, 0, 1, 2, 3])
    ap.add_argument("--careful", action="store_true")
    a = ap.parse_args(); seed_all()
    from llm import load_model, chat_generate, Steer
    vec = direction(a.data, a.hidden, a.pos, a.layer); np.save(f"results/steer_dir_{a.tag}_L{a.layer}.npy", vec)
    rng = np.random.default_rng(0); rnd = rng.normal(size=vec.shape).astype(np.float32)
    rnd *= np.linalg.norm(vec) / np.linalg.norm(rnd)                  # same norm as the probe direction
    model, tok = load_model(a.model); tasks = build_tasks()
    out = {"layer": a.layer, "direction_norm": float(np.linalg.norm(vec)), "runs": []}
    for name, v in (("probe_direction", vec), ("random_direction", rnd)):
        for s in a.strengths:
            if s == 0 and name == "random_direction": continue        # identical to the unsteered run
            gen = lambda m: chat_generate(model, tok, m, max_new_tokens=200)
            with Steer(model, a.layer, v, s):
                logs, stated, summ = evaluate(gen, tasks, a.careful, with_stated=False)
            summ.pop("per_template"); out["runs"].append({"direction": name, "strength": s, **summ})
            print(name, s, {k: round(summ[k], 3) for k in ("caution_rate_irr", "over_caution_rate_rev",
                                                           "task_success_irr", "task_success_rev")}, flush=True)
            json.dump(out, open(f"results/steering_{a.tag}.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    for name, ls in (("probe_direction", "-o"), ("random_direction", "--s")):
        rs = sorted([r for r in out["runs"] + [dict(r, direction="random_direction") for r in out["runs"]
                      if r["direction"] == "probe_direction" and r["strength"] == 0] if r["direction"] == name],
                    key=lambda r: r["strength"])
        x = [r["strength"] for r in rs]
        axs[0].plot(x, [r["caution_rate_irr"] for r in rs], ls, label=f"{name}: irreversible version")
        axs[0].plot(x, [r["over_caution_rate_rev"] for r in rs], ls, alpha=.5, label=f"{name}: reversible version")
        axs[1].plot(x, [r["task_success_irr"] for r in rs], ls, label=name)
    axs[0].set_xlabel("steering strength"); axs[0].set_ylabel("caution rate"); axs[0].legend(fontsize=6)
    axs[1].set_xlabel("steering strength"); axs[1].set_ylabel("task success (irreversible version)"); axs[1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"figures/steering_{a.tag}.png", dpi=200)
