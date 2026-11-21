"""Step 6a (RQ4): activation steering along the probe's irreversible direction.
Direction v = mean(hidden | irreversible) - mean(hidden | reversible) on the TRAIN split, at the best probe layer.
Strengths are calibrated to the residual stream: fraction f means the added vector has norm f * median ||h|| at that
layer (alpha = f * median||h|| / ||v||), so a null result can be told apart from "too weak"; the top fraction should
visibly break the output (watch task success and parse failures). Controls: random directions of the same norm at
+-random_fraction. Every episode is saved.
python steering.py --model Qwen/Qwen2.5-7B-Instruct --data data/sandbox.jsonl data/sandbox_extra.jsonl \
   --hidden data/h_qwen7b_sandbox data/h_qwen7b_sandbox_extra --layer 16 --tag qwen7b
(--layer = best layer from probes.py; index refers to hidden_states, 0 = embeddings; must be >= 1 and < last)"""
import argparse, json
import numpy as np
from utils import read_jsonl, write_jsonl, seed_all
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
    return X[y == 1].mean(0) - X[y == 0].mean(0), float(np.median(np.linalg.norm(X, axis=1)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--hidden", nargs="+", required=True); ap.add_argument("--pos", default="last")
    ap.add_argument("--layer", type=int, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--fractions", type=float, nargs="*", default=[-0.5, -0.25, 0, 0.25, 0.5, 1.0])
    ap.add_argument("--n-random", type=int, default=3); ap.add_argument("--random-fraction", type=float, default=0.5)
    ap.add_argument("--n-per-template", type=int, default=4)
    ap.add_argument("--careful", action="store_true")
    a = ap.parse_args(); seed_all()
    from llm import load_model, chat_generate, Steer
    vec, med = direction(a.data, a.hidden, a.pos, a.layer); np.save(f"results/steer_dir_{a.tag}_L{a.layer}.npy", vec)
    norm = float(np.linalg.norm(vec)); rng = np.random.default_rng(0)
    print(f"layer {a.layer}: ||v|| = {norm:.2f}, median ||h|| = {med:.2f} (||v|| / ||h|| = {norm / med:.3f})", flush=True)
    configs = [("probe_direction", vec, f) for f in a.fractions]
    for i in range(a.n_random):
        r = rng.normal(size=vec.shape).astype(np.float32); r *= norm / np.linalg.norm(r)    # same norm as v
        configs += [(f"random_{i}", r, s * a.random_fraction) for s in (-1, 1)]
    model, tok = load_model(a.model); tasks = build_tasks(a.n_per_template)
    gen = lambda m, n=200: chat_generate(model, tok, m, max_new_tokens=n)
    out = {"layer": a.layer, "direction_norm": norm, "median_hidden_norm": med, "n_pairs": len(tasks), "runs": []}
    for name, v, f in configs:
        alpha = f * med / norm
        with Steer(model, a.layer, v, alpha):
            logs, _, _, summ = evaluate(gen, tasks, a.careful, with_stated=False)
        write_jsonl(f"results/steering_{a.tag}_{name}_{f:+.2f}.jsonl", logs)
        summ.pop("per_template"); out["runs"].append({"direction": name, "fraction": f, "alpha": alpha, **summ})
        print(name, f, {k: round(summ[k], 3) for k in ("caution_rate_irr", "over_caution_rate_rev", "task_success_irr",
                                                       "task_success_rev", "parse_failures_total")}, flush=True)
        json.dump(out, open(f"results/steering_{a.tag}.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    pr = sorted([r for r in out["runs"] if r["direction"] == "probe_direction"], key=lambda r: r["fraction"])
    rd = [r for r in out["runs"] if r["direction"] != "probe_direction"]
    x = [r["fraction"] for r in pr]
    axs[0].plot(x, [r["caution_rate_irr"] for r in pr], "-o", label="probe dir: irreversible version")
    axs[0].plot(x, [r["over_caution_rate_rev"] for r in pr], "-o", alpha=.5, label="probe dir: reversible version")
    axs[0].scatter([r["fraction"] for r in rd], [r["caution_rate_irr"] for r in rd], marker="x", c="gray", label="random dirs: irreversible")
    axs[1].plot(x, [r["task_success_irr"] for r in pr], "-o", label="probe dir")
    axs[1].scatter([r["fraction"] for r in rd], [r["task_success_irr"] for r in rd], marker="x", c="gray", label="random dirs")
    for ax, yl in zip(axs, ("caution rate", "task success (irreversible version)")):
        ax.set_xlabel("steering strength (fraction of median ||h||)"); ax.set_ylabel(yl); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"figures/steering_{a.tag}.png", dpi=200)
