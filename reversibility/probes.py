"""Step 3 (RQ1): linear probes per layer with every baseline/control from Section 7.2,
3 split-seeds, bootstrap CIs, minimal-pair evaluation, cross-environment transfer.

Examples
  python probes.py --data data/sandbox.jsonl data/sciworld.jsonl \
      --hidden data/h_qwen7b_sandbox data/h_qwen7b_sciworld --pos last --tag qwen7b_last
  python probes.py ... --pos mean --tag qwen7b_mean            # ablation: token position
  python probes.py ... --minimal-only --tag qwen7b_minimal     # ablation: minimal pairs only
  python probes.py ... --label-key label_h1 --tag qwen7b_h1    # ablation: horizon 1/2/3
  python probes.py ... --transfer --tag qwen7b_transfer        # ablation: cross-environment transfer
Binary vs three-class is always reported (the two tasks)."""
import argparse, json, os, warnings
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from utils import read_jsonl, seed_all, bootstrap_ci

warnings.filterwarnings("ignore")
LAB3 = ["reversible", "partially_reversible", "irreversible"]


def load(data_paths, hidden_prefixes, pos):
    recs, H = [], []
    for dp, hp in zip(data_paths, hidden_prefixes):
        r = read_jsonl(dp); h = np.load(f"{hp}_{pos}.npy", mmap_mode="r")
        assert len(r) == h.shape[0], f"{dp}: {len(r)} records vs {h.shape[0]} hidden rows"
        recs += r; H.append(h)
    return recs, np.concatenate(H, axis=0)


def make_task(recs, task, label_key):
    if task == "binary":
        lab = lambda r: r.get(label_key, r["label"])      # environments without per-horizon labels fall back to `label`
        idx = [i for i, r in enumerate(recs) if lab(r) in (LAB3[0], LAB3[2])]
        y = np.array([int(lab(recs[i]) == LAB3[2]) for i in idx])
    else:
        lab = lambda r: r.get(label_key, r["label"])
        idx = [i for i, r in enumerate(recs) if lab(r) in LAB3]
        y = np.array([LAB3.index(lab(recs[i])) for i in idx])
    return np.array(idx), y


def split(recs, idx, seed, test_size=0.2):
    groups = np.array([recs[i]["scenario_id"] for i in idx])
    if seed == 0 and all("split" in recs[i] for i in idx):           # stored split = primary split
        te = np.array([recs[i]["split"] == "test" for i in idx]); tr = ~te
        tr, te = np.where(tr)[0], np.where(te)[0]
    else:
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, groups=groups))
    sub, val = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed + 100).split(tr, groups=groups[tr]))
    return tr[sub], tr[val], te        # train_sub, validation (layer selection), test


def clf(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=300))


def metrics(y, pred, proba, k):
    m = {"bal_acc": balanced_accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro")}
    try:
        m["auroc"] = roc_auc_score(y, proba[:, 1]) if k == 2 else \
            roc_auc_score(y, proba, multi_class="ovr", labels=list(range(k)))
    except ValueError:
        m["auroc"] = float("nan")
    return m


def bal_ci(y, p, n=1000):
    return bootstrap_ci(lambda ix: balanced_accuracy_score(y[ix], p[ix]), n=n, size=len(y))


def control_labels(verbs, k, seed):
    """Hewitt & Liang control task: every verb gets a fixed random label (all k labels used)."""
    rng = np.random.default_rng(seed + 7); vs = sorted(set(verbs))
    while True:
        table = {v: int(rng.integers(0, k)) for v in vs}
        if len(set(table.values())) >= min(k, len(vs)): break
    return np.array([table[v] for v in verbs])


def lexical_baseline(texts, y, tr, te, mp_mask, C=1.0):
    vec = CountVectorizer(token_pattern=r"[A-Za-z_]+")
    Xtr = vec.fit_transform([texts[i] for i in tr]); Xte = vec.transform([texts[i] for i in te])
    m = LogisticRegression(C=C, max_iter=500).fit(Xtr, y[tr]); p = m.predict(Xte)
    out = {"bal_acc": balanced_accuracy_score(y[te], p)}
    sel = mp_mask[te]
    out["bal_acc_minimal"] = balanced_accuracy_score(y[te][sel], p[sel]) if sel.any() else float("nan")
    return out


def run(args):
    recs, H = load(args.data, args.hidden, args.pos)
    if args.minimal_only:
        keep = [i for i, r in enumerate(recs) if r["is_minimal"]]
        recs = [recs[i] for i in keep]; H = H[np.array(keep)]
    L = H.shape[1]; layers = list(range(0, L, args.layer_step))
    if (L - 1) not in layers: layers.append(L - 1)
    results = {"config": vars(args), "layers": layers, "tasks": {}}
    for task in ("binary", "three"):
        idx, y = make_task(recs, task, args.label_key)
        k = 2 if task == "binary" else 3
        sub = [recs[i] for i in idx]
        texts = [r["action_text"] for r in sub]; verbs = [r["verb"] for r in sub]
        mp = np.array([bool(r["is_minimal"]) for r in sub])
        per_seed = []
        for seed in args.seeds:
            tr, va, te = split(sub, np.arange(len(sub)), seed)
            rng = np.random.default_rng(seed)
            ctrl_y = control_labels(verbs, k, seed)
            base = {"majority": balanced_accuracy_score(
                        y[te], np.full(len(te), np.bincount(y[tr]).argmax())),
                    "lexical": lexical_baseline(texts, y, np.concatenate([tr, va]), te, mp),
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                    "class_counts_test": np.bincount(y[te], minlength=k).tolist()}
            rows = []
            for l in layers:
                X = np.asarray(H[idx, l, :], dtype=np.float32)
                m = clf(args.C).fit(X[tr], y[tr])
                pv = m.predict(X[va]); pt = m.predict(X[te]); prt = m.predict_proba(X[te])
                row = {"layer": l, "val_bal_acc": balanced_accuracy_score(y[va], pv),
                       **{f"test_{a}": b for a, b in metrics(y[te], pt, prt, k).items()}}
                lo, hi = bal_ci(y[te], pt); row["test_bal_acc_ci"] = [lo, hi]
                sel = mp[te]
                row["test_bal_acc_minimal"] = balanced_accuracy_score(y[te][sel], pt[sel]) if sel.any() else float("nan")
                # shuffled-label probe (should be near chance)
                ysh = rng.permutation(y[tr]); psh = clf(args.C).fit(X[tr], ysh).predict(X[te])
                row["shuffled_bal_acc"] = balanced_accuracy_score(y[te], psh)
                # Hewitt & Liang control task + selectivity
                pc = clf(args.C).fit(X[tr], ctrl_y[tr]).predict(X[te])
                row["control_bal_acc"] = balanced_accuracy_score(ctrl_y[te], pc)
                row["selectivity"] = row["test_bal_acc"] - row["control_bal_acc"]
                rows.append(row)
                if seed == args.seeds[0]:
                    print(f"[{task}] layer {l:2d}  val {row['val_bal_acc']:.3f}  test {row['test_bal_acc']:.3f}  "
                          f"minimal {row['test_bal_acc_minimal']:.3f}  shuf {row['shuffled_bal_acc']:.3f}  "
                          f"sel {row['selectivity']:.3f}", flush=True)
            per_seed.append({"seed": seed, "baselines": base, "rows": rows})
            if seed == args.seeds[0]:
                best = int(np.argmax([r["val_bal_acc"] for r in rows])); bl = layers[best]
                X = np.asarray(H[idx, bl, :], dtype=np.float32); m = clf(args.C).fit(X[tr], y[tr])
                os.makedirs("results", exist_ok=True)
                with open(f"results/probe_preds_{args.tag}_{task}.csv", "w") as f:
                    f.write("id,true,pred,layer\n")
                    for j, p in zip(te, m.predict(X[te])):
                        f.write(f"{sub[j]['id']},{y[j]},{p},{bl}\n")
        # aggregate across seeds
        agg = {}
        for key in ("val_bal_acc", "test_bal_acc", "test_macro_f1", "test_auroc", "test_bal_acc_minimal",
                    "shuffled_bal_acc", "control_bal_acc", "selectivity"):
            M = np.array([[r[key] for r in s["rows"]] for s in per_seed])
            agg[key] = {"mean": np.nanmean(M, 0).tolist(), "std": np.nanstd(M, 0).tolist()}
        ci = np.array([[r["test_bal_acc_ci"] for r in s["rows"]] for s in per_seed]).mean(0)
        agg["test_bal_acc_ci95"] = ci.tolist()
        best = int(np.argmax(agg["val_bal_acc"]["mean"]))
        results["tasks"][task] = {"per_seed": per_seed, "agg": agg, "best_layer": layers[best],
                                  "baselines_mean": {
                                      "majority": float(np.mean([s["baselines"]["majority"] for s in per_seed])),
                                      "lexical": float(np.mean([s["baselines"]["lexical"]["bal_acc"] for s in per_seed])),
                                      "lexical_minimal": float(np.nanmean([s["baselines"]["lexical"]["bal_acc_minimal"] for s in per_seed]))}}
        t = results["tasks"][task]
        print(f"== {task}: best layer {t['best_layer']} (chosen on validation). test bal-acc "
              f"{agg['test_bal_acc']['mean'][best]:.3f} ± {agg['test_bal_acc']['std'][best]:.3f}; "
              f"baselines {t['baselines_mean']}")
    os.makedirs("results", exist_ok=True)
    json.dump(results, open(f"results/probe_{args.tag}.json", "w"), indent=1)
    plot(results, args.tag)


def plot(results, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True); layers = results["layers"]
    for task, t in results["tasks"].items():
        a = t["agg"]; fig, ax = plt.subplots(figsize=(6.2, 3.8))
        m, s = np.array(a["test_bal_acc"]["mean"]), np.array(a["test_bal_acc"]["std"])
        ax.plot(layers, m, "-o", ms=3, label="linear probe (all test)"); ax.fill_between(layers, m - s, m + s, alpha=.2)
        ax.plot(layers, a["test_bal_acc_minimal"]["mean"], "-s", ms=3, label="linear probe (minimal pairs)")
        ax.plot(layers, a["shuffled_bal_acc"]["mean"], ":", label="shuffled-label probe")
        ax.plot(layers, a["control_bal_acc"]["mean"], ":", label="control task (verb labels)")
        ax.axhline(t["baselines_mean"]["majority"], color="gray", ls="--", label="majority")
        ax.axhline(t["baselines_mean"]["lexical"], color="k", ls="--", label="lexical (all)")
        ax.axhline(t["baselines_mean"]["lexical_minimal"], color="r", ls="--", label="lexical (minimal pairs)")
        ax.axvline(t["best_layer"], color="g", alpha=.3); ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylabel("balanced accuracy"); ax.set_title(f"{tag} – {task}"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"figures/probe_{tag}_{task}.png", dpi=200); plt.close(fig)


def transfer(args):
    """Train on one environment, test on the other (binary task), every layer."""
    recs, H = load(args.data, args.hidden, args.pos)
    idx, y = make_task(recs, "binary", args.label_key)
    envs = np.array([recs[i]["environment"] for i in idx]); out = {}
    for a in sorted(set(envs)):
        for b in sorted(set(envs)):
            if a == b: continue
            tr, te = np.where(envs == a)[0], np.where(envs == b)[0]; rows = []
            for l in range(0, H.shape[1], args.layer_step):
                X = np.asarray(H[idx, l, :], dtype=np.float32); m = clf(args.C).fit(X[tr], y[tr])
                p = m.predict(X[te]); lo, hi = bal_ci(y[te], p)
                rows.append({"layer": l, "bal_acc": balanced_accuracy_score(y[te], p), "ci95": [lo, hi]})
                print(f"{a}->{b} layer {l}: {rows[-1]['bal_acc']:.3f}", flush=True)
            out[f"{a}->{b}"] = rows
    json.dump(out, open(f"results/probe_{args.tag}.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, rows in out.items(): ax.plot([r["layer"] for r in rows], [r["bal_acc"] for r in rows], "-o", ms=3, label=k)
    ax.axhline(.5, color="gray", ls="--"); ax.set_xlabel("layer"); ax.set_ylabel("balanced accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(f"figures/probe_{args.tag}.png", dpi=200)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--hidden", nargs="+", required=True, help="prefixes written by extract_hidden.py")
    ap.add_argument("--pos", choices=["last", "mean"], default="last")
    ap.add_argument("--tag", required=True); ap.add_argument("--label-key", default="label")
    ap.add_argument("--minimal-only", action="store_true"); ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--layer-step", type=int, default=1); ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); seed_all()
    transfer(a) if a.transfer else run(a)
