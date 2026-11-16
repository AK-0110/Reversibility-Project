"""Step 3 (RQ1): linear probes per layer with every baseline/control from Section 7.2,
3 split-seeds, bootstrap CIs, minimal-pair evaluation, cross-environment transfer.

Examples
  python probes.py --data data/sandbox.jsonl data/sciworld.jsonl \
      --hidden data/h_qwen7b_sandbox data/h_qwen7b_sciworld --pos last --tag qwen7b_last --lovo
  python probes.py ... --pos pre --tag qwen7b_pre --lovo     # control: token before the action (state only)
  python probes.py ... --pos mean --tag qwen7b_mean            # ablation: token position
  python probes.py ... --minimal-only --tag qwen7b_minimal     # ablation: minimal pairs only
  python probes.py ... --label-key label_h1 --tag qwen7b_h1    # ablation: horizon 1/2/3
  python probes.py ... --transfer --tag qwen7b_transfer        # ablation: cross-environment transfer
  python probes.py ... --eval-data data/sandbox_terse.jsonl --eval-hidden data/h_qwen7b_terse --tag qwen7b_terse
                                                               # frozen probes on reworded copies of the test rows
Binary vs three-class is always reported (the two tasks)."""
import argparse, csv, json, os, warnings
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
# sandbox state lines that mark a setting as "unsafe" (see Sandbox.state_text)
UNSAFE_LINE = {"recycle_bin": "Recycle bin: disabled", "versioning": "Edit history: disabled",
               "email_undo": "Email: mail is delivered immediately", "transfer_reversible": "Transfers: are final"}
# permuted-setting control: each verb is bound to a setting that does NOT decide its reversibility
DERANGE = {"delete_file": "email_undo", "send_email": "transfer_reversible", "transfer_money": "versioning",
           "edit_file": "recycle_bin"}


def load(data_paths, hidden_prefixes, pos):
    recs, H = [], []
    for dp, hp in zip(data_paths, hidden_prefixes):
        r = read_jsonl(dp); h = np.load(f"{hp}_{pos}.npy", mmap_mode="r")
        assert len(r) == h.shape[0], f"{dp}: {len(r)} records vs {h.shape[0]} hidden rows"
        recs += r; H.append(h)
    return recs, np.concatenate(H, axis=0)


def make_task(recs, task, label_key):
    lab = lambda r: r.get(label_key, r["label"])          # environments without per-horizon labels fall back to `label`
    if task == "binary":
        idx = [i for i, r in enumerate(recs) if lab(r) in (LAB3[0], LAB3[2])]
        y = np.array([int(lab(recs[i]) == LAB3[2]) for i in idx])
    else:
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


def setting_control(r):
    """1/0 = the DERANGE setting of this verb is unsafe/safe; -1 for other verbs and environments."""
    s = DERANGE.get(r["verb"]) if r["environment"] == "sandbox" else None
    return -1 if s is None else int(UNSAFE_LINE[s] in r["state_text"])


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
    os.makedirs("results", exist_ok=True)
    for task in ("binary", "three"):
        idx, y = make_task(recs, task, args.label_key)
        k = 2 if task == "binary" else 3
        sub = [recs[i] for i in idx]
        verbs = [r["verb"] for r in sub]
        # lexical baselines: action only (0.5 on minimal pairs by construction), state only, state + action
        texts = {"lexical": [r["action_text"] for r in sub], "state_bow": [r["state_text"] for r in sub],
                 "state_action_bow": [r["state_text"] + " " + r["action_text"] for r in sub]}
        mp = np.array([bool(r["is_minimal"]) for r in sub])
        sc = np.array([setting_control(r) for r in sub]) if task == "binary" else np.full(len(sub), -1)
        per_seed = []
        for seed in args.seeds:
            tr, va, te = split(sub, np.arange(len(sub)), seed)
            rng = np.random.default_rng(seed)
            ctrl_y = control_labels(verbs, k, seed)
            base = {"majority": balanced_accuracy_score(y[te], np.full(len(te), np.bincount(y[tr]).argmax())),
                    **{n: lexical_baseline(t, y, np.concatenate([tr, va]), te, mp) for n, t in texts.items()},
                    "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
                    "class_counts_test": np.bincount(y[te], minlength=k).tolist()}
            s_tr, s_te = tr[sc[tr] >= 0], te[sc[te] >= 0]
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
                if len(s_tr) and len(s_te):
                    # permuted-setting control: same structure (verb x one setting line, same danger phrases) but the
                    # setting is irrelevant to the action. real ~ control => the probe reads any verb-bound setting
                    ps = clf(args.C).fit(X[s_tr], sc[s_tr]).predict(X[s_te])
                    row["setting_control_bal_acc"] = balanced_accuracy_score(sc[s_te], ps)
                    row["setting_real_same_rows"] = balanced_accuracy_score(y[s_te], pt[sc[te] >= 0])
                    row["selectivity_setting"] = row["setting_real_same_rows"] - row["setting_control_bal_acc"]
                rows.append(row)
                if seed == args.seeds[0]:
                    print(f"[{task}] layer {l:2d}  val {row['val_bal_acc']:.3f}  test {row['test_bal_acc']:.3f}  "
                          f"minimal {row['test_bal_acc_minimal']:.3f}  shuf {row['shuffled_bal_acc']:.3f}  "
                          f"sel {row['selectivity']:.3f}  sel_setting {row.get('selectivity_setting', float('nan')):.3f}", flush=True)
            per_seed.append({"seed": seed, "baselines": base, "rows": rows, "split": (tr, te)})
        # aggregate across seeds
        agg = {}
        for key in [k_ for k_ in per_seed[0]["rows"][0] if k_ not in ("layer", "test_bal_acc_ci")]:
            M = np.array([[r[key] for r in s["rows"]] for s in per_seed])
            agg[key] = {"mean": np.nanmean(M, 0).tolist(), "std": np.nanstd(M, 0).tolist()}
        # mean of the per-seed bootstrap endpoints (not a pooled bootstrap); label it so in the report
        agg["test_bal_acc_ci95"] = np.array([[r["test_bal_acc_ci"] for r in s["rows"]] for s in per_seed]).mean(0).tolist()
        best = int(np.argmax(agg["val_bal_acc"]["mean"])); bl = layers[best]
        X = np.asarray(H[idx, bl, :], dtype=np.float32)
        per_verb, lovo = {}, None
        groups = [r["verb"] if r["environment"] == "sandbox" else r["environment"] + " (all verbs)" for r in sub]
        # probe scores for every seed at the aggregate best layer (seed 0 = stored split, used by stated_answer.py)
        with open(f"results/probe_preds_{args.tag}_{task}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "seed", "layer", "true", "pred"] + [f"p{c}" for c in range(k)])
            for s in per_seed:
                tr, te = s.pop("split")
                m = clf(args.C).fit(X[tr], y[tr]); pt = m.predict(X[te]); pr = m.predict_proba(X[te])
                for j, p, q in zip(te, pt, pr): w.writerow([sub[j]["id"], s["seed"], bl, y[j], p] + [f"{v:.4f}" for v in q])
                for v in sorted(set(groups)):
                    sel = np.array([groups[j] == v for j in te])
                    if sel.any():
                        per_verb.setdefault(v, []).append((balanced_accuracy_score(y[te][sel], pt[sel]), int(sel.sum())))
        per_verb = {v: {"bal_acc_mean": float(np.mean([a for a, _ in xs])), "n_test_per_seed": [n for _, n in xs]}
                    for v, xs in per_verb.items()}
        if args.lovo and task == "binary":
            lovo = leave_one_verb_out(sub, y, H[idx], layers, args.C, bl)
        results["tasks"][task] = {"per_seed": per_seed, "agg": agg, "best_layer": bl, "per_verb_at_best_layer": per_verb,
                                  "leave_one_verb_out": lovo,
                                  "baselines_mean": {
                                      "majority": float(np.mean([s["baselines"]["majority"] for s in per_seed])),
                                      **{n + suf: float(np.nanmean([s["baselines"][n][key] for s in per_seed]))
                                         for n in texts for suf, key in (("", "bal_acc"), ("_minimal", "bal_acc_minimal"))}}}
        t = results["tasks"][task]
        print(f"== {task}: best layer {bl} (chosen on validation). test bal-acc "
              f"{agg['test_bal_acc']['mean'][best]:.3f} ± {agg['test_bal_acc']['std'][best]:.3f}; "
              f"baselines {t['baselines_mean']}; per verb {per_verb}")
    json.dump(results, open(f"results/probe_{args.tag}.json", "w"), indent=1)
    plot(results, args.tag)


def leave_one_verb_out(sub, y, H, layers, C, best_layer):
    """Binary, sandbox only: train on every other verb, test on the held-out verb (its label hangs on a setting line
    the probe never saw as relevant). Success means an abstract 'irreversible' direction, not per-line features."""
    sb = np.array([r["environment"] == "sandbox" for r in sub]); verbs = np.array([r["verb"] for r in sub])
    out = {}
    for v in sorted(set(verbs[sb])):
        te = np.where(sb & (verbs == v))[0]; tr = np.where(sb & (verbs != v))[0]
        if len(set(y[te])) < 2: continue                                # held-out verb needs both classes
        rows = []
        for l in layers:
            X = np.asarray(H[:, l, :], dtype=np.float32); p = clf(C).fit(X[tr], y[tr]).predict(X[te])
            lo, hi = bal_ci(y[te], p); rows.append({"layer": l, "bal_acc": balanced_accuracy_score(y[te], p), "ci95": [lo, hi]})
        out[v] = {"n_test": int(len(te)), "at_best_layer": rows[layers.index(best_layer)], "rows": rows}
        print(f"LOVO held-out {v} (n={len(te)}): bal-acc {out[v]['at_best_layer']['bal_acc']:.3f} at layer {best_layer}", flush=True)
    return out


def plot(results, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True); layers = results["layers"]
    for task, t in results["tasks"].items():
        a = t["agg"]; b = t["baselines_mean"]; fig, ax = plt.subplots(figsize=(6.2, 3.8))
        m, s = np.array(a["test_bal_acc"]["mean"]), np.array(a["test_bal_acc"]["std"])
        ax.plot(layers, m, "-o", ms=3, label="linear probe (all test)"); ax.fill_between(layers, m - s, m + s, alpha=.2)
        ax.plot(layers, a["test_bal_acc_minimal"]["mean"], "-s", ms=3, label="linear probe (minimal pairs)")
        ax.plot(layers, a["shuffled_bal_acc"]["mean"], ":", label="shuffled-label probe")
        ax.plot(layers, a["control_bal_acc"]["mean"], ":", label="control task (verb labels)")
        if "setting_control_bal_acc" in a:
            ax.plot(layers, a["setting_control_bal_acc"]["mean"], ":", label="control (permuted setting)")
        ax.axhline(b["majority"], color="gray", ls="--", label="majority")
        ax.axhline(b["lexical_minimal"], color="r", ls="--", label="action BoW (minimal pairs)")
        ax.axhline(b["state_bow_minimal"], color="m", ls="--", label="state BoW (minimal pairs)")
        ax.axvline(t["best_layer"], color="g", alpha=.3); ax.set_xlabel("layer (0 = embeddings)")
        ax.set_ylabel("balanced accuracy"); ax.set_title(f"{tag} – {task}"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(f"figures/probe_{tag}_{task}.png", dpi=200); plt.close(fig)


def transfer(args):
    """Train on one environment, test on the other (binary task), every layer. The layer is chosen on a grouped
    validation split of the SOURCE environment. 'centred' re-centres target features on the source mean
    (unsupervised; JSON and natural-language actions differ systematically)."""
    recs, H = load(args.data, args.hidden, args.pos)
    idx, y = make_task(recs, "binary", args.label_key)
    envs = np.array([recs[i]["environment"] for i in idx]); groups = np.array([recs[i]["scenario_id"] for i in idx])
    out = {}
    for a in sorted(set(envs)):
        src = np.where(envs == a)[0]
        trn, val = (src[x] for x in next(GroupShuffleSplit(1, test_size=0.2, random_state=100).split(src, groups=groups[src])))
        for b in sorted(set(envs)):
            if a == b: continue
            te = np.where(envs == b)[0]; rows = []
            for l in range(0, H.shape[1], args.layer_step):
                X = np.asarray(H[idx, l, :], dtype=np.float32); m = clf(args.C).fit(X[trn], y[trn])
                p = m.predict(X[te]); lo, hi = bal_ci(y[te], p)
                pc = m.predict(X[te] - X[te].mean(0) + X[trn].mean(0))
                rows.append({"layer": l, "val_bal_acc": balanced_accuracy_score(y[val], m.predict(X[val])),
                             "bal_acc": balanced_accuracy_score(y[te], p), "ci95": [lo, hi],
                             "bal_acc_centred": balanced_accuracy_score(y[te], pc)})
                print(f"{a}->{b} layer {l}: val {rows[-1]['val_bal_acc']:.3f} target {rows[-1]['bal_acc']:.3f} "
                      f"centred {rows[-1]['bal_acc_centred']:.3f}", flush=True)
            out[f"{a}->{b}"] = rows
    json.dump(out, open(f"results/probe_{args.tag}.json", "w"), indent=1)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, rows in out.items(): ax.plot([r["layer"] for r in rows], [r["bal_acc"] for r in rows], "-o", ms=3, label=k)
    ax.axhline(.5, color="gray", ls="--"); ax.set_xlabel("layer"); ax.set_ylabel("balanced accuracy"); ax.legend()
    fig.tight_layout(); fig.savefig(f"figures/probe_{args.tag}.png", dpi=200)


def shifted(args):
    """Frozen probes (fit on the stored train split of --data) scored on reworded copies of test rows (--eval-data,
    same ids). Compares accuracy on the original vs the reworded rows, on the same ids, and how often the
    prediction stays the same."""
    recs, H = load(args.data, args.hidden, args.pos); erecs, EH = load(args.eval_data, args.eval_hidden, args.pos)
    L = H.shape[1]; layers = list(range(0, L, args.layer_step)); out = {}
    for task in ("binary", "three"):
        idx, y = make_task(recs, task, "label"); eidx, ey = make_task(erecs, task, "label")
        sub = [recs[i] for i in idx]; tr, va, te = split(sub, np.arange(len(sub)), 0)
        pos = {sub[j]["id"]: j for j in te}
        pairs = [(pos[erecs[i]["id"]], n) for n, i in enumerate(eidx) if erecs[i]["id"] in pos]
        o, e = np.array([p for p, _ in pairs]), np.array([n for _, n in pairs])
        assert len(o) and (y[o] == ey[e]).all(), "reworded rows must keep the original label"
        rows = []
        for l in layers:
            X = np.asarray(H[idx, l, :], dtype=np.float32); EX = np.asarray(EH[eidx, l, :], dtype=np.float32)
            m = clf(args.C).fit(X[tr], y[tr]); po, pe = m.predict(X[o]), m.predict(EX[e])
            rows.append({"layer": l, "val_bal_acc": balanced_accuracy_score(y[va], m.predict(X[va])),
                         "orig_bal_acc": balanced_accuracy_score(y[o], po), "shifted_bal_acc": balanced_accuracy_score(ey[e], pe),
                         "same_prediction": float((po == pe).mean())})
            print(f"[{task}] layer {l}: original {rows[-1]['orig_bal_acc']:.3f}  reworded {rows[-1]['shifted_bal_acc']:.3f}  "
                  f"same pred {rows[-1]['same_prediction']:.3f}", flush=True)
        best = max(rows, key=lambda r: r["val_bal_acc"])
        out[task] = {"n": int(len(o)), "best_layer": best["layer"], "at_best_layer": best, "rows": rows}
    json.dump(out, open(f"results/probe_{args.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--hidden", nargs="+", required=True, help="prefixes written by extract_hidden.py")
    ap.add_argument("--pos", choices=["last", "mean", "pre"], default="last")
    ap.add_argument("--tag", required=True); ap.add_argument("--label-key", default="label")
    ap.add_argument("--minimal-only", action="store_true"); ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--lovo", action="store_true", help="also run leave-one-verb-out (binary, sandbox)")
    ap.add_argument("--eval-data", nargs="+"); ap.add_argument("--eval-hidden", nargs="+")
    ap.add_argument("--layer-step", type=int, default=1); ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args(); seed_all()
    shifted(a) if a.eval_data else transfer(a) if a.transfer else run(a)
