"""Collect everything in results/ into the report tables and figures (Section 8):
  results/tables/probe_summary.md      probe vs baselines per tag/task (RQ1)
  results/tables/ablations.md          one row per ablation tag
  results/tables/stated_probe_behaviour.md   (RQ2/RQ3) + blindness gap with 95% CI
  figures/stated_probed_behaviour.png, figures/blindness_gap.png
python make_tables.py [--model-tag qwen7b]"""
import argparse, glob, json, os
import numpy as np


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def fmt(m, s=None, ci=None):
    t = f"{m:.3f}"
    if s is not None: t += f" ± {s:.3f}"
    if ci: t += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return t


def probe_tables():
    os.makedirs("results/tables", exist_ok=True); rows = []; abl = []
    for p in sorted(glob.glob("results/probe_*.json")):
        d = load(p); tag = os.path.basename(p)[6:-5]
        if "tasks" not in d: continue                                  # transfer files handled below
        for task, t in d["tasks"].items():
            a = t["agg"]; i = d["layers"].index(t["best_layer"]); b = t["baselines_mean"]
            row = (tag, task, f'{t["best_layer"]} (n_test={t["per_seed"][0]["baselines"]["n_test"]})', fmt(a["test_bal_acc"]["mean"][i], a["test_bal_acc"]["std"][i], a["test_bal_acc_ci95"][i]),
                   fmt(a["test_macro_f1"]["mean"][i]), fmt(a["test_auroc"]["mean"][i]),
                   fmt(a["test_bal_acc_minimal"]["mean"][i]), fmt(a["selectivity"]["mean"][i]),
                   fmt(a["shuffled_bal_acc"]["mean"][i]), fmt(b["majority"]), fmt(b["lexical"]), fmt(b["lexical_minimal"]),
                   fmt(a["test_bal_acc"]["mean"][0]))
            rows.append(row); abl.append(row)
    hdr = ("tag", "task", "best layer (test size)", "bal-acc (mean ± sd, 95% CI)", "macro-F1", "AUROC", "bal-acc minimal pairs",
           "selectivity", "shuffled", "majority", "lexical", "lexical (minimal)", "layer-0 probe")
    with open("results/tables/probe_summary.md", "w") as f:
        f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
        for r in rows: f.write("| " + " | ".join(map(str, r)) + " |\n")
    # cross-environment transfer
    tr = []
    for p in sorted(glob.glob("results/probe_*transfer*.json")):
        d = load(p)
        for k, rs in d.items():
            best = max(rs, key=lambda r: r["bal_acc"]); tr.append((os.path.basename(p)[6:-5], k, best["layer"], fmt(best["bal_acc"], None, best["ci95"])))
    with open("results/tables/ablations.md", "w") as f:
        f.write("## Probe ablations (see tag names: *_last/_mean = token position, *_minimal = minimal pairs only, *_h1/_h2/_h3 = horizon, "
                "*_14b = model size; binary vs three-class are the two task rows)\n\n")
        f.write("| " + " | ".join(hdr[:8]) + " |\n|" + "---|" * 8 + "\n")
        for r in abl: f.write("| " + " | ".join(map(str, r[:8])) + " |\n")
        if tr:
            f.write("\n## Cross-environment transfer (binary, best layer)\n\n| tag | direction | layer | bal-acc [95% CI] |\n|---|---|---|---|\n")
            for r in tr: f.write("| " + " | ".join(map(str, r)) + " |\n")
        ag = sorted(glob.glob("results/agent_*.json"))
        if ag:
            f.write("\n## Behaviour ablations (system prompt, model)\n\n| run | sensitivity [95% CI] | caution irr | over-caution rev | success irr | gap [95% CI] |\n|---|---|---|---|---|---|\n")
            for p in ag:
                d = load(p)
                if "caution_rate_irr" not in d: continue
                f.write(f"| {os.path.basename(p)[6:-5]} | {fmt(d['reversibility_sensitivity'], None, d['sensitivity_ci95'])} | "
                        f"{d['caution_rate_irr']:.3f} | {d['over_caution_rate_rev']:.3f} | {d['task_success_irr']:.3f} | "
                        f"{fmt(d['blindness_gap'], None, d['gap_ci95']) if 'blindness_gap' in d else 'n/a'} |\n")


def comparison(tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    pr = load(f"results/probe_{tag}_last.json"); st = load(f"results/stated_{tag}.json"); ag = load(f"results/agent_{tag}_neutral.json")
    if not (pr and st and ag): print("comparison figure skipped: need probe_{tag}_last, stated_{tag}, agent_{tag}_neutral"); return
    t = pr["tasks"]["three"]; i = pr["layers"].index(t["best_layer"])
    vals = {"probe (best layer)": (t["agg"]["test_bal_acc"]["mean"][i], t["agg"]["test_bal_acc_ci95"][i]),
            "stated answer": (st["stated_accuracy"], st["accuracy_ci95"]),
            "K: states correct version": (ag["K_states_correct_version"], ag["K_ci95"]),
            "A: acts more carefully": (ag["A_more_careful_in_irr_version"], ag["A_ci95"])}
    fig, ax = plt.subplots(figsize=(6.2, 3.4)); xs = range(len(vals))
    ax.bar(xs, [v[0] for v in vals.values()], yerr=[[v[0] - v[1][0] for v in vals.values()], [v[1][1] - v[0] for v in vals.values()]], capsize=4)
    ax.set_xticks(list(xs)); ax.set_xticklabels(list(vals), rotation=20, ha="right", fontsize=7); ax.set_ylim(0, 1.05)
    ax.set_title("What the model encodes, says and does"); fig.tight_layout(); fig.savefig("figures/stated_probed_behaviour.png", dpi=200)
    fig, ax = plt.subplots(figsize=(4.2, 3.2)); runs = sorted(glob.glob("results/agent_*.json")); labs, g, lo, hi = [], [], [], []
    for p in runs:
        d = load(p)
        if "blindness_gap" in d:
            labs.append(os.path.basename(p)[6:-5]); g.append(d["blindness_gap"]); lo.append(d["blindness_gap"] - d["gap_ci95"][0]); hi.append(d["gap_ci95"][1] - d["blindness_gap"])
    if g:
        ax.bar(range(len(g)), g, yerr=[lo, hi], capsize=4); ax.axhline(0, color="k", lw=.5)
        ax.set_xticks(range(len(g))); ax.set_xticklabels(labs, rotation=20, ha="right", fontsize=7); ax.set_ylabel("Gap = K − A")
        fig.tight_layout(); fig.savefig("figures/blindness_gap.png", dpi=200)
    with open("results/tables/stated_probe_behaviour.md", "w") as f:
        f.write("| quantity | value [95% CI] |\n|---|---|\n")
        for k, (m, ci) in vals.items(): f.write(f"| {k} | {fmt(m, None, ci)} |\n")
        f.write(f"| kappa(stated, probe) | {st.get('kappa_stated_vs_probe', float('nan')):.3f} |\n")
        f.write(f"| blindness gap K−A | {fmt(ag['blindness_gap'], None, ag['gap_ci95'])} |\n")
        f.write(f"| reversibility sensitivity | {fmt(ag['reversibility_sensitivity'], None, ag['sensitivity_ci95'])} (McNemar p = {ag['mcnemar_p']:.3g}) |\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model-tag", default="qwen7b"); a = ap.parse_args()
    probe_tables(); comparison(a.model_tag); print("tables written to results/tables/")
