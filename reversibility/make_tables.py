"""Collect everything in results/ into the report tables and figures (Section 8):
  results/tables/probe_summary.md      probe vs baselines/controls per tag/task (RQ1)
  results/tables/ablations.md          per-verb, leave-one-verb-out, transfer, reworded test rows, behaviour runs
  results/tables/stated_probe_behaviour_<tag>.md   (RQ2/RQ3) incl. chance-corrected blindness gap
  results/tables/steering.md           (RQ4, if run)
  figures/stated_probed_behaviour_<tag>.png, figures/blindness_gap.png (all models; copy every results/agent_*.json into one results/ first)
python make_tables.py [--model-tag qwen3b]"""
import argparse, glob, json, os
import numpy as np


def load(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def fmt(m, s=None, ci=None):
    if m is None: return "n/a"
    t = f"{m:.3f}"
    if s is not None: t += f" ± {s:.3f}"
    if ci: t += f" [{ci[0]:.3f}, {ci[1]:.3f}]"
    return t


def table(f, hdr, rows):
    f.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
    for r in rows: f.write("| " + " | ".join(map(str, r)) + " |\n")


def probe_tables():
    os.makedirs("results/tables", exist_ok=True); rows, verbs = [], []
    for p in sorted(glob.glob("results/probe_*.json")):
        d = load(p); tag = os.path.basename(p)[6:-5]
        if "tasks" not in d: continue                                  # transfer / reworded files handled below
        for task, t in d["tasks"].items():
            a = t["agg"]; i = d["layers"].index(t["best_layer"]); b = t["baselines_mean"]
            g = lambda k: fmt(a[k]["mean"][i]) if k in a else "n/a"
            rows.append((tag, task, f'{t["best_layer"]} (n_test={t["per_seed"][0]["baselines"]["n_test"]})',
                         fmt(a["test_bal_acc"]["mean"][i], a["test_bal_acc"]["std"][i], a["test_bal_acc_ci95"][i]),
                         g("test_macro_f1"), g("test_auroc"), g("test_bal_acc_minimal"),
                         *(f'{fmt(b[n])} / {fmt(b[n + "_minimal"])}' for n in ("lexical", "state_bow", "state_action_bow")),
                         g("selectivity"), g("selectivity_setting"), g("shuffled_bal_acc"), fmt(b["majority"]),
                         fmt(a["test_bal_acc"]["mean"][0])))
            lv = t.get("leave_one_verb_out")
            if not lv: continue                                       # per-verb rows only for the --lovo runs (last, pre)
            for v, pv in t["per_verb_at_best_layer"].items():
                verbs.append((tag, task, v, pv["n_test_per_seed"][0], fmt(pv["bal_acc_mean"]),
                              fmt(lv[v]["at_best_layer"]["bal_acc"], None, lv[v]["at_best_layer"]["ci95"]) if v in lv else "—"))
    hdr = ("tag", "task", "best layer (test size)", "bal-acc (mean ± sd over 3 splits; CI = mean of per-split 95% bootstraps)",
           "macro-F1", "AUROC", "bal-acc minimal pairs", "action BoW all / minimal", "state BoW all / minimal",
           "state+action BoW all / minimal", "selectivity (verb control)", "selectivity (permuted setting)", "shuffled",
           "majority", "layer-0 probe")
    with open("results/tables/probe_summary.md", "w", encoding="utf-8") as f:
        f.write("Tags: *_last / *_mean / *_pre = token position (pre = last token before the action: sees the state, not the "
                "action); *_minimal = minimal pairs only; *_h1 / *_h2 = horizon. An action-specific claim needs *_last > "
                "*_pre and > state BoW on minimal pairs.\n\n")
        table(f, hdr, rows)
    with open("results/tables/ablations.md", "w", encoding="utf-8") as f:
        f.write("## Per verb (test, best layer, mean over splits) and leave-one-verb-out (binary, train on the other verbs)\n\n")
        table(f, ("tag", "task", "verb", "n test (split 0)", "bal-acc", "leave-one-verb-out bal-acc [95% CI]"), verbs)
        tr = []
        for p in sorted(glob.glob("results/probe_*transfer*.json")):
            for k, rs in load(p).items():
                r = max(rs, key=lambda r: r["val_bal_acc"])            # layer chosen on SOURCE validation, never on target
                tr.append((os.path.basename(p)[6:-5], k, r["layer"], fmt(r["bal_acc"], None, r["ci95"]), fmt(r["bal_acc_centred"])))
        if tr:
            f.write("\n## Cross-environment transfer (binary; layer chosen on source validation)\n\n")
            table(f, ("tag", "direction", "layer", "bal-acc [95% CI]", "bal-acc, target re-centred"), tr)
        sh = []
        for p in sorted(glob.glob("results/probe_*_terse.json") + glob.glob("results/probe_*_flip.json")):
            for task, t in load(p).items():
                r = t["at_best_layer"]
                sh.append((os.path.basename(p)[6:-5], task, t["n"], r["layer"], fmt(r["orig_bal_acc"]), fmt(r["shifted_bal_acc"]),
                           fmt(r["same_prediction"])))
        if sh:
            f.write("\n## Frozen probes on reworded test rows (terse = on/off wording, flip = irrelevant setting flipped; "
                    "same ids and labels)\n\n")
            table(f, ("tag", "task", "n", "layer", "bal-acc original", "bal-acc reworded", "same prediction"), sh)
        ag = [(os.path.basename(p)[6:-5], load(p)) for p in sorted(glob.glob("results/agent_*.json"))]
        ag = [(n, d) for n, d in ag if "caution_rate_irr" in d]
        if ag:
            f.write("\n## Behaviour runs (system prompt ablation, pilot, model); CIs over pairs / over templates\n\n")
            table(f, ("run", "pairs", "target taken irr / rev", "refused irr / rev", "caution irr / rev",
                      "sensitivity A* [templates CI]", "A_out*", "success irr / rev", "parse failures"),
                  [(n, d["n_pairs"], f'{d["target_action_rate_irr_version"]:.2f} / {d["target_action_rate_rev_version"]:.2f}',
                    f'{d["refusal_rate_irr"]:.2f} / {d["refusal_rate_rev"]:.2f}', f'{d["caution_rate_irr"]:.2f} / {d["over_caution_rate_rev"]:.2f}',
                    fmt(d["A_star"], None, d["A_star_ci95"]["templates"]), fmt(d["A_out_star"]),
                    f'{d["task_success_irr"]:.2f} / {d["task_success_rev"]:.2f}', d["parse_failures_total"]) for n, d in ag])


def comparison(tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    st = load(f"results/stated_{tag}.json"); ag = load(f"results/agent_{tag}_neutral.json")
    if not (st and ag): print(f"comparison skipped: need stated_{tag}, agent_{tag}_neutral"); return
    vp = st["vs_probe"].get("binary")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    if vp:                                                     # same test ids for both bars (binary, balanced accuracy)
        axs[0].bar([0, 1], [vp["probe_bal_acc"], vp["stated_bal_acc"]]); axs[0].axhline(.5, color="gray", ls="--")
        axs[0].set_xticks([0, 1]); axs[0].set_xticklabels(["probe (encodes)", "stated (says)"]); axs[0].set_ylim(0, 1.05)
        axs[0].set_ylabel("balanced accuracy, binary"); axs[0].set_title(f"same {vp['n']} test rows", fontsize=8)
    keys = [("K_star", "K* (knows)"), ("A_star", "A* (acts, caution)"), ("A_out_star", "A_out* (acts, outcome)"), ("gap_star", "Gap* = K* − A*")]
    m = [ag[k] for k, _ in keys]; c = [ag[k + "_ci95"]["templates"] for k, _ in keys]
    axs[1].bar(range(4), m, yerr=[[x - l for x, (l, _) in zip(m, c)], [h - x for x, (_, h) in zip(m, c)]], capsize=4)
    axs[1].axhline(0, color="k", lw=.5); axs[1].set_xticks(range(4)); axs[1].set_xticklabels([l for _, l in keys], rotation=15, fontsize=7)
    axs[1].set_ylim(-1.05, 1.05); axs[1].set_title("chance-corrected; 95% CI over templates", fontsize=8)
    fig.tight_layout(); fig.savefig(f"figures/stated_probed_behaviour_{tag}.png", dpi=200); plt.close(fig)
    runs = [(os.path.basename(p)[6:-5], load(p)) for p in sorted(glob.glob("results/agent_*.json"))]
    runs = [(n, d) for n, d in runs if "gap_star" in d and "_pilot" not in n]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.bar(range(len(runs)), [d["gap_star"] for _, d in runs],
           yerr=[[d["gap_star"] - d["gap_star_ci95"]["templates"][0] for _, d in runs],
                 [d["gap_star_ci95"]["templates"][1] - d["gap_star"] for _, d in runs]], capsize=4)
    ax.axhline(0, color="k", lw=.5); ax.set_xticks(range(len(runs))); ax.set_xticklabels([n for n, _ in runs], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Gap* = K* − A*"); fig.tight_layout(); fig.savefig("figures/blindness_gap.png", dpi=200); plt.close(fig)
    with open(f"results/tables/stated_probe_behaviour_{tag}.md", "w", encoding="utf-8") as f:
        f.write("| quantity | value [95% CI] |\n|---|---|\n")
        rows = [("stated answer, binary bal-acc (strict: parse failures wrong)", fmt(st["binary_strict"]["bal_acc"])),
                ("stated answer, 3-class bal-acc (strict)", fmt(st["three_strict"]["bal_acc"])),
                ("stated parse failures", f'{st["parse_failures"]} / {st["n"]}')]
        for task, v in st["vs_probe"].items():
            rows += [(f"{task}: probe vs stated bal-acc on the same {v['n']} rows ({v.get('n_dropped_parse_failures', '?')} parse failures dropped)", f'{v["probe_bal_acc"]:.3f} vs {v["stated_bal_acc"]:.3f}'),
                     (f"{task}: kappa(stated, probe)", fmt(v["kappa_stated_vs_probe"]))]
        T = lambda k: fmt(ag[k], None, ag[k + "_ci95"]["templates"])
        rows += [("K: names the irreversible version (both orders; chance 0.5)", fmt(ag["K_states_correct_version"], None, ag["K_ci95"])),
                 ("K both orders correct (chance 0.25)", fmt(ag["K_both_orders_correct"])),
                 ("K_single: right on each version's state alone", T("K_single")),
                 ("K* = 2K − 1", T("K_star")), ("A (brief): careful only in the irreversible version", fmt(ag["A_more_careful_in_irr_version"], None, ag["A_ci95"])),
                 ("A* = sensitivity", T("A_star")), ("A_out*: avoids the shortcut only where irreversible", T("A_out_star")),
                 ("Gap (brief) = K − A, ~0.5 under no knowledge", fmt(ag["blindness_gap"], None, ag["gap_ci95"])),
                 ("Gap* = K* − A* (headline)", T("gap_star")), ("Gap*_outcome = K* − A_out*", T("gap_star_outcome")),
                 ("McNemar p (caution irr vs rev)", f'{ag["mcnemar_p"]:.3g}'), ("careful 2×2 (both / irr only / rev only / neither)",
                  " / ".join(str(ag["careful_2x2"][k]) for k in ("both", "irr_only", "rev_only", "neither")))]
        for k, v in rows: f.write(f"| {k} | {v} |\n")
    s = load(f"results/steering_{tag}.json")
    if s:
        with open("results/tables/steering.md", "w", encoding="utf-8") as f:
            f.write(f"Layer {s['layer']}, ||v|| = {s['direction_norm']:.2f}, median ||h|| = {s['median_hidden_norm']:.2f}, {s['n_pairs']} pairs\n\n")
            table(f, ("direction", "fraction of median ||h||", "caution irr / rev", "target taken irr", "success irr / rev", "parse failures"),
                  [(r["direction"], r["fraction"], f'{r["caution_rate_irr"]:.2f} / {r["over_caution_rate_rev"]:.2f}',
                    f'{r["target_action_rate_irr_version"]:.2f}', f'{r["task_success_irr"]:.2f} / {r["task_success_rev"]:.2f}',
                    r["parse_failures_total"]) for r in s["runs"]])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--model-tag", default="qwen3b"); a = ap.parse_args()
    probe_tables(); comparison(a.model_tag); print("tables written to results/tables/")
