"""Demo for the presentation: one task, its reversible and irreversible version side by side, from saved agent logs.
python demo.py results/agent_qwen3b_neutral.jsonl              # picks a pair where behaviour differs (else the first)
python demo.py results/agent_qwen3b_neutral.jsonl cleanup-3    # a given task id
Writes results/demo_<task>.md (no model needed)."""
import sys
from utils import read_jsonl

logs = read_jsonl(sys.argv[1]); by = {(l["task_id"], l["version"]): l for l in logs}
ids = sorted({l["task_id"] for l in logs})
tid = sys.argv[2] if len(sys.argv) > 2 else next((i for i in ids if by[(i, "irr")]["careful"] != by[(i, "rev")]["careful"]), ids[0])
first = {v: by[(tid, v)]["messages"][1]["content"] for v in ("rev", "irr")}     # the prompt the agent saw
text = first["irr"].split("Task: ")[1].split("\n\nCurrent state:")[0]
rev, irr = (first[v].split("Current state:\n")[1].splitlines() for v in ("rev", "irr"))
out = [f"# Demo: `{tid}`", "", f"**Task:** {text}", "",
       "**The one state line that differs:** " + "; ".join(f"reversible version `{a}` vs irreversible version `{b}`"
                                                          for a, b in zip(rev, irr) if a != b), ""]
for v, name in (("rev", "Reversible version"), ("irr", "Irreversible version")):
    l = by[(tid, v)]
    out += [f"## {name}", "", f"careful: **{l['careful']}** (tools: {', '.join(l['careful_tools']) or 'none'}; refused: {l['refused']}) · "
            f"took the shortcut: **{l['target_called']}** · irreversible action happened: **{l['irreversible_action']}** · "
            f"task success: {l['success']} · parse failures: {l['parse_failures']}", ""]
    for i, s in enumerate(l["steps"], 1):
        out.append(f"{i}. `{s['raw'].strip()[:200]}`" + (f"  \n   → {s['obs'][:160]}" if s.get("obs") else "  \n   → (could not parse)"))
    out.append("")
open(f"results/demo_{tid}.md", "w", encoding="utf-8").write("\n".join(out)); print("\n".join(out))
