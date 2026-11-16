"""CPU check of "unverified" ScienceWorld labels (no model, no ScienceWorld needed): python test_unverified_labels.py
A failed re-search must mark the row unverified (never keep the budget-60 label), and re-scoring must drop it from the
scores but keep its saved model answer, also when re-scored twice."""
import json, os, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
run = lambda d, *args: subprocess.run([sys.executable, os.path.join(HERE, args[0]), *args[1:]], cwd=d, check=True,
                                      capture_output=True, env={**os.environ, "PYTHONPATH": HERE})
rd = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]

with tempfile.TemporaryDirectory() as d:
    os.makedirs(f"{d}/results")
    base = {"split": "test", "environment": "sciworld", "is_minimal": False, "source": "sw:task:var0:k0", "_seed": 0}
    rows = [{**base, "id": "m", "action_text": "1", "label": "irreversible"}] + \
           [{**base, "id": f"r{i}", "action_text": "open door", "label": "reversible"} for i in range(3)] + \
           [{**base, "id": "i0", "action_text": "eat apple", "label": "irreversible"}]
    open(f"{d}/v1.jsonl", "w").write("".join(json.dumps(r) + "\n" for r in rows))
    # re-search only the menu-number row: it must come back unverified, the rest unchanged and in the same order
    open(f"{d}/menu.jsonl", "w").write(json.dumps(rows[0]) + "\n")
    run(d, "sciworld_label.py", "--relabel", "menu.jsonl", "--out", "menu_out.jsonl", "--workers", "1")
    m = rd(f"{d}/menu_out.jsonl")[0]
    assert m["label"] == m["label_h1"] == "unverified" and m["label_budget60"] == "irreversible", m
    open(f"{d}/relabel.jsonl", "w").write("".join(json.dumps(m if r["id"] == "m" else r) + "\n" for r in rows))

    open(f"{d}/results/stated_t.jsonl", "w").write(
        "".join(json.dumps({k: r[k] for k in ("id", "environment", "is_minimal", "split")} | {"raw": "IRREVERSIBLE"}) + "\n"
                for r in rows))
    for _ in range(2):
        run(d, "stated_answer.py", "--model", "none", "--data", "relabel.jsonl", "--tag", "t", "--test-only", "--rescore")
        saved, out = rd(f"{d}/results/stated_t.jsonl"), json.load(open(f"{d}/results/stated_t.json"))
        assert len(saved) == 5 and [r["true"] for r in saved if r["id"] == "m"] == [None], saved
        assert out["n"] == 4, out
print("unverified-label checks passed")
