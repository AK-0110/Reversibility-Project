"""RQ5 data prep for ByteSized32-State-Prediction (github.com/cognitiveailab/GPT-simulator, data/test.zip -> test.jsonl).
Licence: none shown on the repository -> research use only; ask the authors before redistributing any part.

Each row holds the state before the action (current_state), the state right after the action (action_state) and after the
game tick (tick_state). We predict the NEXT state (tick_state) from the previous state + action.

IMPORTANT (honest limitation): the rows do not contain the game's executable simulator or the action history, so the
Section 4.2 search (save state, apply action, try <=3 further actions) cannot be run on them directly. Two options:
  (a) re-run the BYTESIZED32 game code (github.com/cognitiveailab/BYTESIZED32) and replay from the game's initial state;
  (b) label a stratified sample by hand with the rubric of toolemu_cases.py (two annotators, report Cohen's kappa).
This script supports (b): `export` writes the same stratified-by-chance sample of changed transitions to two annotation
sheets, `build` turns the agreed labels into the uncertainty dataset. Report class counts; if irreversible transitions
are rare in the sample, oversample them by exporting more rows (--n).

  python prepare_bytesized.py export --test test.jsonl --n 400
  python prepare_bytesized.py build  --test test.jsonl --annotations data/bytesized_annotator1.csv [--second ...annotator2.csv]
  python uncertainty.py --model ... --data data/bytesized_labelled.jsonl --tag ...
"""
import argparse, csv, json, random
from utils import write_jsonl

LABELS = {"reversible", "partially_reversible", "irreversible"}


def state_text(s):
    return f"Task: {s['taskDesc']}\nObservation: {s['observation']}\n{s['look']}\nInventory: {s['inventory']}"


def rows(path):
    with open(path) as f:
        for i, l in enumerate(f):
            r = json.loads(l); r["_i"] = i; yield r


def export(a):
    rng = random.Random(0); allr = [r for r in rows(a.test) if r["action_state"]["lastAction"]]
    # keep only transitions where the action changed something visible (otherwise reversibility is moot)
    ch = [r for r in allr if r["current_state"]["look"] != r["tick_state"]["look"]
          or r["current_state"]["inventory"] != r["tick_state"]["inventory"]]
    sample = rng.sample(ch, min(a.n, len(ch)))
    for who in ("annotator1", "annotator2"):
        with open(f"data/bytesized_{who}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "game", "before", "action", "after", "your_label"])
            for r in sample:
                w.writerow([f"bs32-{r['_i']}", r["game"], state_text(r["current_state"]), r["action_state"]["lastAction"],
                            state_text(r["tick_state"]), ""])
    print(len(allr), "transitions;", len(ch), "with a visible change;", len(sample), "exported")


def build(a):
    by = {f"bs32-{r['_i']}": r for r in rows(a.test)}
    first = {r["id"]: r["your_label"].strip() for r in csv.DictReader(open(a.annotations))}
    second = {r["id"]: r["your_label"].strip() for r in csv.DictReader(open(a.second))} if a.second else first
    out = []
    for i, lab in first.items():
        if lab not in LABELS or second.get(i) != lab: continue
        r = by[i]
        out.append({"id": i, "label": lab, "game": r["game"],
                    "prompt": f"{state_text(r['current_state'])}\n\nAction taken: {r['action_state']['lastAction']}",
                    "target_state": f"Observation: {r['tick_state']['observation']}\n{r['tick_state']['look']}\nInventory: {r['tick_state']['inventory']}"})
    write_jsonl("data/bytesized_labelled.jsonl", out); print(len(out), "labelled transitions")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("export"); e.add_argument("--test", required=True); e.add_argument("--n", type=int, default=400)
    b = sp.add_parser("build"); b.add_argument("--test", required=True); b.add_argument("--annotations", required=True)
    b.add_argument("--second", default=None)
    a = ap.parse_args(); export(a) if a.cmd == "export" else build(a)
