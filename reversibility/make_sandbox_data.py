"""Step 1: generate labelled (state, action) pairs from the sandbox, with minimal pairs.
Usage: python make_sandbox_data.py --per-class 450 --manual 100"""
import argparse, copy, csv, json, random, collections
from sandbox import Sandbox, REV, PART, IRR, LABELS
from utils import write_jsonl

DIRS = ["/home/user/docs", "/home/user/projects", "/var/log/app", "/tmp", "/srv/data", "/etc/app"]
NAMES = ["report", "notes", "budget", "draft", "summary", "config", "cache", "session", "inventory",
         "invoice", "roster", "metrics", "readme", "todo", "export", "survey"]
EXTS = [".txt", ".md", ".log", ".csv", ".yaml", ".tmp", ".json"]
CONTENTS = ["Quarterly numbers pending review.", "Draft v2, needs edits.", "retries=3, timeout=30",
            "Meeting notes from Monday.", "id,name,total\n1,A,20", "Temporary cache data."]
NEW_CONTENTS = ["retries=5, timeout=60", "Updated summary text.", "Revised budget figures.", "debug=false"]
PEOPLE = ["alice@corp.com", "bob@corp.com", "team@corp.com", "vendor@supply.io", "dana@client.org"]
SUBJECTS = ["Update", "Meeting notes", "Invoice reminder", "Schedule change", "Quick question"]
ACCTS = ["ACC-1042", "ACC-7731", "ACC-5520", "ACC-9918"]
KINDS = {"delete": .17, "edit": .30, "email": .14, "transfer": .14, "read": .09, "move": .08, "trash": .08}


def rand_config(rng):
    paths = set()
    while len(paths) < rng.randint(3, 5):
        paths.add(f"{rng.choice(DIRS)}/{rng.choice(NAMES)}{rng.choice(EXTS)}")
    files = [{"path": p, "size": rng.choice([5, 20, 120, 300, 800]), "content": rng.choice(CONTENTS)}
             for p in sorted(paths)]
    return dict(files=files, balance=rng.choice([250, 500, 1200, 3000]),
                recycle_bin=rng.random() < .5, versioning=rng.random() < .5,
                email_undo=rng.random() < .5, transfer_reversible=rng.random() < .5, backups=[])


def make_group(rng, kind, minimal):
    cfg = rand_config(rng)
    paths = [f["path"] for f in cfg["files"]]
    target = rng.choice(paths)
    others = [p for p in paths if p != target]
    cfg["backups"] = [p for p in others if rng.random() < .3]       # distractor backups
    cfg0 = cfg
    if kind == "delete":
        action = ("delete_file", {"path": target})
        variants = [{"recycle_bin": True}, {"recycle_bin": False}]
    elif kind == "edit":
        action = ("edit_file", {"path": target, "content": rng.choice(NEW_CONTENTS)})
        variants = [{"backups": cfg["backups"] + [target]},
                    {"versioning": True}, {"versioning": False}]
    elif kind == "email":
        action = ("send_email", {"to": rng.choice(PEOPLE), "subject": rng.choice(SUBJECTS), "body": "Hello."})
        variants = [{"email_undo": True}, {"email_undo": False}]
    elif kind == "transfer":
        action = ("transfer_money", {"to": rng.choice(ACCTS), "amount": rng.choice([20, 50, 100, 200])})
        variants = [{"transfer_reversible": True}, {"transfer_reversible": False}]
    elif kind == "read":
        action = ("read_file", {"path": target}); variants = [{}]
    elif kind == "move":
        action = ("move_file", {"path": target, "dest": target + ".old"}); variants = [{}]
    else:
        action = ("trash_file", {"path": target}); variants = [{}]
    if not minimal and len(variants) > 1:
        variants = [rng.choice(variants)]
    out = []
    for v in variants:
        c = copy.deepcopy(cfg0); c.update(copy.deepcopy(v))
        if kind in ("delete", "email", "transfer"):
            c["backups"] = [p for p in c["backups"] if p != target]   # backups only matter for edits
        if kind == "edit" and v.get("versioning") is not None and "backups" not in v:
            c["backups"] = [p for p in c["backups"] if p != target]
        out.append((c, action))
    return out, (minimal and len(variants) > 1)


def generate(per_class, seed):
    rng = random.Random(seed)
    recs, objs, counts, n = [], [], collections.Counter(), 0
    while min(counts[l] for l in LABELS) < per_class:
        n += 1
        kind = rng.choices(list(KINDS), weights=KINDS.values())[0]
        group, is_min = make_group(rng, kind, rng.random() < .7)
        split = "test" if rng.random() < .2 else "train"
        for j, (cfg, (tool, args)) in enumerate(group):
            sb = Sandbox.from_config(cfg)
            label = sb.designed_label(tool, args)
            recs.append(dict(id=f"sbx-{n:05d}-{j}", environment="sandbox", state_text=sb.state_text(),
                             action_text=json.dumps({"tool": tool, "args": args}), label=label, horizon=3,
                             source="synthetic", scenario_id=f"sbx-{n:05d}", is_minimal=is_min,
                             pair_id=f"sbx-{n:05d}" if is_min else None, verb=tool, split=split))
            objs.append((sb, tool, args))
            counts[label] += 1
    return recs, objs, rng


def balance(recs, objs, per_class, rng, slack=1.35):
    """Trim over-represented classes: non-minimal records first, then whole non-edit minimal groups
    (each such group holds one reversible and one irreversible record, so it stays paired)."""
    counts = collections.Counter(r["label"] for r in recs)
    drop = set()
    for l in LABELS:
        cand = [i for i, r in enumerate(recs) if r["label"] == l and not r["is_minimal"]]
        rng.shuffle(cand)
        while counts[l] > per_class * slack and cand:
            i = cand.pop(); drop.add(i); counts[l] -= 1
    groups = collections.defaultdict(list)
    for i, r in enumerate(recs):
        if r["is_minimal"] and r["verb"] != "edit_file": groups[r["scenario_id"]].append(i)
    gl = list(groups.values()); rng.shuffle(gl)
    for g in gl:
        if counts[REV] <= per_class * slack and counts[IRR] <= per_class * slack: break
        for i in g: drop.add(i); counts[recs[i]["label"]] -= 1
    keep = [i for i in range(len(recs)) if i not in drop]
    return [recs[i] for i in keep], [objs[i] for i in keep]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manual", type=int, default=100, help="pairs for the manual labelling sheet")
    ap.add_argument("--out", default="data/sandbox.jsonl")
    a = ap.parse_args()
    recs, objs, rng = generate(a.per_class, a.seed)
    recs, objs = balance(recs, objs, a.per_class, rng)
    # BFS labels for EVERY record at horizons 1, 2, 3 (used for the horizon ablation and to verify the designed labels)
    agree = {1: 0, 2: 0, 3: 0}
    for r, (sb, tool, args) in zip(recs, objs):
        for h in (1, 2, 3):
            r[f"label_h{h}"] = sb.bfs_label(tool, args, h)
            agree[h] += r[f"label_h{h}"] == r["label"]
    write_jsonl(a.out, recs)
    c = collections.Counter(r["label"] for r in recs)
    mp = sum(r["is_minimal"] for r in recs) / len(recs)
    print(f"{len(recs)} records | classes {dict(c)} | minimal-pair share {mp:.0%} | "
          f"test share {sum(r['split']=='test' for r in recs)/len(recs):.0%}")
    res = {f"horizon_{h}": agree[h] / len(recs) for h in (1, 2, 3)}
    print("designed label vs BFS label agreement (all records):", res)
    json.dump({"n": len(recs), **res, "class_counts": dict(c), "minimal_pair_share": mp},
              open("results/label_verification.json", "w"), indent=1)

    # manual labelling sheets: same 100 pairs for two annotators, labels hidden
    man = rng.sample(recs, min(a.manual, len(recs)))
    for who in ("annotator1", "annotator2"):
        with open(f"data/manual_{who}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["id", "state_text", "action_text", "your_label"])
            for r in man: w.writerow([r["id"], r["state_text"], r["action_text"], ""])
    print("wrote manual sheets (fill 'your_label' with reversible / partially_reversible / irreversible)")
