"""Second testbed: ScienceWorld (physical actions). Labels are computed AUTOMATICALLY (Section 4.2):
save the state (by history), apply the action, then search action sequences of length <= 3 for the saved state.

ScienceWorld has no save/restore function, so we rebuild a state by reset + replaying the same task variation's
action history (slow, ~0.25 s per replayed step). To keep the search tractable we (a) sample states along the gold
path, (b) search only a ranked shortlist of candidate follow-up actions (inverse verbs, same objects), and (c) cap the
number of search nodes. A pair is labelled irreversible when no restoring sequence is found within that budget, so
irreversible labels are an UPPER BOUND on irreversibility. Report this limitation.

State comparison (object tree, uuids and temperature ignored; living things tracked atomically because they grow on their
own; the target state is the world after the same number of idle "wait1" ticks, because the simulation also evolves
on its own):
  full signature      = names, types, nesting, isDeleted and every non-temperature property -> exact restoration
  structure signature = names, types, nesting, isDeleted                                       -> approximate restoration
Labels: exact within h steps -> reversible; only structure restored -> partially_reversible; else irreversible.
Stored for h = 1, 2, 3 (label_h1..3); `label`/`horizon` = h 3.

python sciworld_label.py --per-class 250 --workers 4 --out data/sciworld.jsonl      (resumable: re-run to add more)"""
import argparse, collections, hashlib, json, os, random, re
from multiprocessing import Pool
from utils import read_jsonl, write_jsonl

REV, PART, IRR = "reversible", "partially_reversible", "irreversible"
SKIP_VERBS = {"look", "examine", "read", "wait", "wait1", "focus", "eat", "teleport", "inventory", "help",
              "connect", "disconnect"}   # connect/disconnect are electrical-circuit commands, mostly no-ops here
INVERSE = {"open": "close", "close": "open", "activate": "deactivate", "deactivate": "activate",
           "pick": "put", "put": "pick", "go": "go", "move": "move", "pour": "pour", "wear": "remove",
           "connect": "disconnect", "disconnect": "connect"}
STOP = {"the", "to", "in", "a", "an", "of", "on", "and", "with", "containing", "inventory"}
BRANCH = {1: 6, 2: 4, 3: 3}
_ENV = {}


def kids(n):
    c = n.get("contents") or {}
    return list(c.values()) if isinstance(c, dict) else list(c)


def is_life(n):
    p = n.get("propLife")
    return isinstance(p, dict) and bool(p.get("lifeformType"))


def canon(n, props):
    d = {"name": n["name"], "type": n["type"], "del": n["isDeleted"]}
    if is_life(n):                       # living things grow/age on their own every tick: track them atomically
        return d
    if props:
        for k, v in n.items():
            if k.startswith("prop") and v is not None:
                d[k] = {kk: vv for kk, vv in v.items() if kk != "temperatureC"} if isinstance(v, dict) else v
    d["c"] = sorted(json.dumps(canon(c, props), sort_keys=True) for c in kids(n))
    return d


class Labeler:
    def __init__(self, task, var):
        from scienceworld import ScienceWorldEnv
        self.task, self.var = task, var
        self.env = ScienceWorldEnv("", envStepLimit=1000)
        self.env.load(task, var, "", True); self.env.reset()
        self.gold = list(self.env.get_gold_action_sequence()); self.desc = self.env.get_task_description()
        self.nodes = 0

    def replay(self, hist):
        self.env.load(self.task, self.var, "", False); self.env.reset()
        for a in hist: self.env.step(a)

    def sigs(self):
        t = self.env.getObjectTree()
        h = lambda p: hashlib.md5(json.dumps(canon(t, p), sort_keys=True).encode()).hexdigest()
        return h(True), h(False)

    def state_text(self):
        return f"Task: {self.desc}\nObservation:\n{self.env.look()}\nInventory:\n{self.env.inventory()}"

    def valid(self):
        return self.env.get_valid_action_object_combinations()

    def origin_words(self, action):
        """Names of the containers that currently hold the objects mentioned in the action (to put them back)."""
        words, a = set(), action.lower()
        def walk(n, parent):
            if parent and n["name"] and n["name"].lower() in a and parent["name"] not in ("universe", ""):
                words.add(parent["name"].lower())
            for c in kids(n): walk(c, n)
        walk(self.env.getObjectTree(), None)
        return words

    def candidates(self, last, prev_room, k, rng):
        toks = lambda s: {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP}
        lt, lv = toks(last), last.split()[0]
        scored = []
        for v in self.valid():
            vv = v.split()[0]
            if vv in SKIP_VERBS: continue
            sc = len(lt & toks(v)) + (3 if INVERSE.get(lv) == vv else 0)
            if lv == "go" and vv == "go" and prev_room and prev_room in v: sc += 5
            if vv in ("move", "put") and any(w in v for w in self.origin): sc += 4
            scored.append((sc + rng.random() * 0.01, v))
        return [v for _, v in sorted(scored, reverse=True)[:k]]

    def search(self, base, action, h, budget, rng, prev_room):
        """DFS to depth h from the post-action state; returns (exact_found, structure_found, exhausted)."""
        self.replay(base + [action]); res = {"exact": False, "struct": False, "exhausted": True}

        def rec(suffix, depth):
            full, st = self.sigs()
            full0, st0 = self.ref[depth + 1]          # world after the same number of idle ticks
            if full == full0: res["exact"] = True; return True
            if st == st0: res["struct"] = True
            if depth == h: return False
            last = suffix[-1] if suffix else action
            cands = self.candidates(last, prev_room, BRANCH[depth + 1], rng)
            for i, c in enumerate(cands):
                if self.nodes >= budget: res["exhausted"] = False; return False
                self.nodes += 1
                obs = self.env.step(c)[0]
                if "No known action matches" in obs: continue
                if rec(suffix + [c], depth + 1): return True
                if i < len(cands) - 1 or depth > 0:
                    self.replay(base + [action] + suffix)
            return False
        rec([], 0); return res

    def label_pair(self, base, action, budget, rng):
        self.replay(base); text = self.state_text()
        m = re.search(r"called the ([a-z ]+)\.", self.env.look()); prev_room = m.group(1) if m else ""
        self.origin = self.origin_words(action)
        # Reference states: the world after n idle ticks (wait1). ScienceWorld evolves by itself every step
        # (plants grow, fountains run), so "restored" means: equal to the world had we only waited the same time.
        self.ref = {}
        for n in range(1, 5):
            self.env.step("wait1"); self.ref[n] = self.sigs()
        self.replay(base)
        obs = self.env.step(action)[0]
        if "No known action matches" in obs: return None
        self.nodes = 0; out = {"state_text": text}; found = None; approx = False; hlabels = {}
        full1, st1 = self.sigs()
        if full1 == self.ref[1][0]: return None            # no effect on the world: carries no reversibility signal
        for h in (1, 2, 3):
            if found is None:
                r = self.search(base, action, h, budget, rng, prev_room)
                approx = approx or r["struct"]
                if r["exact"]: found = h
            hlabels[h] = REV if (found is not None and found <= h) else (PART if (approx or st1 == self.ref[1][1]) else IRR)
        out.update({f"label_h{h}": hlabels[h] for h in (1, 2, 3)}); out["label"] = hlabels[3]
        out["min_restore_steps"] = found; out["nodes_searched"] = self.nodes
        return out


def job(args):
    task, var, k_prefix, m, budget, seed = args
    rng = random.Random(seed)
    try:
        lab = Labeler(task, var); base = lab.gold[:k_prefix]
        lab.replay(base); valid = [v for v in lab.valid() if v.split()[0] not in SKIP_VERBS]
        cnt = collections.Counter(v.split()[0] for v in valid)
        weights = [1.0 / cnt[v.split()[0]] for v in valid]
        gold_next = [lab.gold[k_prefix]] if k_prefix < len(lab.gold) and lab.gold[k_prefix].split()[0] not in SKIP_VERBS else []
        acts = rng.choices(valid, weights, k=m) + gold_next
        recs = []
        for j, a in enumerate(dict.fromkeys(acts)):
            r = lab.label_pair(base, a, budget, rng)
            if r is None: continue
            recs.append({**r, "environment": "sciworld", "action_text": a, "horizon": 3,
                         "source": f"ScienceWorld:{task}:var{var}:k{k_prefix}", "scenario_id": f"sw-{task}",
                         "is_minimal": False, "pair_id": None, "verb": a.split()[0], "_seed": seed, "_j": j})
        lab.env.close(); return recs
    except Exception as e:                                       # never crash the pool; report instead
        return [{"_error": f"{task} var{var}: {e!r}"}]


def job_stream(tasks_vars, args, rng):
    n = 0
    while True:
        task, vars_ = rng.choice(tasks_vars)
        n += 1
        yield (task, rng.choice(vars_), rng.randint(0, 14), args.per_state, args.budget, rng.randint(0, 10**9))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-jobs", type=int, default=3000); ap.add_argument("--per-state", type=int, default=3)
    ap.add_argument("--budget", type=int, default=60, help="max search nodes per pair")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", default="data/sciworld.jsonl")
    a = ap.parse_args()
    from scienceworld import ScienceWorldEnv
    e = ScienceWorldEnv("", envStepLimit=10); names = a.tasks or sorted(e.get_task_names()); tv = []
    for t in names:
        e.load(t, 0, ""); tv.append((t, e.get_variations_train()))
    e.close()
    test_tasks = {t for i, t in enumerate(names) if i % 5 == 0}
    recs = read_jsonl(a.out) if os.path.exists(a.out) else []
    counts = collections.Counter(r["label"] for r in recs); rng = random.Random(a.seed + len(recs))
    done = lambda: all(counts[l] >= a.per_class for l in (REV, PART, IRR))
    with Pool(a.workers) as pool:
        for i, out in enumerate(pool.imap_unordered(job, (j for _, j in zip(range(a.max_jobs), job_stream(tv, a, rng))))):
            for r in out:
                if "_error" in r: print("WARN", r["_error"]); continue
                r["split"] = "test" if r["scenario_id"][3:] in test_tasks else "train"
                r["id"] = f"sw-{len(recs):05d}"; recs.append(r); counts[r["label"]] += 1
            if i % 10 == 0:
                write_jsonl(a.out, recs); print(f"jobs {i}: {len(recs)} pairs {dict(counts)}", flush=True)
            if done(): pool.terminate(); break
    write_jsonl(a.out, recs); print("final", len(recs), dict(counts))
