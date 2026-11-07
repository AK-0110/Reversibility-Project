# Data card: reversibility dataset (fill in the numbers from your run)

**Task.** Each row is a (state, action) pair labelled by how well the action can be undone within 3 further actions.
**Labels.** reversible / partially_reversible / irreversible (definitions: Section 4.2 of the project brief).
**Fields.** id, environment, state_text, action_text, label, horizon, source (+ scenario_id, pair_id, is_minimal, verb, split, label_h1..h3 for ScienceWorld).

## Construction
- *sandbox* (`make_sandbox_data.py`): simulated files / outbox / wallet. Ground truth comes from the design (`Sandbox.designed_label`) and is checked by breadth-first search (`bfs_label`): agreement at horizon 1/2/3 = ___ / ___ / ___ (`results/label_verification.json`). Minimal pairs: ___ % of rows (same action, only the relevant setting differs).
- *sciworld* (`sciworld_label.py`): states sampled along gold paths; labels from a bounded search over ranked candidate follow-ups; "restored" = equal to the world after the same number of idle ticks (the simulation evolves by itself); living things tracked atomically; temperature ignored. **Irreversible = no restoring sequence found within the search budget (upper bound).**
- *toolemu* (optional): own labels on the 144 ToolEmu cases' potential risky actions (Apache-2.0 source); two annotators.

## Counts (fill in)
| environment | reversible | partial | irreversible | train | test |
|---|---|---|---|---|---|

## Quality control
- Two annotators, 100 random pairs, labels hidden: Cohen's kappa annotator1–2 = ___; human consensus vs automatic = ___ (`agreement.py`).

## Known limitations
Small simulated sandbox; 3-step horizon; sandbox undo almost always needs ≤ 1–2 steps, so horizon effects are mostly visible in ScienceWorld; ScienceWorld search budget; partially_reversible is rare outside the edit-with-history case; text-only states; English only.

## Licences
Sandbox data: yours. ScienceWorld-derived rows: check the repository LICENSE before release. ToolEmu: Apache-2.0. ByteSized32: no licence file, do not redistribute.
