# Data card: reversibility dataset

**Task.** Each row is a (state, action) pair labelled by how well the action can be undone within 3 further actions.
**Labels.** reversible / partially_reversible / irreversible (definitions: Section 4.2 of the project brief). Binary (reversible vs irreversible) is the primary task; see "Partial" below.
**Fields.** id, environment, state_text, action_text, label, horizon, source, scenario_id, pair_id, is_minimal, verb, split, label_h1..h3 (+ min_restore_steps, nodes_searched for ScienceWorld).

## Files (frozen inputs of the GPU run)
| file | rows | reversible | partial | irreversible | train / test | notes |
|---|---|---|---|---|---|---|
| `data/sandbox.jsonl` | 1,664 | 607 | 450 | 607 | 1,359 / 305 | 97% minimal pairs; 4 verbs; 663 scenarios |
| `data/sandbox_extra.jsonl` | 350 | 244 | 18 | 88 | 266 / 84 | single-variant supplement, new ids; 7 verbs incl. read/move/trash |
| `data/sciworld_v1.jsonl` | 259 | 203 | 17 | 39 | 213 / 46 | 14 verbs; 30 tasks; split by task |
| **total** | **2,273** | 1,054 | 485 | 734 | | 71% minimal pairs overall |

Reworded copies of the sandbox test rows (same ids and labels, for frozen-probe checks; `make_variants.py`):
`data/sandbox_terse.jsonl` (389 rows; settings say only "on/off") and `data/sandbox_flip.jsonl` (359 rows; only a setting irrelevant to the action is flipped).

## Construction
- *sandbox* (`make_sandbox_data.py`): simulated files / outbox / wallet. Ground truth comes from the design (`Sandbox.designed_label`) and is checked by breadth-first search (`bfs_label`): agreement at horizon 1/2/3 = 0.878 / 1.000 / 1.000 (`results/label_verification.json`); supplement 0.983 / 1.000 / 1.000 (`results/label_verification_sandbox_extra.json`). Minimal pairs: same action, only the relevant setting differs. The search runs the same simulator, so it checks that the code agrees with itself, not what the labels mean; the human labels do that.
- *sandbox supplement* (`make_sandbox_data.py --extra 50 --seed 1`): one variant per scenario, so verb and label are not balanced within a scenario. It exists so that "all rows" vs "minimal pairs only" is a real contrast and the brief's verbs (read, move, trash) appear.
- *sciworld* (`sciworld_label.py`): states sampled along gold paths; labels from a bounded search over a ranked shortlist of follow-up actions; "restored" = equal to the world after the same number of idle ticks (the simulation evolves by itself); living things tracked atomically; temperature ignored. Frozen at 259 rows on 2026-11-16 (labelling on one laptop is ~100 pairs/hour).

## Known label issues (report these)
- **ScienceWorld non-reversible labels are mostly "search stopped".** With the original 60-node budget, 37 of 39 irreversible and all 17 partial rows hit the budget before the depth-3 shortlist (≈138 nodes) was searched. Spot checks show clear errors (e.g. `open cupboard` → partial, `pick up red wire` → irreversible). Planned fix: re-search these 56 rows with budget 200, which covers the whole shortlist (`sciworld_label.py --relabel data/sciworld_v1.jsonl --budget 200 --workers 2 --out data/sciworld_v1_relabel.jsonl`; same rows in the same order, the old label is kept in `label_budget60`; a row whose re-search fails, and the two rows whose action is a bare menu number ("0", "1"), become "unverified" and are excluded), then `PROBES_ONLY=1 SW=data/sciworld_v1_relabel.jsonl MODEL=... TAG=... bash run_all.sh` (no GPU needed, labels do not change hidden states). A first attempt on the laptop was stopped for low memory. Because of these labels, RQ1 probes and RQ2 use the sandbox only; ScienceWorld is used for transfer and a separate stated-answer table (Amendment 2). Even after the re-search, "irreversible" means "no restoring sequence among the shortlisted follow-ups", an upper bound.
- **Partial (sandbox).** Partial = edit with history and no backup: revert restores the content but changes the file's mtime, which the state text never shows. Annotators and models will often call these rows reversible, so binary is the primary metric.
- **Lexical shortcuts.** The state text names reversibility ("removed permanently", "cannot be recalled", "are final"). A bag-of-words over the state alone reaches 0.735 balanced accuracy (binary) when trained and tested on minimal pairs only (0.705 on the minimal pairs of the full test split; mean over 3 splits); the action-only baseline is 0.500 by construction. Probes are compared with both, with a pre-action hidden state, and on the reworded copies.
- **Composition.** edit_file is 74% of sandbox rows and all of sandbox partial; per-verb results are reported.
- **Horizon.** In the sandbox h2 and h3 equal the main label on every row; h1 differs on 203 edit rows (backup + history). Horizon effects come mainly from ScienceWorld.
- **Split.** No scenario spans splits and there are no duplicate rows, but the sandbox label is a deterministic function of a few settings, so the test split mostly tests interpolation within seen setting combinations.

## Quality control
- Two annotators, 100 random sandbox pairs, labels hidden (`ANNOTATION_GUIDE.md`, labelled with `label_tool.py`; `results/label_agreement.json`): Cohen's kappa annotator 1 vs 2 = 0.847 (three classes), 0.903 (irreversible vs not); the two annotators gave the same three-class label on 90 of 100 rows. Each annotator vs the automatic label: 0.862 / 0.985 (three classes), 0.903 / 1.000 (binary). The automatic labels come from the simulator's rules and a search over the same simulator, so these numbers show the labels match what people judge from the state text; they do not test the simulator itself.
- ScienceWorld spot check: not done (planned in Amendment 2; listed as a deviation in the paper). ScienceWorld labels are unchecked by people.

## Licences
Sandbox data: yours. ScienceWorld-derived rows: ScienceWorld is Apache-2.0 (allenai/ScienceWorld LICENSE, checked 2026-11-20). ToolEmu: Apache-2.0. ByteSized32: no licence file, do not redistribute.
