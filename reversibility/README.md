# Do open-weight LLM agents represent and act on action reversibility? (ADNN Project Option 1)

Pipeline for the six steps of the brief. Four people, ~2 months in the brief; every stage below is scripted and seeded.

## Layout
| Step | Script | Output |
|---|---|---|
| 1 Data | `make_sandbox_data.py`, `sciworld_label.py`, (`toolemu_cases.py`, `prepare_bytesized.py`) | `data/*.jsonl`, `results/label_verification.json`, manual-labelling sheets |
| 2 Hidden states | `extract_hidden.py` | `data/h_<tag>_<env>_{last,mean}.npy` |
| 3 RQ1 probes | `probes.py` | `results/probe_*.json`, `figures/probe_*.png` |
| 4 RQ2 stated answer | `stated_answer.py` | `results/stated_<tag>.json` |
| 5 RQ3 behaviour | `tasks.py`, `agent_eval.py` | `results/agent_*.json(l)` (full logs) |
| 6 Bonus RQ4/RQ5 | `steering.py`, `uncertainty.py` | `results/steering_*.json`, `uncertainty_*.json` |
| Report | `make_tables.py`, `ANALYSIS_PLAN.md`, `DATA_CARD.md`, `CONTRIBUTIONS.md` | `results/tables/*.md`, figures |

`bash run_all.sh` runs everything for one model (`MODEL=... TAG=... bash run_all.sh`).

## Setup
`pip install -r requirements.txt` (needs Java 11+ for ScienceWorld). Models: `Qwen/Qwen2.5-7B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct`
(request Llama access first). Load one model at a time. After the first successful run keep `requirements.lock.txt` (the script writes it).

## Suggested order
1. Commit `ANALYSIS_PLAN.md` before touching real results.
2. `python make_sandbox_data.py` (~4 min). Two people label `data/manual_annotator{1,2}.csv`, then `python agreement.py ...` (report kappa).
3. `python sciworld_label.py --per-class 250 --workers 8` (hours; resumable, re-run to add more; use `nohup`). Total dataset must reach 2,000+ pairs.
4. Run `run_all.sh` for Qwen, then Llama. Read the best layer from `results/probe_<tag>_last.json` (`tasks.three.best_layer`) and rerun with `BEST_LAYER=<n>` for steering.
5. `python make_tables.py --model-tag <tag>`.

## Design decisions worth stating in the report
- Labels are from the design plus BFS verification (sandbox) and bounded search (ScienceWorld). Sandbox: designed vs BFS agreement 100% at horizons 2-3, 88% at horizon 1 (edit with backup + history needs two undo steps).
- Minimal pairs: 97% of sandbox rows (same action, only the relevant setting differs). Lexical baseline is 0.5 on them by construction.
- Splits are grouped by scenario (sandbox) or task (ScienceWorld); layer chosen on validation, never on test.
- Behaviour tasks keep every non-tested setting "safe" in both versions, so exactly one prompt line differs between versions.
- `K` is measured with a forced A/B question about the planned action; `A` is the share of pairs where caution(irr) > caution(rev). Gap = K - A with bootstrap CI.
- Parse failures are counted and reported, never dropped.
- Probe layer k = `hidden_states[k]` = output of decoder block k-1 (0 = embeddings); `Steer` handles the offset.

## Limitations (also in `DATA_CARD.md`)
- ScienceWorld "irreversible" = no restoring sequence found within the search budget (upper bound). "Restored" = equal to the world after the same number of idle ticks, ignoring temperature and treating living things atomically.
- Partial labels are rarer in ScienceWorld than in the sandbox; report class counts per environment.
- RQ5 needs hand labels: ByteSized32 rows do not include the game simulator (see `prepare_bytesized.py`).
- GPU-dependent scripts (`extract_hidden`, `stated_answer`, `agent_eval`, `steering`, `uncertainty`) were not executed during development; CPU parts were tested on synthetic data and a mock policy.

## Still on you (cannot be scripted)
Verify every cited paper and search arXiv for prior work on this exact experiment; hand-label 100 pairs with two annotators; write the 6-8 page report (use `results/tables`, `figures/`) and the contribution log.
