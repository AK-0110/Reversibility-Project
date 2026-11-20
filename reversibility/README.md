# Do open-weight LLM agents represent and act on action reversibility? 

Pipeline for the six steps of the brief; every stage below is scripted and seeded. Results for Qwen2.5-3B and Llama-3.1-8B (4-bit) are committed in `results/` and `figures/`; the report is `paper/main.tex`.

## Layout
| Step | Script | Output |
|---|---|---|
| 1 Data | `make_sandbox_data.py` (+ `--extra` supplement), `sciworld_label.py` (+ `--relabel`), `make_variants.py`, (`toolemu_cases.py`, `prepare_bytesized.py`) | `data/*.jsonl`, `results/label_verification*.json`, manual-labelling sheets |
| 2 Hidden states | `extract_hidden.py` | `data/h_<tag>_<file>_{last,mean,pre}.npy`, `..._prompts.jsonl` |
| 3 RQ1 probes | `probes.py` | `results/probe_*.json`, `results/probe_preds_*.csv` (scores, every split), `figures/probe_*.png` |
| 4 RQ2 stated answer | `stated_answer.py` | `results/stated_<tag>.json` |
| 5 RQ3 behaviour | `tasks.py`, `agent_eval.py` | `results/agent_*.json(l)` (full logs) |
| 6 Bonus RQ4/RQ5 | `steering.py`, `uncertainty.py` | `results/steering_*.json`, `uncertainty_*.json` |
| Report | `make_tables.py`, `demo.py`, `ANALYSIS_PLAN.md`, `DATA_CARD.md`, `CONTRIBUTIONS.md`, `paper/` | `results/tables/*.md`, figures, `results/demo_<task>.md` |
| Checks | `test_agent_metrics.py` (RQ3 metrics on mock policies), `agreement.py` | |

`bash run_all.sh` runs everything for one model on a GPU box (`MODEL=... TAG=... bash run_all.sh`; `QUICK=1` plumbing test, `PILOT=1` agent pilot, `LOAD_4BIT=1` for 8B on a T4). On Colab use `colab_run.ipynb`. The data files are frozen inputs; the script never regenerates them.

## Setup
`pip install -r requirements.txt` (Java 11+ only for ScienceWorld labelling). Models used: `Qwen/Qwen2.5-3B-Instruct` (fp16 on a T4; 7B does not fit) and `meta-llama/Llama-3.1-8B-Instruct` in 4-bit (gated: request access first). Load one model at a time. `run_all.sh` writes `results/requirements.lock_<tag>.txt`.

## Reproduce
The data is frozen (`data/*.jsonl`); do not regenerate it (the hand-label sheets are tied to its ids).
1. Model runs (GPU): `MODEL=Qwen/Qwen2.5-3B-Instruct TAG=qwen3b bash run_all.sh` and `LOAD_4BIT=1 MODEL=meta-llama/Llama-3.1-8B-Instruct TAG=llama8b bash run_all.sh` (or `colab_run.ipynb`). Steering: notebook cell 8 (layer = binary best layer in `results/probe_<tag>_last.json`).
2. Tables and figures (CPU, from the committed results): `python make_tables.py --model-tag qwen3b` then `--model-tag llama8b`.
3. Re-score stated answers without a model: `python stated_answer.py --model x --data data/sandbox.jsonl data/sandbox_extra.jsonl --tag <tag> --test-only --rescore --probe-preds results/probe_preds_<tag>_last_binary.csv results/probe_preds_<tag>_last_three.csv`.
4. Hand-label agreement: `python agreement.py data/manual_annotator1.csv data/manual_annotator2.csv data/sandbox.jsonl`.

| Paper item | File | Made by |
|---|---|---|
| Fig. 1 probe accuracy vs layer | `figures/probe_<tag>_last_binary.png` | `probes.py` (in `run_all.sh`) |
| Fig. 2 stated vs probed vs behaviour | `figures/stated_probed_behaviour_<tag>.png` | `make_tables.py` |
| Fig. 3 blindness gap with CIs | `figures/blindness_gap.png` (all models in `results/`) | `make_tables.py` |
| Table I probes and controls | `results/tables/probe_summary.md` | `make_tables.py` |
| Table II stated answers and behaviour | `results/tables/stated_probe_behaviour_<tag>.md` | `make_tables.py` |
| Table III ablations | `results/tables/ablations.md` | `make_tables.py` |
| Steering curve (bonus) | `figures/steering_<tag>.png`, `results/tables/steering.md` | `steering.py`, `make_tables.py` |
| Hand-label kappa | `results/label_agreement.json` | `agreement.py` |

## Design decisions worth stating in the report
- Labels are from the design plus BFS verification (sandbox) and bounded search (ScienceWorld). Sandbox: designed vs BFS agreement 100% at horizons 2-3, 88% at horizon 1 (edit with backup + history needs two undo steps).
- Minimal pairs: 97% of sandbox rows (same action, only the relevant setting differs). The action-only lexical baseline is 0.5 on them by construction, but a state-only bag-of-words reaches 0.735 (trained and tested on minimal pairs only), so probes are also compared with state baselines, a pre-action hidden state, a permuted-setting control, leave-one-verb-out, and reworded test rows.
- Splits are grouped by scenario (sandbox) or task (ScienceWorld); layer chosen on validation, never on test.
- Behaviour tasks keep every non-tested setting "safe" in both versions, so exactly one prompt line differs between versions.
- `K` is a forced A/B question asked in both orders; `A` is the share of pairs where caution(irr) > caution(rev). The brief's Gap = K - A is ~0.5 for a model that knows nothing, so the headline is the chance-corrected Gap* = K* - A* (see `ANALYSIS_PLAN.md`, Amendment 1, and `test_agent_metrics.py`). CIs over pairs and over templates.
- Parse failures are counted and reported, never dropped.
- Probe layer k = `hidden_states[k]` = output of decoder block k-1 (0 = embeddings); `Steer` handles the offset.

## Limitations (also in `DATA_CARD.md`)
- ScienceWorld "irreversible" = no restoring sequence found within the search budget (upper bound). "Restored" = equal to the world after the same number of idle ticks, ignoring temperature and treating living things atomically.
- Partial labels are rarer in ScienceWorld than in the sandbox; report class counts per environment.
- RQ5 needs hand labels: ByteSized32 rows do not include the game simulator (see `prepare_bytesized.py`).
- Checks before the real run: RQ3 metrics on mock policies (`test_agent_metrics.py`), every probe mode and `make_tables.py` on synthetic hidden states, hidden-state extraction with Qwen2.5-0.5B on CPU, and the `QUICK=1` run on Colab. Those numbers are meaningless; real results come only from the GPU run.
