# Reversibility project: working rules

Brief: `~/Downloads/ADNN_Project_Option_1_Irreversibility.docx`. Pipeline lives in `reversibility/`.

## How to work (what makes this report good)
1. **Don't trust a good number.** Sanity-check anything surprising before using it (e.g. the negative "selectivity" values in the toy run: bug or tiny-model artefact?).
2. **Minimal pairs are the headline, but the bar is the state baseline.** A state-only bag-of-words gets 0.745 on minimal pairs (action-only gets 0.5 by construction). Claim action-specific reversibility only if probe(last) beats the state BoW AND probe(pre), beats the permuted-setting control, survives leave-one-verb-out and the reworded test rows.
3. **Headline gap is Gap* = K* - A*** (chance-corrected; the brief's K - A is ~0.5 for a know-nothing model). Report both, CIs over pairs and templates (`test_agent_metrics.py` checks the metrics).
4. **Agent tasks must tempt the model.** If it never takes the irreversible action there is nothing to measure; check this before the real run.
5. **Report failures openly.** Count parse failures, list every experiment run, state what did not work. Never invent or hand-fill results.
6. **Reproducible.** One command per table, exact library versions pinned (`results/requirements.colab.txt` from the Colab run), fixed seeds.
7. **Demo:** one reversible and one irreversible task, narrated.

## Facts and decisions
- Model results come only from the GPU run (Colab). Local CPU runs are plumbing tests; their numbers are meaningless.
- Do not regenerate `data/sandbox.jsonl` or the `manual_annotator*.csv` sheets: the sheets are tied to its ids and hand-labelling depends on them.
- Data is frozen for the GPU run: sandbox 1,664 + `sandbox_extra` 350 + `sciworld_v1` 259 = 2,273 rows, plus reworded test copies (`sandbox_terse`, `sandbox_flip`). Labels can still be fixed later (hidden states do not depend on labels).
- ScienceWorld non-reversible labels were mostly search cut-offs (budget 60 < ~138-node shortlist); re-search with `sciworld_label.py --relabel data/sciworld_v1.jsonl --budget 200 --workers 2 --out data/sciworld_v1_relabel.jsonl` (first attempt killed for low memory on 2026-11-16: run it alone, 1-2 workers, nothing else open), then `PROBES_ONLY=1 SW=data/sciworld_v1_relabel.jsonl` re-runs probes without a GPU. Failed re-searches and bare menu-number actions become "unverified" (excluded).
- RQ1 probes and RQ2 use the sandbox only (2,014 rows); ScienceWorld is only the transfer target plus its own stated-answer table (Amendment 2). Do not pool it back in.
- `ANALYSIS_PLAN.md` Amendment 1 (2026-11-16) fixes every metric change; it must be committed before the real run. Later changes = deviations.
- Shell heredocs mangle backslashes on this machine: write scripts with the Write tool.
- Brief requires two model families. Llama-3.1-8B access is requested by the user on Hugging Face (they must accept Meta's licence themselves). Fallback (needs instructor OK): microsoft/Phi-3.5-mini-instruct (MIT, ungated, 3.8B, fp16 on T4, template tested). Not Mistral-7B-v0.3: 7B in fp16 does not fit a T4 (its template keeps the system prompt, merged into the user turn; checked offline).
- Paper is `reversibility/paper/main.tex` (IEEEtran). `refs.bib` is unverified: check every entry against the paper before submitting.

## Plan and order (agreed 2026-11-16)
Anything that changes data or agent tasks goes BEFORE the real Colab run and before hand labelling starts.
1. User: request Llama-3.1-8B access (HF); decide whether to regenerate the sandbox with read/move/trash verbs.
2. Claude: regenerate sandbox (if yes) + re-verify label agreement; check the 72 agent tasks tempt the model; re-zip to Downloads.
3. In parallel: user runs the real Colab sandbox run; two people hand-label 100 pairs (ANNOTATION_GUIDE.md);
   ScienceWorld labelling keeps running locally; Claude writes README, demo script, figure/table scripts.
4. After the sandbox run: pin versions from results/requirements.colab.txt; fill paper results only from real numbers.
5. Freeze ScienceWorld at a cut-off, then a second shorter Colab pass for its hidden states + transfer ablation.
6. If Llama access arrives, run the same pipeline on it; else say so in the report.
7. User: agreement.py kappa, verify citations, contribution log, record demo.
8. Last: ponytail review, commit, push (existing git author, no Claude attribution), submit.
Cut order if time runs short: steering (RQ4), then Llama, then shrink ScienceWorld. Keep sandbox probes, stated answers, blindness gap.

## Compute facts
- Free Colab T4 (about 14.5 GB usable) OOMs on Qwen2.5-7B bf16 (tested 2026-11-16). Use Qwen2.5-3B-Instruct there; 7B needs an L4/A100.
- T4 has no native bf16: llm.py uses fp16 before Ampere; 8B models need LOAD_4BIT=1. Runner: `run_all.sh` (QUICK=1 / PILOT=1); notebook `colab_run.ipynb` unzips reversibility-4.zip into /content/run4.
- The first Qwen-3B Colab run (old code: repetition penalty on, no pre position, biased gap) is a pilot only.
