# Reversibility project: working rules

Brief: `~/Downloads/ADNN_Project_Option_1_Irreversibility.docx`. Pipeline lives in `reversibility/`.

## How to work (what makes this report good)
1. **Don't trust a good number.** Sanity-check anything surprising before using it (e.g. the negative "selectivity" values in the toy run: bug or tiny-model artefact?).
2. **Minimal pairs are the headline, but the bar is the state baseline.** A state-only bag-of-words gets 0.745 on minimal pairs (action-only gets 0.5 by construction). Claim action-specific reversibility only if probe(last) beats the state BoW AND probe(pre), beats the permuted-setting control, survives leave-one-verb-out and the reworded test rows.
3. **Headline gap is Gap* = K* - A*** (chance-corrected; the brief's K - A is ~0.5 for a know-nothing model). Report both, CIs over pairs and templates (`test_agent_metrics.py` checks the metrics).
4. **Agent tasks must tempt the model.** If it never takes the irreversible action there is nothing to measure; check this before the real run.
5. **Report failures openly.** Count parse failures, list every experiment run, state what did not work. Never invent or hand-fill results.
6. **Reproducible.** One command per table, exact library versions pinned (`results/requirements.lock_<tag>.txt` from the Colab run), fixed seeds.
7. **Demo:** one reversible and one irreversible task, narrated (`demo.py` on the saved agent logs).
8. **Human labels must come from humans.** Never fill the annotator sheets for the user and never write "human-labelled" about labels a human did not produce. If AI-labelled, say so in the paper. (Done 2026-11-17: both sheets labelled by the team with `label_tool.py`, kappa 0.85 / binary 0.90, in `results/label_agreement.json`, DATA_CARD, paper.)
9. **Secrets:** the user creates the Hugging Face token and the Colab `HF_TOKEN` secret themselves. Never ask for, read or paste it; the second Claude may only check the secret exists.

## Facts and decisions
- Model results come only from the GPU run (Colab). Local CPU runs are plumbing tests; their numbers are meaningless.
- Do not regenerate `data/sandbox.jsonl` or the `manual_annotator*.csv` sheets: the sheets are tied to its ids.
- Data is frozen for the GPU run: sandbox 1,664 + `sandbox_extra` 350 + `sciworld_v1` 259 = 2,273 rows, plus reworded test copies (`sandbox_terse`, `sandbox_flip`). Labels can still be fixed later (hidden states do not depend on labels).
- ScienceWorld non-reversible labels were mostly search cut-offs (budget 60 < ~138-node shortlist). Re-search with `sciworld_label.py --relabel data/sciworld_v1.jsonl --budget 200 --workers 1 --out data/sciworld_v1_relabel.jsonl`; it was killed for low memory twice on 2026-11-16/17 (7.6 GB laptop, ~1.5-3.8 GB free): run it alone, 1 worker, close the browser. Then `PROBES_ONLY=1 SW=data/sciworld_v1_relabel.jsonl` re-runs probes without a GPU. Failed re-searches and bare menu-number actions become "unverified" (excluded).
- RQ1 probes and RQ2 use the sandbox only (2,014 rows); ScienceWorld is only the transfer target plus its own stated-answer table (Amendment 2). Do not pool it back in.
- `ANALYSIS_PLAN.md` Amendments 1 and 2 (2026-11-16) fix every metric change and are pushed. Later changes = deviations to list in the report.
- Local Python: use `reversibility/.venv/Scripts/python.exe` (has statsmodels; plain `py -3` does not). Before `make_tables.py`, copy every model's `results/agent_*.json` (Colab Qwen + Kaggle Llama) into one `results/` so `blindness_gap.png` shows both.
- After `git reset`, re-stage `git rm --cached` removals and check `git ls-tree origin/main` after pushing (the ByteSized untrack once silently missed).
- Shell heredocs mangle backslashes on this machine: write scripts with the Write tool, not `python - <<EOF`.
- Models: Qwen2.5-3B-Instruct fp16 (7B bf16 OOMs on a T4); Llama-3.1-8B-Instruct in 4-bit (access granted 2026-11-17 to the Instruct repo; `HF_TOKEN` secret set). Fallback only if Llama fails: microsoft/Phi-3.5-mini-instruct (needs instructor OK). Both are deviations caused by the free T4, to state in the paper. Instructor email draft: `instructor_email_draft.txt` (user sends it).
- Paper is `reversibility/paper/main.tex` (IEEEtran). `refs.bib`: all 10 entries verified by search 2026-11-17 (turner pages from the brief; yin has no venue, kept as arXiv).
- Git: existing author, Conventional Commits written like a human, no Claude attribution, ponytail review before each commit. History was reworded and force-pushed once with the user's permission.

## Status 2026-11-20
- Both models done and committed: Qwen2.5-3B (Colab, results_qwen3b.zip of 2026-11-17) and Llama-3.1-8B 4-bit (Kaggle 2xT4, 9 h). Tables/figures rebuilt locally with both.
- Hidden states: Qwen's are gone (Colab); Llama's still sit in the Kaggle Version #1 output (2.46 GB), not downloaded. Steering needs Qwen sandbox hidden states re-extracted (Kaggle, Chrome agent).
- Key findings to write up: probes ~0.97-0.98 (beat pre ~0.67 and state BoW 0.705; minimal pairs 0.98 vs 0.735); leave-one-verb-out ~0.50 (per-verb rules, no shared direction); terse rewording drops (three-class ~0.57-0.59); transfer to ScienceWorld fails (~0.44). Stated: Qwen binary 0.70, Llama says PARTIAL on every item (0.50). Behaviour: A* ~0 for both, careful prompt does not help.
- Paper drafted 2026-11-20 (~7 pages, all numbers from results/, 2 \todo left: names, steering). Next: Chrome agent: Qwen steering on Kaggle + citation/licence lookups; user: contributions, email, slides, repo visibility after the Kaggle clone.

## Plan (updated 2026-11-17, 10-15 hours to submission; goal: finish ALL tasks including the bonus)
Status when written: Qwen-3B STEP 3 running on Colab (driven by a second Claude with browser access, reports every ~25 min). Hand labels done. ScienceWorld re-label not done.
1. **Now, in parallel with STEP 3 (no GPU needed):** instructor email sent (user); contribution log and presentation outline; search arXiv for newer work on this exact experiment; check the paper compiles (Overleaf); ToolEmu cases and ByteSized32 labelling prep for RQ5.
2. **When STEP 3 finishes:** save `results_qwen3b.zip` first, then the pip-freeze; paste results to Claude; read the numbers with rule 1; fill the paper only from them; `make_tables.py` figures; `demo.py`.
3. **Llama-3.1-8B (4-bit):** same notebook cell (pilot, full run, zip). Needs STEP 3 to finish first (one cell at a time, never interrupt a running cell).
4. **Bonus RQ4 steering:** after Llama if time allows (~2-3 h on the T4), at the binary best layer from the Qwen probes; 6 strengths incl. negative + 3 random directions; report success next to caution. A clean negative result with the controls also counts.
5. **Bonus RQ5 (uncertainty, ByteSized32):** only after the compulsory items; needs the ByteSized data, our own inference code (`uncertainty.py`), and two people's labels (`data/bytesized_annotator*.csv`). Decide go/no-go once Llama is running, based on remaining hours.
6. **ScienceWorld re-label** whenever the laptop is idle (1 worker); then re-run transfer probes with `PROBES_ONLY=1`.
7. **Last 2 hours:** ponytail review, final paper read, commit, push, submit. Do not make changes that cannot be checked.
Cut order if time runs short: RQ5, then RQ4, then Llama results (report one family as a limitation), then ScienceWorld re-label. Never cut: Qwen probes + stated answers + blindness gap, hand-label agreement, paper, presentation and demo, contribution log.

## Open tasks (rewritten 2026-11-20; everything else is done and committed)
Done: both model runs, CIs, careful-prompt gap, both gaps in the figure, full paper draft (~7 pages, all numbers from results/), related work + verified citations (no published paper runs this experiment), README reproduce map, DATA_CARD/REPORT fixes, licences.
User:
- [ ] `CONTRIBUTIONS.md` (one paragraph per person; now a 39-word stub).
- [ ] Names and student IDs in `paper/main.tex` (the `\todo` in the author line).
- [ ] Overleaf compile check (upload `paper/` and `figures/`, keep the folder layout); report errors and the page count.
- [ ] Make the GitHub repo private (Kaggle clone is done) and give the grader access.
- [ ] Send the instructor email once Claude updates the draft.
- [ ] Slides (15 min), demo recording, one timed rehearsal.
- [ ] Check the course AI-use policy (CLAUDE.md is in the repo).
- [ ] Optional: two people spot-check ~40 ScienceWorld labels (else it stays a listed deviation).
Claude:
- [ ] Update `instructor_email_draft.txt` (Llama granted, Phi not needed, Kaggle/Colab, no prior publication found).
- [ ] Steering zip from Kaggle: check it, CIs over pairs, steering figure, fill the paper's RQ4 section (or write "not run").
- [ ] Presentation outline + demo plan; run `demo.py` on the saved logs.
- [ ] Pin `requirements.txt` from `results/requirements.lock_*.txt`.
- [ ] Final read of the paper against the tables, then commit and push.
Dropped: RQ5 (needs 400 rows labelled by two people), ScienceWorld re-label (listed as a deviation).

## Compute facts
- Free Colab T4 (about 14.5 GB usable) OOMs on Qwen2.5-7B bf16 (tested). Use Qwen2.5-3B-Instruct; Llama 8B needs `LOAD_4BIT=1`.
- T4 has no native bf16: `llm.py` uses fp16 before Ampere; `BF16=1` forces bf16 if fp16 output is garbage.
- Runner `run_all.sh` (modes QUICK=1 / PILOT=1 / PROBES_ONLY=1); notebook `colab_run.ipynb` unzips `reversibility-4.zip` into /content/run4. Interrupting a Colab cell restarts the session and deletes /content: never interrupt.
- The first Qwen-3B Colab run (old code: repetition penalty on, no pre position, biased gap) is a pilot only.
- Laptop (Windows, 7.6 GB RAM): sleep disabled while plugged in during the run; background jobs get killed when memory is low.
