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
- Hidden states (.npy) of both runs are gone (not in the zips). Steering needs Qwen sandbox hidden states re-extracted (Kaggle, Chrome agent).
- Key findings to write up: probes ~0.97-0.98 (beat pre ~0.67 and state BoW 0.705; minimal pairs 0.98 vs 0.735); leave-one-verb-out ~0.50 (per-verb rules, no shared direction); terse rewording drops (three-class ~0.57-0.59); transfer to ScienceWorld fails (~0.44). Stated: Qwen binary 0.70, Llama says PARTIAL on every item (0.50). Behaviour: A* ~0 for both, careful prompt does not help.
- Paper drafted 2026-11-20 (~7 pages, all numbers from results/, 3 \todo left: names, newer related work, steering). Next: Chrome agent: Qwen steering on Kaggle + citation/licence lookups; user: contributions, email, slides, repo visibility after the Kaggle clone.

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

## Open tasks checklist (added 2026-11-17; tick off, do not forget)
User:
- [ ] Send the instructor email (short version: Qwen 3B fp16 + Llama 4-bit, both because of the free T4; draft in `instructor_email_draft.txt`, fill in name/IDs).
- [ ] Write `CONTRIBUTIONS.md`: one paragraph per team member (what they built, experiments run, sections written, bugs debugged). Only the user knows this.
- [ ] Check the paper compiles: upload `paper/main.tex` + `paper/refs.bib` to Overleaf and report any LaTeX errors.
- [ ] Read the Group A papers (Grinsztajn, Alain & Bengio, Hewitt & Liang, ToolEmu) if not done.
- [ ] After Qwen STEP 3: save `results_qwen3b.zip`, paste the results to Claude; then start the Llama cell.
Claude:
- [x] Commit and push hand labels, label_tool, refs.bib, sandbox list-path fix (Llama careful-run crash), tagged comparison outputs, PROBES_ONLY keeps the pip lock, .gitignore fixed + ByteSized sheets untracked (2026-11-17).
From the external critique (2026-11-17; checked items only):
- [ ] (User) Copy hidden states `data/h_<tag>_*` and the results zip to Drive/PC after each GPU run (the notebook zip leaves out the .npy files; steering and the ScienceWorld re-run need them). Download `llama8b_partial_results.zip` from Kaggle now.
- [ ] (User) Decide: the repo is public and ByteSized32 text is in old commits. Make it private until marking (grader gets access) or rewrite history.
- [ ] (User) New cut order: ScienceWorld re-label first, then RQ5, then RQ4. Never cut Llama (shrink N_PER instead). Consider dropping RQ5 now (400 rows need two human labellers).
- [ ] (Team) Block ~4 h to write the paper from real numbers (now ~2 pages, 0 figures/tables, 17 TODOs; brief needs 6-8 pages + the named figures/tables).
- [ ] (Claude) 95% CIs for stated bal-acc, kappa, vs_probe, steering; label baseline spread as sd over 3 splits.
- [ ] (Claude) Careful-prompt run gets a gap (reuse the neutral K); brief's Gap next to Gap* in the figure.
- [ ] (Claude) Paper fixes (done: "two amendments", parse-failure sentence): deviations + every run (OOM attempt, old-code pilot, killed re-labels, Kaggle crash + fix, N_PER changes); selectivity explanation; CI method wording; caution = before the first shortcut call; replace pre-run 0.745; SW horizon counts h1/h2/h3; ScienceWorld spot check done or listed as a deviation.
- [ ] (Claude) arXiv search before the email; update the email (Llama granted, Phi not needed, why not course GPUs). (User) send it.
- [ ] (Claude) README figure/table -> command map; REPORT.md points to the PDF; commit results/, figures/, *_prompts.jsonl, main.pdf (no .npy, every file < 100 MB).
- [ ] (User) SafeAgentBench version read; check wang2022scienceworld; ScienceWorld licence; course AI-use policy.
- [ ] Draft the 15-minute presentation outline and the demo plan (slides need real numbers later).
- [ ] Search arXiv for newer work on reversibility in LLM agents; cite it or rule it out (paper Related Work TODO; the brief requires telling the instructor at once if someone published this exact experiment).
- [ ] Look into RQ5 (ByteSized32): download and check feasibility first; decide go/no-go (needs our inference code and two people's labels).
- [ ] When Qwen results arrive: read numbers with rule 1, fill the paper from real numbers only, run `make_tables.py`, `demo.py`.
- [ ] Pin library versions in `requirements.txt` from the Colab pip freeze; update DATA_CARD/README/REPORT with real results.
- [ ] Optional spare-time job: ScienceWorld re-label (not a bonus item; it only improves the weak ScienceWorld labels). Run it on the user's second desktop if available.
Bonus (up to 2 marks): RQ4 steering with controls (after Llama, if at least ~5 h remain), RQ5 uncertainty (lowest priority).

## Compute facts
- Free Colab T4 (about 14.5 GB usable) OOMs on Qwen2.5-7B bf16 (tested). Use Qwen2.5-3B-Instruct; Llama 8B needs `LOAD_4BIT=1`.
- T4 has no native bf16: `llm.py` uses fp16 before Ampere; `BF16=1` forces bf16 if fp16 output is garbage.
- Runner `run_all.sh` (modes QUICK=1 / PILOT=1 / PROBES_ONLY=1); notebook `colab_run.ipynb` unzips `reversibility-4.zip` into /content/run4. Interrupting a Colab cell restarts the session and deletes /content: never interrupt.
- The first Qwen-3B Colab run (old code: repetition penalty on, no pre position, biased gap) is a pilot only.
- Laptop (Windows, 7.6 GB RAM): sleep disabled while plugged in during the run; background jobs get killed when memory is low.
