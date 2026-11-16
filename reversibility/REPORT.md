# Do open-weight LLM agents represent and act on action reversibility?

*Draft skeleton. Sections marked TODO need real numbers from the GPU run; do not fill them with anything else.*

## Abstract
TODO (write last): question, setup (sandbox + ScienceWorld, Qwen2.5-3B-Instruct + second family), main finding in one sentence, main caveat.

## 1. Introduction
- Agents run commands, edit files, send messages; some actions cannot be undone.
- Two questions: does the model represent reversibility (RQ1), does it say so when asked (RQ2), does it act on it (RQ3).
- "Irreversibility blindness" = knows when asked, ignores it when acting.
- Related work to verify before citing: Grinsztajn et al. 2021 (arXiv 2106.04480); Hewitt & Liang 2019 (control tasks); ToolEmu (2309.15817); Tang et al. 2025 (2508.13465); Rimsky et al. 2024 (steering). Search arXiv for newer work.

## 2. Data
- Sandbox: files, outbox, wallet; ground-truth labels from design, checked by breadth-first search.
- 1,664 pairs: 607 reversible, 450 partially reversible, 607 irreversible; 97% minimal pairs; 18% test; split grouped by scenario.
- Label check, designed vs search: 0.878 (horizon 1), 1.000 (horizons 2 and 3).
- Human check: TODO two annotators, 100 pairs, Cohen's kappa = ___.
- Supplement: 350 single-variant sandbox pairs (read/move/trash included). ScienceWorld: 259 pairs; most non-reversible labels were search cut-offs (re-search pending), so it is used only for transfer and a separate stated-answer table; probes and stated answers use the sandbox (2,014 rows). Total 2,273.
- Reworded test copies (terse on/off wording; irrelevant setting flipped) for frozen-probe checks.

## 3. Methods
- RQ1: linear probes on every layer's hidden state (last action token, mean over action tokens, token before the action); layer chosen on validation; baselines: action / state / state+action bag-of-words (state-only already 0.745 on minimal pairs); verb control and permuted-setting control; leave-one-verb-out; reworded test rows; transfer; horizon.
- RQ2: forced-choice question about the action; compare with probe predictions.
- RQ3: tasks where exactly one prompt line differs between reversible and irreversible versions; measure confirming, checking, refusing, safer alternative; also a "be careful" prompt ablation. Headline Gap* = K* − A* (chance-corrected; the brief's K − A is ~0.5 for a model that knows nothing).
- RQ4 (bonus, only if run): steering along the probe direction at the best layer, with random-direction control.
- Seeds, greedy decoding, parse failures counted and reported.

## 4. Results
### RQ1 Probes — TODO (best layer, accuracy per layer, control-task gap, minimal-pairs-only result)
### RQ2 Stated answer vs probe — TODO
### RQ3 Behaviour — TODO (caution rate irreversible vs reversible, Gap* with CIs over pairs and templates, brief's K − A with its null reference)
### RQ4 Steering — TODO or "not run"

## 5. Discussion
TODO: one clear finding, let the data decide. Say plainly if the result is negative.

## 6. Limitations
- One model family unless the second is run; simulated, text-only, English.
- ScienceWorld is small (259 pairs) and its "irreversible" label is an upper bound from a bounded search.
- Task prompts push for the shortcut, so behaviour partly measures instruction following.
- Sandbox undo is mostly 1–2 steps, so horizon effects are weak.
- partially_reversible is rare outside the edit-with-history case.
- Hand-label agreement TODO.

## 7. Contributions
See CONTRIBUTIONS.md.
