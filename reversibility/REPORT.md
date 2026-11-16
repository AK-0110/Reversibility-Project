# Do open-weight LLM agents represent and act on action reversibility?

*Draft skeleton. Sections marked TODO need real numbers from the GPU run; do not fill them with anything else.*

## Abstract
TODO (write last): question, setup (sandbox, Qwen2.5-7B-Instruct), main finding in one sentence, main caveat.

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
- ScienceWorld: not completed, so no second-environment result (see Limitations).

## 3. Methods
- RQ1: linear probes on every layer's hidden state (last action token and mean over action tokens); layer chosen on validation, never on test; control task and lexical baseline (0.5 on minimal pairs by construction); horizon 1/2/3 ablation.
- RQ2: forced-choice question about the action; compare with probe predictions.
- RQ3: tasks where exactly one prompt line differs between reversible and irreversible versions; measure confirming, checking, refusing, safer alternative; also a "be careful" prompt ablation.
- RQ4 (bonus, only if run): steering along the probe direction at the best layer, with random-direction control.
- Seeds, greedy decoding, parse failures counted and reported.

## 4. Results
### RQ1 Probes — TODO (best layer, accuracy per layer, control-task gap, minimal-pairs-only result)
### RQ2 Stated answer vs probe — TODO
### RQ3 Behaviour — TODO (caution rate irreversible vs reversible, knowing-acting gap K - A with bootstrap CI)
### RQ4 Steering — TODO or "not run"

## 5. Discussion
TODO: one clear finding, let the data decide. Say plainly if the result is negative.

## 6. Limitations
- Sandbox only; one model unless Llama is run; simulated, text-only, English.
- ScienceWorld labelling was not finished, so no cross-environment test.
- Sandbox undo is mostly 1–2 steps, so horizon effects are weak.
- partially_reversible is rare outside the edit-with-history case.
- Hand-label agreement TODO.

## 7. Contributions
See CONTRIBUTIONS.md.
