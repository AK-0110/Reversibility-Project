# Presentation outline (15 min) and demo plan

All numbers are from `paper/main.tex` / `results/tables/`. About one minute per slide; the demo is slides 9-10.

| # | Slide | Content | Figure / source | Time |
|---|---|---|---|---|
| 1 | Title | Do AI agents know which actions cannot be undone? Names, course. | | 0:30 |
| 2 | Why it matters | Agents delete files, send emails, move money. A careful human checks before a permanent action. Question: does the model *know* an action is permanent, *say* it, and *act* on it? | | 1:00 |
| 3 | Three measurements | Encodes (probe on hidden states, RQ1) / says (direct question, RQ2) / does (agent tasks, RQ3), on the same labelled data. "Irreversibility blindness" = knows but acts anyway. | diagram | 1:00 |
| 4 | Data | Simulated sandbox (files, email, payments, settings such as recycle bin). 2,014 sandbox + 259 ScienceWorld pairs; 71% minimal pairs (same action, one setting differs). Labels verified by search; two annotators kappa 0.85. | `DATA_CARD.md` | 1:15 |
| 5 | Models and setup | Qwen2.5-3B (Colab T4, fp16) and Llama-3.1-8B (Kaggle, 4-bit); why not 7B (OOM on a free T4). Pre-registered plan + amendments. | | 0:45 |
| 6 | RQ1 result | Probe at the action token: 0.980 / 0.966 balanced accuracy. Beats state-only bag-of-words (0.705) and the probe that sees the state but not the action (0.68 / 0.67). | Fig. 1 `probe_*_last_binary.png` | 1:30 |
| 7 | RQ1 caveats | Leave one verb out: chance (0.49-0.53). Terse rewording: 0.98 -> 0.80. No transfer to ScienceWorld (0.44). Permuted-setting control 0.90 / 0.83. So: verb-specific rules, not one general concept. | Table III | 1:15 |
| 8 | RQ2 + RQ3 result | Asked: Qwen 0.70, Llama says "partial" to everything. Agent: same caution in both versions (A* = -0.04 / -0.01); careful prompt does not help. Gap* = 0.22 / 0.40; the brief's K-A is ~0.5 even for a know-nothing model, hence the corrected gap. | Fig. 2, Fig. 3 `blindness_gap.png` | 2:00 |
| 9 | Demo 1: blind | `cleanup-0`: same task, recycle bin on vs off. Qwen runs the identical 5 steps in both, permanently deleting in the irreversible one. Point out that the first step in both runs is a rejected call (it passed `paths` instead of `path`); the sandbox counts it as a failed action. | `results/demo_cleanup-0.md` | 1:30 |
| 10 | Demo 2: backwards | `email-4`: Qwen asks the user and checks settings when the email *can* be recalled, and sends straight away when it cannot. | `results/demo_email-4.md` | 1:00 |
| 11 | RQ4 steering (bonus) | Adding the probe direction removes caution in both versions (task success stays ~0.7); subtracting it adds caution but breaks the task. A random direction of the same norm can also remove caution, but with parse failures and lower success. A* never beyond what random directions give. Steering moves caution, not its dependence on reversibility. | Fig. 4 `steering_qwen3b.png` | 1:30 |
| 12 | Limitations + conclusion | Small simulated world; two models, not the named sizes; 6 templates. Takeaway: the knowledge is in the representation, the bottleneck is between representation and action. | | 1:00 |

## Demo plan
- Run (no GPU, a few seconds): `python demo.py results/agent_qwen3b_neutral.jsonl cleanup-0` and `... email-4`. On Windows set `PYTHONIOENCODING=utf-8` first, or just open the written `results/demo_<task>.md`.
- Narrate: (1) read the task, (2) point at the one state line that differs, (3) walk through the steps side by side, (4) say what the metric records (careful: False in both; irreversible action: True only in the irreversible version).
- Backup: screenshots of both markdown files in the slides in case the laptop fails; the recording should show the command and output.
- Likely questions: why not 7B (T4 memory); why Gap* instead of K-A (chance level 0.5); is the probe just reading the verb (leave-one-verb-out, permuted-setting control); does steering prove the model does not use it (no: one layer, one direction, linear only).
