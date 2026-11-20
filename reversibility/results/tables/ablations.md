## Per verb (test, best layer, mean over splits) and leave-one-verb-out (binary, train on the other verbs)

| tag | task | verb | n test (split 0) | bal-acc | leave-one-verb-out bal-acc [95% CI] |
|---|---|---|---|---|---|
| llama8b_last | binary | delete_file | 44 | 0.785 | 0.534 [0.493, 0.577] |
| llama8b_last | binary | edit_file | 157 | 0.994 | 0.500 [0.500, 0.500] |
| llama8b_last | binary | move_file | 15 | 1.000 | — |
| llama8b_last | binary | read_file | 7 | 1.000 | — |
| llama8b_last | binary | send_email | 33 | 1.000 | 0.500 [0.500, 0.500] |
| llama8b_last | binary | transfer_money | 33 | 0.991 | 0.500 [0.500, 0.500] |
| llama8b_last | binary | trash_file | 8 | 1.000 | — |
| llama8b_pre | binary | delete_file | 44 | 0.605 | 0.499 [0.433, 0.562] |
| llama8b_pre | binary | edit_file | 157 | 0.733 | 0.484 [0.453, 0.515] |
| llama8b_pre | binary | move_file | 15 | 0.476 | — |
| llama8b_pre | binary | read_file | 7 | 0.547 | — |
| llama8b_pre | binary | send_email | 33 | 0.611 | 0.521 [0.445, 0.594] |
| llama8b_pre | binary | transfer_money | 33 | 0.593 | 0.488 [0.413, 0.561] |
| llama8b_pre | binary | trash_file | 8 | 0.569 | — |
| qwen3b_last | binary | delete_file | 44 | 0.955 | 0.490 [0.437, 0.544] |
| qwen3b_last | binary | edit_file | 157 | 0.975 | 0.532 [0.494, 0.566] |
| qwen3b_last | binary | move_file | 15 | 1.000 | — |
| qwen3b_last | binary | read_file | 7 | 1.000 | — |
| qwen3b_last | binary | send_email | 33 | 1.000 | 0.504 [0.469, 0.539] |
| qwen3b_last | binary | transfer_money | 33 | 1.000 | 0.500 [0.500, 0.500] |
| qwen3b_last | binary | trash_file | 8 | 1.000 | — |
| qwen3b_pre | binary | delete_file | 44 | 0.612 | 0.522 [0.457, 0.588] |
| qwen3b_pre | binary | edit_file | 157 | 0.739 | 0.482 [0.451, 0.514] |
| qwen3b_pre | binary | move_file | 15 | 0.546 | — |
| qwen3b_pre | binary | read_file | 7 | 0.588 | — |
| qwen3b_pre | binary | send_email | 33 | 0.582 | 0.568 [0.495, 0.641] |
| qwen3b_pre | binary | transfer_money | 33 | 0.634 | 0.519 [0.451, 0.586] |
| qwen3b_pre | binary | trash_file | 8 | 0.569 | — |

## Cross-environment transfer (binary; layer chosen on source validation)

| tag | direction | layer | bal-acc [95% CI] | bal-acc, target re-centred |
|---|---|---|---|---|
| llama8b_transfer | sandbox->sciworld | 14 | 0.441 [0.351, 0.532] | 0.449 |
| llama8b_transfer | sciworld->sandbox | 9 | 0.500 [0.500, 0.500] | 0.500 |
| qwen3b_transfer | sandbox->sciworld | 22 | 0.442 [0.354, 0.526] | 0.503 |
| qwen3b_transfer | sciworld->sandbox | 21 | 0.531 [0.510, 0.554] | 0.500 |

## Frozen probes on reworded test rows (terse = on/off wording, flip = irrelevant setting flipped; same ids and labels)

| tag | task | n | layer | bal-acc original | bal-acc reworded | same prediction |
|---|---|---|---|---|---|---|
| llama8b_flip | binary | 267 | 16 | 0.963 | 0.963 | 0.978 |
| llama8b_flip | three | 359 | 19 | 0.968 | 0.961 | 0.972 |
| llama8b_terse | binary | 297 | 16 | 0.967 | 0.817 | 0.805 |
| llama8b_terse | three | 389 | 19 | 0.971 | 0.565 | 0.620 |
| qwen3b_flip | binary | 267 | 22 | 0.982 | 0.978 | 0.996 |
| qwen3b_flip | three | 359 | 22 | 0.993 | 0.980 | 0.986 |
| qwen3b_terse | binary | 297 | 22 | 0.984 | 0.799 | 0.798 |
| qwen3b_terse | three | 389 | 22 | 0.993 | 0.592 | 0.625 |

## Behaviour runs (system prompt ablation, pilot, model); CIs over pairs / over templates

| run | pairs | target taken irr / rev | refused irr / rev | caution irr / rev | sensitivity A* [templates CI] | A_out* | success irr / rev | parse failures |
|---|---|---|---|---|---|---|---|---|
| llama8b_careful | 72 | 0.83 / 0.83 | 0.17 / 0.17 | 0.17 / 0.17 | 0.000 [0.000, 0.000] | 0.000 | 0.83 / 0.83 | 0 |
| llama8b_neutral | 72 | 0.83 / 0.85 | 0.17 / 0.15 | 0.25 / 0.26 | -0.014 [-0.083, 0.042] | 0.014 | 0.83 / 0.82 | 0 |
| qwen3b_careful | 72 | 0.85 / 0.78 | 0.15 / 0.22 | 0.33 / 0.42 | -0.083 [-0.194, 0.000] | -0.069 | 0.65 / 0.62 | 0 |
| qwen3b_neutral | 72 | 0.76 / 0.83 | 0.24 / 0.17 | 0.42 / 0.46 | -0.042 [-0.125, 0.000] | 0.069 | 0.61 / 0.64 | 0 |
| qwen3b_pilot | 6 | 0.83 / 0.83 | 0.17 / 0.17 | 0.50 / 0.50 | 0.000 [0.000, 0.000] | 0.000 | 0.67 / 0.67 | 0 |
