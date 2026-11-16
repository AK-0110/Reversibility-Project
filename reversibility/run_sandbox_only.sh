#!/usr/bin/env bash
# Sandbox-only pipeline (no ScienceWorld, no cross-env transfer). Use on a GPU box (Colab/Kaggle):
#   !pip install -r requirements.txt && MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b bash run_sandbox_only.sh
set -euo pipefail
MODEL=${MODEL:?set MODEL}; TAG=${TAG:?set TAG}
BEST_LAYER=${BEST_LAYER:-}
mkdir -p data results figures
DATA="data/sandbox.jsonl"; HID="data/h_${TAG}_sandbox"
python extract_hidden.py --model "$MODEL" --data "$DATA" --out "$HID"
python probes.py --data "$DATA" --hidden "$HID" --pos last --tag ${TAG}_last
python probes.py --data "$DATA" --hidden "$HID" --pos mean --tag ${TAG}_mean
python probes.py --data "$DATA" --hidden "$HID" --pos last --minimal-only --tag ${TAG}_minimal
for h in 1 2 3; do python probes.py --data "$DATA" --hidden "$HID" --pos last --label-key label_h$h --tag ${TAG}_h$h; done
python stated_answer.py --model "$MODEL" --data "$DATA" --tag ${TAG} --test-only --probe-preds results/probe_preds_${TAG}_last_three.csv
python agent_eval.py --model "$MODEL" --tag ${TAG}_neutral
python agent_eval.py --model "$MODEL" --tag ${TAG}_careful --careful
[ -n "$BEST_LAYER" ] && python steering.py --model "$MODEL" --data "$DATA" --hidden "$HID" --layer $BEST_LAYER --tag ${TAG}
python make_tables.py --model-tag ${TAG}
