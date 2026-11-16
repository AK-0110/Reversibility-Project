#!/usr/bin/env bash
# Sandbox-only pipeline (no ScienceWorld, no cross-env transfer). Use on a GPU box (Colab/Kaggle):
#   MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b bash run_sandbox_only.sh
#   QUICK=1 MODEL=Qwen/Qwen2.5-0.5B-Instruct TAG=quick bash run_sandbox_only.sh   # ~5-10 min plumbing test, numbers meaningless
set -euo pipefail
MODEL=${MODEL:?set MODEL}; TAG=${TAG:?set TAG}
BEST_LAYER=${BEST_LAYER:-}
N_PER=${N_PER:-12}
mkdir -p data results figures
DATA="data/sandbox.jsonl"; HID="data/h_${TAG}_sandbox"
if [ -n "${QUICK:-}" ]; then
  DATA="data/sandbox_quick.jsonl"; shuf -n 400 --random-source=<(yes) data/sandbox.jsonl > "$DATA"; N_PER=1
fi
python extract_hidden.py --model "$MODEL" --data "$DATA" --out "$HID"
python probes.py --data "$DATA" --hidden "$HID" --pos last --tag ${TAG}_last   # needed by stated_answer
# remaining probes are CPU-only: run them in the background while the GPU stages go
( python probes.py --data "$DATA" --hidden "$HID" --pos mean --tag ${TAG}_mean
  python probes.py --data "$DATA" --hidden "$HID" --pos last --minimal-only --tag ${TAG}_minimal
  for h in 1 2 3; do python probes.py --data "$DATA" --hidden "$HID" --pos last --label-key label_h$h --tag ${TAG}_h$h; done ) &
PROBES=$!
python stated_answer.py --model "$MODEL" --data "$DATA" --tag ${TAG} --test-only --probe-preds results/probe_preds_${TAG}_last_three.csv
python agent_eval.py --model "$MODEL" --tag ${TAG}_neutral --n-per-template $N_PER
python agent_eval.py --model "$MODEL" --tag ${TAG}_careful --careful --n-per-template $N_PER
wait $PROBES
[ -z "$BEST_LAYER" ] || python steering.py --model "$MODEL" --data "$DATA" --hidden "$HID" --layer $BEST_LAYER --tag ${TAG}
python make_tables.py --model-tag ${TAG}
