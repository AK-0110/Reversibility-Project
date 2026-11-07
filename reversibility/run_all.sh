#!/usr/bin/env bash
# Full pipeline for ONE model. Run once per model:
#   MODEL=Qwen/Qwen2.5-7B-Instruct TAG=qwen7b bash run_all.sh
#   MODEL=meta-llama/Llama-3.1-8B-Instruct TAG=llama8b bash run_all.sh
#   MODEL=Qwen/Qwen2.5-14B-Instruct TAG=qwen14b bash run_all.sh     (optional, model-size ablation)
# One model is loaded per process, never two at once (GPU ~50 GB). Everything is greedy and seeded.
set -euo pipefail
MODEL=${MODEL:?set MODEL}; TAG=${TAG:?set TAG}
BEST_LAYER=${BEST_LAYER:-}          # fill in after step 3 (read it from results/probe_${TAG}_last.json) for steering
mkdir -p data results figures
pip freeze > requirements.lock.txt

# ---- once, model-independent (skip if data/*.jsonl already exist) ----
[ -f data/sandbox.jsonl ]  || python make_sandbox_data.py --per-class 450 --manual 100
[ -f data/sciworld.jsonl ] || python sciworld_label.py --per-class 250 --workers 8 --out data/sciworld.jsonl
# optional: python toolemu_cases.py build ...   python prepare_bytesized.py build ...   (after manual labelling)

DATA="data/sandbox.jsonl data/sciworld.jsonl"
HID="data/h_${TAG}_sandbox data/h_${TAG}_sciworld"

# ---- Step 2: hidden states (last action token + mean over action tokens, every layer) ----
python extract_hidden.py --model $MODEL --data data/sandbox.jsonl  --out data/h_${TAG}_sandbox
python extract_hidden.py --model $MODEL --data data/sciworld.jsonl --out data/h_${TAG}_sciworld

# ---- Step 3 (RQ1) + compulsory ablations (token position, minimal pairs only, cross-env transfer) + optional ones ----
python probes.py --data $DATA --hidden $HID --pos last --tag ${TAG}_last
python probes.py --data $DATA --hidden $HID --pos mean --tag ${TAG}_mean
python probes.py --data $DATA --hidden $HID --pos last --minimal-only --tag ${TAG}_minimal
python probes.py --data $DATA --hidden $HID --pos last --transfer --tag ${TAG}_transfer
for h in 1 2 3; do python probes.py --data $DATA --hidden $HID --pos last --label-key label_h$h --tag ${TAG}_h$h; done

# ---- Step 4 (RQ2) ----
python stated_answer.py --model $MODEL --data $DATA --tag ${TAG} --test-only \
    --probe-preds results/probe_preds_${TAG}_last_three.csv

# ---- Step 5 (RQ3): neutral prompt, and the "be careful" prompt ablation ----
python agent_eval.py --model $MODEL --tag ${TAG}_neutral
python agent_eval.py --model $MODEL --tag ${TAG}_careful --careful

# ---- Step 6 (bonus): steering + uncertainty ----
if [ -n "$BEST_LAYER" ]; then
  python steering.py --model $MODEL --data $DATA --hidden $HID --layer $BEST_LAYER --tag ${TAG}
fi
# python uncertainty.py --model $MODEL --data data/bytesized_labelled.jsonl --tag ${TAG}

python make_tables.py --model-tag ${TAG}
