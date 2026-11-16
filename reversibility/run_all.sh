#!/usr/bin/env bash
# Full pipeline for ONE model on a GPU box (Colab/Kaggle). Run once per model:
#   PILOT=1 MODEL=Qwen/Qwen2.5-3B-Instruct TAG=qwen3b bash run_all.sh    # agent pilot only: 6 pairs, task seed 99 (not the eval set)
#   MODEL=Qwen/Qwen2.5-3B-Instruct TAG=qwen3b bash run_all.sh
#   LOAD_4BIT=1 MODEL=meta-llama/Llama-3.1-8B-Instruct TAG=llama8b bash run_all.sh
#   QUICK=1 MODEL=Qwen/Qwen2.5-0.5B-Instruct TAG=quick bash run_all.sh    # plumbing test on small subsets, numbers meaningless
#   PROBES_ONLY=1 MODEL=... TAG=... bash run_all.sh   # after a label fix: CPU probes + re-scored stated answers, no model
#   PROBES_ONLY=1 SW=data/sciworld_v1_relabel.jsonl MODEL=... TAG=... bash run_all.sh   # after the ScienceWorld re-search
# Inputs are frozen files in data/ (sandbox, supplement, ScienceWorld cut-off, reworded test copies); nothing is regenerated.
# One model per process, greedy decoding, fixed seeds.
set -euo pipefail
MODEL=${MODEL:?set MODEL}; TAG=${TAG:?set TAG}
N_PER=${N_PER:-12}
mkdir -p results figures
pip freeze > "results/requirements.lock_${TAG}.txt"
if [ -n "${PILOT:-}" ]; then
  python agent_eval.py --model "$MODEL" --tag "${TAG}_pilot" --n-per-template 1 --task-seed 99; exit 0
fi

D=data; PA=""
if [ -n "${QUICK:-}" ]; then
  D=data/quick; mkdir -p "$D"; N_PER=1; PA="--layer-step 6 --seeds 0"
  for n in sandbox sandbox_extra sciworld_v1; do shuf -n 150 --random-source=<(yes) "data/$n.jsonl" > "$D/$n.jsonl"; done
  python make_variants.py "$D/sandbox.jsonl" "$D/sandbox_extra.jsonl"
fi
# RQ1/RQ2 use the sandbox only. ScienceWorld is the transfer target plus its own stated-answer table: most of its
# non-reversible v1 labels are search cut-offs, so pooling would put them in the headline (Amendment 2).
SB="$D/sandbox.jsonl $D/sandbox_extra.jsonl"; HSB="data/h_${TAG}_sandbox data/h_${TAG}_sandbox_extra"
SW=${SW:-$D/sciworld_v1.jsonl}; HSW="data/h_${TAG}_sciworld_v1"     # SW = re-searched copy (same rows, same order) later

# ---- Step 2: hidden states (last action token, mean over action tokens, token before the action), every layer ----
[ -n "${PROBES_ONLY:-}" ] || python extract_hidden.py --model "$MODEL" --data $SB "$SW" "$D/sandbox_terse.jsonl" "$D/sandbox_flip.jsonl" \
  --out $HSB "$HSW" "data/h_${TAG}_sandbox_terse" "data/h_${TAG}_sandbox_flip"

# ---- Step 3 (RQ1): probes are CPU-only, so they run in the background while the GPU does Steps 5 and 4 ----
( set +e
  python probes.py --data $SB --hidden $HSB --pos last --lovo --tag "${TAG}_last" $PA        # main result; stated_answer needs it
  python probes.py --data $SB --hidden $HSB --pos pre --lovo --tag "${TAG}_pre" $PA          # action-free control position
  python probes.py --data $SB --hidden $HSB --pos last --eval-data "$D/sandbox_terse.jsonl" \
    --eval-hidden "data/h_${TAG}_sandbox_terse" --tag "${TAG}_terse" $PA                        # keyword vs concept
  python probes.py --data $SB --hidden $HSB --pos last --eval-data "$D/sandbox_flip.jsonl" \
    --eval-hidden "data/h_${TAG}_sandbox_flip" --tag "${TAG}_flip" $PA                          # irrelevant-line flip
  python probes.py --data $SB "$SW" --hidden $HSB "$HSW" --pos last --transfer --tag "${TAG}_transfer" $PA  # compulsory ablation
  python probes.py --data $SB --hidden $HSB --pos last --minimal-only --tag "${TAG}_minimal" $PA
  python probes.py --data $SB --hidden $HSB --pos mean --layer-step 2 --tag "${TAG}_mean" $PA
  for h in 1 2; do                                     # h3 = the main label by definition
    python probes.py --data $SB --hidden $HSB --pos last --label-key "label_h$h" --layer-step 2 --tag "${TAG}_h$h" $PA
  done ) > "results/probes_background_${TAG}.log" 2>&1 &
PROBES=$!

# ---- Step 5 (RQ3) on the GPU meanwhile, then Step 4 (RQ2) once the probe predictions exist ----
if [ -z "${PROBES_ONLY:-}" ]; then
  python agent_eval.py --model "$MODEL" --tag "${TAG}_neutral" --n-per-template "$N_PER"
  python agent_eval.py --model "$MODEL" --tag "${TAG}_careful" --careful --n-per-template "$N_PER"
fi
wait $PROBES || true                                  # a failed ablation must not stop RQ2 or the tables
if grep -q Traceback "results/probes_background_${TAG}.log"; then
  echo "WARNING: a background probe run failed, see results/probes_background_${TAG}.log"
fi
python stated_answer.py --model "$MODEL" --data $SB --tag "${TAG}" --test-only ${PROBES_ONLY:+--rescore} \
  --probe-preds "results/probe_preds_${TAG}_last_binary.csv" "results/probe_preds_${TAG}_last_three.csv"
python make_tables.py --model-tag "${TAG}"
# ScienceWorld stated answers last (cheap, 46 rows), so a re-search can re-score them without a GPU
python stated_answer.py --model "$MODEL" --data "$SW" --tag "${TAG}_sciworld" --test-only ${PROBES_ONLY:+--rescore} \
  || echo "WARNING: ScienceWorld stated answers failed"
