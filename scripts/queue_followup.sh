#!/bin/bash
# RQ1 follow-ups, added after the first 4k ablations: two LoRA variants failed (rank 64 diverged; attention-only never
# left the uniform-guess plateau). These separate a real effect from single-seed bad luck and test rank fairly.
#     caffeinate -i nice -n 10 bash scripts/queue_followup.sh
cd "$(dirname "$0")/.." || exit 1
job() { echo "[queue] $(date '+%F %T') START $*"; "$@"; local rc=$?; echo "[queue] $(date '+%F %T') END rc=$rc $*"; }
TRAIN="uv run python scripts/train.py"
DEV='[["decision-v7","development"],["transfer-v4","development"]]'
SMALL=(train_limit=4000 evals_per_epoch=2 "final_evals=$DEV")
job $TRAIN --config configs/tino-adapter.json --set name=abl-attn-only-4k-s1 seed=1 'lora={"rank":16,"scale":2.0,"dropout":0.05,"keys":["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]}' "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-rank64-a32-4k 'lora={"rank":64,"scale":0.5,"dropout":0.05}' "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-adapter-4k-s1 seed=1 "${SMALL[@]}"
echo "[queue] $(date '+%F %T') FOLLOWUP DONE"
