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
# Two more collapsed 4k runs (uniform predictions on dev), rerun with another seed to tell config from chance.
LREV=$(python3 -c "import json,glob; f=sorted(glob.glob('runs/*-teacher-qwen3-4b/final.json'))[-1]; print(json.load(open(f)).get('labels_revision',''))")
job $TRAIN --config configs/tino-adapter.json --set name=abl-lr2e-4-4k-s1 seed=1 backbone_lr=2e-4 "${SMALL[@]}"
job $TRAIN --config configs/tino-distil.json --set name=abl-distil-a05-4k-s1 seed=1 "distil={\"labels_repo\":\"BrandeisPatrick/tino-teacher-labels\",\"labels_prefix\":\"qwen3-4b-instruct-2507\",\"labels_revision\":\"$LREV\",\"alpha\":0.5,\"teacher_temperature\":\"manifest\"}" "${SMALL[@]}"
# RQ4: teacher-only distillation (alpha 1.0) gave the best new-source accuracy at 4k; confirm at full data with test reads.
job $TRAIN --config configs/tino-distil.json --set name=tino-distil-a1 "distil={\"labels_repo\":\"BrandeisPatrick/tino-teacher-labels\",\"labels_prefix\":\"qwen3-4b-instruct-2507\",\"labels_revision\":\"$LREV\",\"alpha\":1.0,\"teacher_temperature\":\"manifest\"}"
d=$(ls -d runs/*-tino-distil-a1 2>/dev/null | tail -1); [ -n "$d" ] && [ -f "$d/final.json" ] && job uv run python scripts/publish.py --run "$d"
# RQ4: the adapter's two seeds differ by 3.8 points out of domain, so distillation needs a second seed too.
job $TRAIN --config configs/tino-distil.json --set name=tino-distil-s1 seed=1 "distil={\"labels_repo\":\"BrandeisPatrick/tino-teacher-labels\",\"labels_prefix\":\"qwen3-4b-instruct-2507\",\"labels_revision\":\"$LREV\",\"alpha\":0.5,\"teacher_temperature\":\"manifest\"}"
echo "[queue] $(date '+%F %T') FOLLOWUP DONE"
