#!/bin/bash
# Experiment queue for RESEARCH.md (RQ1-RQ4). One job at a time, each at low priority under the resource guard.
# Every job tracks itself (runs/ on GitHub, weights and labels on the Hub). A failed or aborted job is logged and the
# queue moves on. Run it wrapped in caffeinate:
#     caffeinate -i nice -n 10 bash scripts/queue.sh
cd "$(dirname "$0")/.." || exit 1

job() { echo "[queue] $(date '+%F %T') START $*"; "$@"; local rc=$?; echo "[queue] $(date '+%F %T') END rc=$rc $*"; }
latest() { ls -d runs/*-"$1" 2>/dev/null | tail -1; }
publish() { local d; d=$(latest "$1"); [ -n "$d" ] && [ -f "$d/final.json" ] && job uv run python scripts/publish.py --run "$d"; }
TRAIN="uv run python scripts/train.py"
DEV='[["decision-v7","development"],["transfer-v4","development"]]'
SMALL=(evals_per_epoch=2 "final_evals=$DEV")

# Teacher labels must be complete before any distillation job; pin the dataset revision the labeling run recorded.
while pgrep -f scripts/label_teacher.py >/dev/null; do sleep 60; done
LREV=$(python3 -c "import json,glob; f=sorted(glob.glob('runs/*-teacher-qwen3-4b/final.json'))[-1]; print(json.load(open(f)).get('labels_revision',''))")
[ -z "$LREV" ] && { echo "[queue] teacher labels incomplete; distillation jobs will fail"; }
distil() { echo "{\"labels_repo\":\"BrandeisPatrick/tino-teacher-labels\",\"labels_prefix\":\"qwen3-4b-instruct-2507\",\"labels_revision\":\"$LREV\",\"alpha\":$1,\"teacher_temperature\":\"manifest\"}"; }
echo "[queue] teacher labels revision $LREV"

# RQ4: distill the autoregressive teacher into the decision student (released model).
job $TRAIN --config configs/tino-distil.json --set "distil=$(distil 0.5)"
publish tino-distil

# RQ3a: instruct vs thinking Qwen as zero-shot decision makers.
job uv run python scripts/think_eval.py --config configs/think-eval.json

# RQ3b: post-trained, thinking-capable Qwen3-0.6B as the student's starting point (released model).
job $TRAIN --config configs/tino-adapter.json --set name=tino-adapter-post base_model=Qwen/Qwen3-0.6B base_revision=c1899de289a04d12100db370d81485cdf75e47ca
publish tino-adapter-post

# RQ1 + RQ2: ablations on a fixed 4,000-question subset, judged on development splits.
job $TRAIN --config configs/tino-adapter.json --set name=abl-adapter-4k train_limit=4000 "${SMALL[@]}"
job $TRAIN --config configs/tino-finetuned.json --set name=abl-finetuned-4k train_limit=4000 "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-adapter-1k train_limit=1000 "${SMALL[@]}"
job $TRAIN --config configs/tino-finetuned.json --set name=abl-finetuned-1k train_limit=1000 "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-rank4-4k train_limit=4000 'lora={"rank":4,"scale":2.0,"dropout":0.05}' "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-rank64-4k train_limit=4000 'lora={"rank":64,"scale":2.0,"dropout":0.05}' "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-attn-only-4k train_limit=4000 'lora={"rank":16,"scale":2.0,"dropout":0.05,"keys":["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj"]}' "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-lr5e-5-4k train_limit=4000 backbone_lr=5e-5 "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-lr2e-4-4k train_limit=4000 backbone_lr=2e-4 "${SMALL[@]}"
job $TRAIN --config configs/tino-adapter.json --set name=abl-brier-4k train_limit=4000 brier_weight=1.0 "${SMALL[@]}"

# RQ4 sweep: how much weight on the teacher (1.0 = teacher only, no gold labels).
job $TRAIN --config configs/tino-distil.json --set name=abl-distil-a025-4k train_limit=4000 "distil=$(distil 0.25)" "${SMALL[@]}"
job $TRAIN --config configs/tino-distil.json --set name=abl-distil-a05-4k train_limit=4000 "distil=$(distil 0.5)" "${SMALL[@]}"
job $TRAIN --config configs/tino-distil.json --set name=abl-distil-a1-4k train_limit=4000 "distil=$(distil 1.0)" "${SMALL[@]}"

# RQ2: a second seed at full data for both methods.
job $TRAIN --config configs/tino-adapter.json --set name=tino-adapter-s1 seed=1
job $TRAIN --config configs/tino-finetuned.json --set name=tino-finetuned-s1 seed=1

echo "[queue] $(date '+%F %T') ALL DONE"
