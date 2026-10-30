#!/bin/bash
# File-driven experiment queue. Runs the commands in queue/pending.txt one at a time, in order, and records each
# finished line with its exit code in queue/done.txt. Lines can be added to pending.txt while the queue runs (an
# evaluation between two training jobs, the next tier); the queue exits when the file is empty. Every job tracks
# itself (runs/ on GitHub). Start it wrapped in caffeinate at low priority:
#     caffeinate -i nice -n 10 bash scripts/queue_run.sh
cd "$(dirname "$0")/.." || exit 1
mkdir -p queue
PENDING=queue/pending.txt; DONE=queue/done.txt; LOG=queue/log.txt
touch "$PENDING"
while :; do
  line=$(grep -vE '^[[:space:]]*(#|$)' "$PENDING" | head -1)
  [ -z "$line" ] && break
  python3 - "$PENDING" "$line" <<'EOF'
import sys
path, job = sys.argv[1], sys.argv[2]
lines = open(path).read().split("\n")
for i, l in enumerate(lines):
    if l == job:
        del lines[i]
        break
open(path, "w").write("\n".join(lines))
EOF
  echo "[queue] $(date '+%F %T') START $line" | tee -a "$LOG"
  bash -c "$line"; rc=$?
  echo "[queue] $(date '+%F %T') END rc=$rc $line" | tee -a "$LOG"
  echo "$(date '+%F %T') rc=$rc $line" >> "$DONE"
done
echo "[queue] $(date '+%F %T') QUEUE EMPTY" | tee -a "$LOG"
