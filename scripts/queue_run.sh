#!/bin/bash
# File-driven experiment queue. Runs the commands in queue/pending.txt one at a time, in order, and records each
# finished line with its exit code in queue/done.txt. Lines can be added to pending.txt while the queue runs (an
# evaluation between two training jobs, the next tier); the queue exits when the file is empty. Every job tracks
# itself (runs/ on GitHub). Start it wrapped in caffeinate at low priority:
#     caffeinate -i nice -n 10 bash scripts/queue_run.sh
# Before each job the runner waits until the Mac has memory to spare: the guard's free-memory level
# (kern.memorystatus_level) at or above QUEUE_MIN_FREE_PCT (default 35) for three checks a minute apart, and no
# Ollama model server running. A crowded machine then delays the queue instead of aborting job after job.
cd "$(dirname "$0")/.." || exit 1
mkdir -p queue
PENDING=queue/pending.txt; DONE=queue/done.txt; LOG=queue/log.txt
MIN_FREE=${QUEUE_MIN_FREE_PCT:-35}
touch "$PENDING"

wait_for_memory() {
  local ok=0 last_log=0 now free why
  while [ "$ok" -lt 3 ]; do
    free=$(sysctl -n kern.memorystatus_level)
    why=""
    if pgrep -x llama-server >/dev/null 2>&1; then why="an Ollama model is loaded"; fi
    if [ "${free:-0}" -lt "$MIN_FREE" ]; then why="${why:+$why, }${free}% free, need ${MIN_FREE}%"; fi
    if [ -z "$why" ]; then
      ok=$((ok+1))
    else
      ok=0; now=$(date +%s)
      if [ $((now - last_log)) -ge 600 ]; then
        echo "[queue] $(date '+%F %T') WAITING for memory: $why" | tee -a "$LOG"; last_log=$now
      fi
    fi
    [ "$ok" -lt 3 ] && sleep 60
  done
}

while :; do
  line=$(grep -vE '^[[:space:]]*(#|$)' "$PENDING" | head -1)
  [ -z "$line" ] && break
  wait_for_memory
  line=$(grep -vE '^[[:space:]]*(#|$)' "$PENDING" | head -1)   # the file may have changed while waiting
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
