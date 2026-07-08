#!/bin/bash
# Self-healing watchdog for the full-model TB2.1 run (cron */5).
#  - GLM down          -> pause harbor (don't burn tasks on a down model)
#  - GLM up, harbor down, not finished -> resume
#  - GLM up, harbor up, job.log stale >75min -> wedge restart
export PATH=$HOME/.local/bin:$PATH
J=/root/tb/tb-jobs/glm52-full-tb21-v1
LOG=/root/tb/harbor-watchdog.log
ts(){ date +%Y-%m-%dT%H:%M:%S; }
GLM=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 http://<HEAD_IP>:8210/v1/models)
HARBOR=$(pgrep -f "harbor run.*glm52-full-tb21-v1" >/dev/null && echo up || echo down)
DONE=$(python3 -c "import json;print(1 if json.load(open('$J/result.json')).get('finished_at') else 0)" 2>/dev/null || echo 0)
AGE=999999; [ -f "$J/job.log" ] && AGE=$(( $(date +%s) - $(stat -c %Y "$J/job.log") ))
echo "$(ts) GLM=$GLM HARBOR=$HARBOR DONE=$DONE AGE=${AGE}s" >> "$LOG"
[ "$DONE" = "1" ] && exit 0
if [ "$GLM" != "200" ]; then
  [ "$HARBOR" = "up" ] && { pkill -9 -f "harbor run.*glm52-full-tb21-v1"; echo "$(ts) GLM down -> paused harbor" >> "$LOG"; }
  exit 0
fi
if [ "$HARBOR" = "down" ]; then
  bash /root/tb/tb-run.sh >> "$LOG" 2>&1; echo "$(ts) GLM up + harbor down -> resumed" >> "$LOG"; exit 0
fi
if [ "$AGE" -gt 4500 ]; then
  bash /root/tb/tb-run.sh >> "$LOG" 2>&1; echo "$(ts) wedge age=${AGE}s -> restarted" >> "$LOG"
fi
