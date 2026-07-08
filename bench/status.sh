#!/bin/bash
export PATH=$HOME/.local/bin:$PATH
J=/root/tb/tb-jobs/glm52-full-tb21-v1
T=$(date +%H:%M:%S)
GLM=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 http://<HEAD_IP>:8210/v1/models)
H=$(pgrep -f "harbor run" >/dev/null && echo up || echo down)
S=$(python3 -c "import json;d=json.load(open('$J/result.json'))['stats'];print('done=%d err=%d pend=%d run=%d'%(d['n_completed_trials'],d['n_errored_trials'],d['n_pending_trials'],d['n_running_trials']))" 2>/dev/null || echo "warming-up")
P=$(python3 -c "import json;e=json.load(open('$J/result.json'))['stats']['evals'];v=list(e.values())[0]['reward_stats']['reward'];print('pass=%d fail=%d'%(len(v.get('1.0',[])),len(v.get('0.0',[]))))" 2>/dev/null || echo "")
CUR=$(docker ps --format "{{.Names}}" 2>/dev/null | grep env-main | sed -E "s/__.*//" | head -1)
PEAKMEM=$(grep "^Mem:" /root/tb/resource-metrics.log 2>/dev/null | awk '{print $3}' | sort -n | tail -1)
echo "[$T] GLM=$GLM harbor=$H | $S $P | task=$CUR | peakMemMB=${PEAKMEM:-?}/8192"
