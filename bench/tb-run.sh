#!/bin/bash
# Launch/resume the full-model TB2.1 run (89 tasks) against glm-5.2-full over tailscale.
# Resumable: kills any running harbor, drops an incomplete current trial, resumes same job.
export PATH=$HOME/.local/bin:$PATH
GLM=http://<HEAD_IP>:8210/v1
JOB=glm52-full-tb21-v1
cd /root/tb || exit 1
pkill -9 -f "harbor run.*glm52-full-tb21-v1" 2>/dev/null; sleep 2
# drop incomplete current trial (no result.json) so resume redoes it cleanly
CUR=$(basename "$(ls -dt tb-jobs/$JOB/*/ 2>/dev/null | head -1)" 2>/dev/null)
if [ -n "$CUR" ] && [ -d "tb-jobs/$JOB/$CUR" ] && [ ! -f "tb-jobs/$JOB/$CUR/result.json" ]; then
  rm -rf "tb-jobs/$JOB/$CUR"
fi
export OPENAI_API_KEY=dummy OPENAI_API_BASE=$GLM OPENAI_BASE_URL=$GLM
setsid harbor run -d terminal-bench/terminal-bench-2-1 -a terminus-2 -m openai/glm-5.2-full \
  --ae OPENAI_API_KEY=dummy --ae OPENAI_API_BASE=$GLM --ae OPENAI_BASE_URL=$GLM \
  --agent-kwarg "model_info={\"max_input_tokens\":100000,\"max_output_tokens\":32768}" \
  --agent-kwarg "llm_call_kwargs={\"timeout\":3600}" \
  -n 1 --agent-timeout-multiplier 6 -r 2 -o ./tb-jobs --job-name $JOB </dev/null >/root/tb-run.log 2>&1 &
sleep 5
echo "harbor run pid(s): $(pgrep -f 'harbor run' | tr '\n' ' ')"
