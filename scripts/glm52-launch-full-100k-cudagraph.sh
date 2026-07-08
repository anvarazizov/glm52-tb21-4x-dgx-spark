#!/usr/bin/env bash
# glm52-launch-full-100k.sh — FULL (unpruned) GLM-5.2 Int4-Int8Mix on our PROVEN
# p5 serving config (nvfp4 KV, 100K, enforce-eager, MTP3, gmu 0.87), TP=4.
# = glm52-launch-p5.sh with weights swapped p5 -> full (both serve + MTP drafter),
#   --enable-return-routed-experts dropped. No FULL-cudagraph, no fp8, no 200K —
#   the two things that wedged the 200K verbatim attempt are gone.
#   ./glm52-launch-full-100k.sh [--dry-run|--stop]
set -uo pipefail

# ===== CONFIG =====
NODES=(10.78.0.1 10.78.0.2 10.78.0.3 10.78.0.4)   # rank 0..3, head first
IMAGE="vllm-glm52-recon-nvfp4:latest"
NAME="vllm_slot"
PORT=8210
MASTER_PORT=29501
WEIGHTS_DIR="/opt/glm52"   # LOCAL on every node: hub/{glm52-int4-int8mix,nccl-2.30.4}
# ==================

say(){ printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
die(){ printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
DRYRUN=0; STOP=0
for a in "$@"; do case "$a" in --dry-run)DRYRUN=1;; --stop)STOP=1;; *)die "bad arg $a";; esac; done
NNODES="${#NODES[@]}"; HEAD="${NODES[0]}"

runon(){ local ip="$1"; shift; if [ "$ip" = "$HEAD" ]; then bash -c "$*"; else ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$ip" "$*" </dev/null; fi; }

if [ "$STOP" = 1 ]; then
  say "stopping '$NAME' on all $NNODES nodes"
  for ip in "${NODES[@]}"; do runon "$ip" "docker rm -f $NAME 2>/dev/null" && printf '   stopped on %s\n' "$ip"; done
  exit 0
fi

ENVV=(
  -e "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800"
  -e "LD_PRELOAD=/cache/huggingface/hub/nccl-2.30.4/libnccl.so.2"
  -e "HF_HOME=/cache/huggingface"
  -e "TRITON_CACHE_DIR=/cache/huggingface/.tritoncache"
  -e "HF_HUB_OFFLINE=1"
  -e "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1"
  -e "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256"
  -e "GLM52_BIND_HOST_TRITON=1"
  -e "GLM52_MQA_LOGITS_TRITON=1"
  -e "GLM52_PAGED_MQA_TRITON=1"
  -e "GLM52_PAGED_MQA_TOPK_CHUNK_SIZE=8192"
  -e "GLM52_B12X_MLA=1"
  -e "TORCH_CUDA_ARCH_LIST=12.1a"
  -e "NCCL_NET=IB"
  -e "NCCL_IB_DISABLE=0"
  -e "NCCL_IB_HCA=rocep1s0f1"
  -e "NCCL_SOCKET_IFNAME=enp1s0f1np1"
  -e "GLOO_SOCKET_IFNAME=enp1s0f1np1"
  -e "NCCL_IB_GID_INDEX=3"
  -e "NCCL_CROSS_NIC=1"
  -e "NCCL_CUMEM_ENABLE=0"
  -e "NCCL_IGNORE_CPU_AFFINITY=1"
  -e "NCCL_DEBUG=WARN"
)
BASE=(
  --cap-add IPC_LOCK --ulimit memlock=-1:-1
  --network host --ipc host --shm-size 10gb --gpus all
  --device /dev/infiniband:/dev/infiniband
  -v "$WEIGHTS_DIR:/cache/huggingface"
  -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro
)
SERVE=(
  vllm serve /cache/huggingface/hub/glm52-int4-int8mix
  --served-model-name glm-5.2-full --host 0.0.0.0 --port "$PORT"
  --trust-remote-code --reasoning-parser glm45 --tool-call-parser glm47 --enable-auto-tool-choice
  --enable-prefix-caching
  --speculative-config '{"model":"/cache/huggingface/hub/glm52-int4-int8mix","method":"mtp","num_speculative_tokens":3,"attention_backend":"FLASHMLA_SPARSE"}'
  --tensor-parallel-size 4 --pipeline-parallel-size 1
  --max-model-len 100000 --max-num-seqs 2 --max-num-batched-tokens 4096
  --gpu-memory-utilization 0.90 --kv-cache-dtype nvfp4_ds_mla
  --distributed-executor-backend mp --compilation-config '{"cudagraph_mode":"FULL"}'
)

docker_run_cmd(){ # rank headless
  local rank="$1" headless="$2"
  local cmd=(docker run -d --name "$NAME" "${BASE[@]}" "${ENVV[@]}"
             -e "NODE_RANK=$rank" -e "MASTER_ADDR=$HEAD"
             "$IMAGE" "${SERVE[@]}"
             --nnodes "$NNODES" --node-rank "$rank" --master-addr "$HEAD" --master-port "$MASTER_PORT")
  [ "$headless" = 1 ] && cmd+=(--headless)
  local out="" t; for t in "${cmd[@]}"; do out+=" $(printf '%q' "$t")"; done; printf '%s' "${out# }"
}

say "GLM-5.2 FULL 100K/nvfp4 CUDAGRAPH launch: $NNODES nodes, head=$HEAD:$PORT, image=$IMAGE"
[ "$DRYRUN" = 1 ] && echo "   (dry-run)"
for ((rank=1; rank<NNODES; rank++)); do
  w="${NODES[$rank]}"; run="$(docker_run_cmd "$rank" 1)"; shell="docker rm -f $NAME 2>/dev/null; $run"
  if [ "$DRYRUN" = 1 ]; then printf '\n# worker %s rank %d\n%s\n' "$w" "$rank" "$shell"
  else printf '   worker %s rank=%d\n' "$w" "$rank"; runon "$w" "$shell" || die "worker launch failed on $w"; fi
done
run="$(docker_run_cmd 0 0)"; shell="docker rm -f $NAME 2>/dev/null; $run"
if [ "$DRYRUN" = 1 ]; then printf '\n# head %s rank 0\n%s\n' "$HEAD" "$shell"; exit 0; fi
printf '   head %s rank=0\n' "$HEAD"; runon "$HEAD" "$shell" || die "head launch failed"
say "launched — poll: curl -s localhost:$PORT/v1/models ; logs: docker logs -f $NAME"
