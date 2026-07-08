# GLM-5.2 (753B, 4-bit) on 4× DGX Spark — 70.8% on Terminal-Bench 2.1

Full GLM-5.2 (753B MoE) quantized to **Int4-Int8Mix + NVFP4 4-bit KV cache**, served with
vLLM at **TP=4 across 4× NVIDIA DGX Spark (GB10)** at **100K context**, and benchmarked on
**Terminal-Bench 2.1** with the **Terminus-2** agent (same scaffold as the official numbers).

| | Official (full precision) | This repo (4-bit, 4× GB10) |
|---|---|---|
| Terminal-Bench 2.1 | **81.0%** | **70.8%** (63/89, pass@1) |
| Agent | Terminus-2 | Terminus-2 |
| Context | 256K | 100K |
| Decode | — | ~27.5 tok/s (MTP + CUDA graphs) |

"Clean" score excluding 2 tasks the harness can't start (`qemu-*` tmux failures, model never
attempts them): **72.4% (63/87)**. Single pass@1 run — 95% CI ≈ ±9 pts. The ~10-pt gap
bundles quantization **+** the 100K-vs-256K context cap **+** a smaller token budget; this
repo makes no isolated-quant claim. Per-task outcomes: [`results/RESULTS.md`](results/RESULTS.md).

## The rig

- 4× DGX Spark / GB10 — sm_121a, 128 GB unified memory each, ~273 GB/s, ConnectX-7 on a
  100 G RoCE fabric (MikroTik CRS504), TP=4.
- Weights: `QuantTrio/GLM-5.2-Int4-Int8Mix` (experts 4-bit → Marlin MoE, attention 8-bit),
  ~378 GB on disk, staged locally on every node (`/opt/glm52/hub/glm52-int4-int8mix`).
- vLLM rebuilt for sm_121a: portable Triton sparse-MLA kernels + DeepGEMM bypass (upstream
  DeepGEMM rejects sm_121). Image referenced in the launch script as
  `vllm-glm52-recon-nvfp4:latest`.
- KV cache `nvfp4_ds_mla` (4-bit) — the thing that unlocks 100K context; fp8 KV tops out
  ~64K on this hardware.
- MTP speculative decode (depth 3, in-checkpoint draft head) + `--compilation-config
  '{"cudagraph_mode":"FULL"}'` → ~27.5 tok/s. Eager is ~17–21 tok/s but keeps ~6 GB free
  instead of ~2.5 GB.

## Repo layout

```
scripts/glm52-launch-full-100k-cudagraph.sh   # 4-node TP=4 launcher (workers → head), all flags
bench/tb-run.sh                               # Harbor + Terminus-2 launcher (resumable)
bench/harbor-watchdog.sh                      # cron self-healer for the multi-day run
bench/status.sh                               # one-line progress probe (done/err/pass/fail)
audit/tb-final-audit.py                       # errored-vs-failed classifier (see below)
results/RESULTS.md                            # per-task PASS/FAIL lists
```

Replace `<HEAD_IP>` in `bench/*` with the address your bench box uses to reach the head node.
`10.78.0.x` in the launch script is the cluster-internal RoCE subnet — adapt to yours.

## Running the benchmark

On any box that can reach the endpoint (we used a $6 DigitalOcean droplet over Tailscale):

```bash
uv tool install harbor        # harbor 0.17.1
bash bench/tb-run.sh          # 89 tasks, pass@1, resumable; ~3 days at ~27 tok/s
crontab -e                    # */5 * * * * bash /root/tb/harbor-watchdog.sh
bash bench/status.sh          # progress one-liner
```

`tb-run.sh` is resume-safe: it kills a stale harbor, drops the one incomplete trial dir
(no `result.json`), and re-invokes with the same `--job-name` — Harbor picks up exactly
where it left off. This also survives pausing the run or endpoint crashes.

## Getting an honest number (audit + re-run)

Terminal-Bench separates **errored** (no verdict) from **failed** (model tried, verifier
said no). Infra flakes — endpoint crashes, connection errors, harness bugs — land in
*errored* and silently deflate your pass-rate if you ignore the distinction.

```bash
python3 audit/tb-final-audit.py <job-name>    # classifies errored trials by exception_info
```

Then re-run only the infra-errored tasks. Harbor refuses to resume with a changed config
(no `-x` mid-job), so the clean trick is:

```bash
cp tb-jobs/<job>/result.json tb-jobs/<job>/result.json.raw   # back up the raw score
rm -rf tb-jobs/<job>/<task>__*                                # drop errored trial dirs
bash bench/tb-run.sh                                          # same config → re-runs exactly those
```

In our run this converted one ambiguous connection-error (`extract-elf`) into a genuine
FAIL, and confirmed the 2 `qemu-*` errors as reproducible harness failures ("Failed to
start tmux session") — excluded from the clean denominator, kept in the raw one.

## Operational gotchas (all hit live)

1. **Unified memory:** raising `gpu-memory-utilization` leaves *less* free system RAM.
   gmu 0.83 dies with "No available memory for cache blocks"; 0.90 boots and holds with
   ~2–2.5 GB free. The 200K/fp8-KV reference config hard-wedged all 4 nodes (SSH-dead,
   power-cycle) — don't assume a datacenter recipe fits a unified-memory box.
2. **Engine crashes:** two request-triggered vLLM engine deaths over 72.5 h (scheduler
   `KeyError` → `EngineDeadError`; `RuntimeError: cancelled`). Neither OOM nor NCCL.
3. **Partial death is the trap:** rank-0 died, 3 workers stayed up holding ~115 GB each.
   `docker rm -f` (SIGKILL) wedges the next launch — `docker stop` (SIGTERM) each worker,
   wait for memory to actually free to single-digit GB, then relaunch (~9 min total).
   The launcher's `--stop` mode does this order for you.
4. **Never trust a benchmark that ran through a crash** — audit the errored bucket (above).

## Credits

- Weights: [QuantTrio/GLM-5.2-Int4-Int8Mix](https://huggingface.co/QuantTrio/GLM-5.2-Int4-Int8Mix)
- sm_121a vLLM recon (Triton sparse-MLA + DeepGEMM bypass) builds on CosmicRaisins' GB10 work
- Benchmark: [Terminal-Bench 2.1](https://www.tbench.ai/) via Harbor, Terminus-2 agent
- Official GLM-5.2 numbers: Z.ai release blog (Terminus-2 @256K = 81.0)

License: MIT for the scripts in this repo. Model weights under their own licenses.
