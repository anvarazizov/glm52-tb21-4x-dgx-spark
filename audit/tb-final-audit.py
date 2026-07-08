#!/usr/bin/env python3
"""Final audit of a TB job: authoritative pass/fail from reward_stats, then classify
the ERRORED tasks (attempted, no final verdict) by exception_type from per-trial
result.json -> decide infra (re-run) vs genuine-slow/timeout."""
import json, os, glob, sys

JOB = sys.argv[1] if len(sys.argv) > 1 else "glm52-full-tb21-v1"
base = f"/root/tb/tb-jobs/{JOB}"
d = json.load(open(f"{base}/result.json"))
s = d["stats"]
ev = next(iter(s["evals"].values()))
rew = ev["reward_stats"]["reward"]
passed = [t.split("__")[0] for t in rew.get("1.0", [])]
failed = [t.split("__")[0] for t in rew.get("0.0", [])]
n_total = d.get("n_total_trials")

print(f"### {JOB}")
print(f"total_tasks(trials)={n_total} completed={s['n_completed_trials']} "
      f"errored={s['n_errored_trials']} retries={s['n_retries']}")
print(f"PASS={len(passed)}  FAIL={len(failed)}  (mean reward={ev['metrics'][0]['mean']:.3f})")

# errored tasks = attempted trial dirs whose task is NOT in passed/failed
verdict = set(passed) | set(failed)
INFRA = ("Connection", "InternalServerError", "EngineDead", "500", "ConnectError",
         "tmux", "Failed to start tmux", "APIConnection", "ServiceUnavailable",
         "RemoteProtocol", "cancelled", "ReadTimeout")
buckets = {"infra_rerun": [], "timeout": [], "other": []}
seen = set()
for td in sorted(glob.glob(f"{base}/*__*")):
    task = os.path.basename(td).split("__")[0]
    if task in verdict or task in seen:
        continue
    tp = os.path.join(td, "result.json")
    etype, emsg = "NO_RESULT(incomplete)", ""
    if os.path.exists(tp):
        try:
            t = json.load(open(tp))
            ei = t.get("exception_info") or {}
            etype = ei.get("exception_type", "unknown")
            emsg = (ei.get("exception_message", "") or "")[:90]
        except Exception as e:
            etype = f"parse_err:{e}"
    # also scan trial.log for infra signatures
    lp = os.path.join(td, "trial.log")
    loghit = ""
    if os.path.exists(lp):
        for l in open(lp, errors="replace"):
            if any(w in l for w in ("Connection error", "tmux", "InternalServerError",
                                    "Connection refused", "500 Internal")):
                loghit = l.strip()[:90]; break
    blob = f"{etype} {emsg} {loghit}"
    if any(w in blob for w in INFRA):
        buckets["infra_rerun"].append((task, etype, loghit or emsg))
    elif "Timeout" in etype:
        buckets["timeout"].append((task, etype, emsg))
    else:
        buckets["other"].append((task, etype, emsg))
    seen.add(task)

print(f"\n-- ERRORED tasks (not in pass/fail), classified --")
for b, items in buckets.items():
    print(f"\n[{b}] ({len(items)}):")
    for task, et, info in items:
        print(f"  {task:<34} {et:<22} {info}")

print(f"\n== RE-RUN CANDIDATES (infra + harness flakes) = "
      f"{len(buckets['infra_rerun'])} tasks ==")
print(f"== timeouts (re-run under no-contention to test slow-vs-artifact) = "
      f"{len(buckets['timeout'])} ==")
print(f"== raw pass = {len(passed)}/{len(passed)+len(failed)+sum(len(b) for b in buckets.values())} ==")
