#!/usr/bin/env python3
"""Paper run suite for single-GPU BFF: repeats, concurrency sweep (Fig 2), B13 ablation.

Runs every job through run_single_bff.sh on whichever GPU is idle (memory.used below a bar at launch
time), one job per GPU, and retries a job once if its server dies or no result JSON appears. Writes a
manifest (commit, diff, env, job list) and a suite log. Results land in

  results/<suite>/{repeats,conc_sweep,ablation}/   f1_<tag>.json, server_<tag>.log, f1_<tag>.log,
                                                   raw/<tag>/raw_outputs.json, stdout/<tag>.out

Usage:  python3 run_paper_suite.py [--suite paper_20261125] [--gpus 1,2,3,4,5,6,7] [--only repeats]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "../../.."))

MAGICODER = dict(F1_DATASET="ise-uiuc/Magicoder-Evol-Instruct-110K", F1_SPLIT="train",
                 F1_INPUT_KEY="instruction", F1_OUTPUT_KEY="response", HF_DATASETS_OFFLINE="1",
                 MIN_TOKENS="1536", MAX_IN_TOKENS="6000", MAX_TOKENS="1024", ASYNC_SCHED="1")

ARMS = {
    "vanilla":    dict(ARM="vanilla"),
    "fuse0":      dict(ARM="fuse0"),
    "fuse1":      dict(ARM="fuse1"),                                            # per-step nr_tree, raw
    "online085":  dict(ARM="fuse1", BFF_EXTRA_ENV="BFF_ONLINE=1 BFF_REG_THRESHOLD=0.85"),
    "online080":  dict(ARM="fuse1", BFF_EXTRA_ENV="BFF_ONLINE=1 BFF_REG_THRESHOLD=0.80"),
    "b13norm":    dict(ARM="fuse1", BFF_SCALE_MODE="norm"),                     # paper-literal Alg. 1
}


def jobs():
    out = []
    # 1) repeats at c128 (512 prompts): identical config, run-to-run (batching) variance
    for rep in (1, 2, 3):
        for arm in ("vanilla", "online085", "online080"):
            out.append(("repeats", f"rep{rep}_{arm}_c128",
                        dict(MAGICODER, **ARMS[arm], NUM_PROMPTS="512", MAX_CONCURRENCY="128",
                             SEED=str(1023 + rep))))
    # 2) Fig 2 concurrency sweep (1024 prompts)
    for con in (64, 96, 128, 160, 224, 256):
        for arm in ("vanilla", "fuse0", "fuse1", "online085", "online080"):
            out.append(("conc_sweep", f"sweep_{arm}_c{con}",
                        dict(MAGICODER, **ARMS[arm], NUM_PROMPTS="1024", MAX_CONCURRENCY=str(con))))
    # 3) ablation: B13 paper-literal merge (centroid written into rep + own-norm rescale) vs raw
    for arm in ("fuse1", "b13norm"):
        out.append(("ablation", f"ablate_{arm}_c128",
                    dict(MAGICODER, **ARMS[arm], NUM_PROMPTS="512", MAX_CONCURRENCY="128")))
    return out


def gpu_mem():
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout
    return {int(a): int(b) for a, b in (l.split(",") for l in q.strip().splitlines())}


def done_ok(subdir, tag):
    try:
        d = json.load(open(os.path.join(subdir, f"f1_{tag}.json")))
        return d.get("completed", 0) > 0 and "evaluation" in d
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="paper_" + datetime.now().strftime("%Y%m%d"))
    ap.add_argument("--gpus", default="1,2,3,4,5,6,7")
    ap.add_argument("--only", default="")
    ap.add_argument("--idle-mib", type=int, default=1500)
    a = ap.parse_args()
    root = os.path.join(HERE, "results", a.suite)
    os.makedirs(root, exist_ok=True)
    pool = [int(x) for x in a.gpus.split(",")]
    todo = [j for j in jobs() if not a.only or j[0] in a.only.split(",")]
    log = open(os.path.join(root, "suite.log"), "a", buffering=1)

    def say(msg):
        line = f"{datetime.now():%H:%M:%S} {msg}"
        print(line, flush=True); log.write(line + "\n")

    git = lambda *c: subprocess.run(["git", *c], cwd=REPO, capture_output=True, text=True).stdout
    with open(os.path.join(root, "uncommitted.diff"), "w") as f:
        f.write(git("diff", "HEAD"))
    man = dict(created=datetime.now().isoformat(), commit=git("rev-parse", "HEAD").strip(),
               branch=git("branch", "--show-current").strip(), host=os.uname().nodename,
               model="NousResearch/Meta-Llama-3.1-8B-Instruct", gpu="A100-PCIE-40GB",
               launcher="examples/online_serving/single_gpu_bff/run_single_bff.sh",
               script_defaults=dict(BLOCK_SIZE=16, MAX_MODEL_LEN=16384, MAX_NUM_BATCHED_TOKENS=16384,
                                    GPU_UTIL=0.90, BFF_MERGE="nr_tree", BFF_THRESHOLD=0.75,
                                    BFF_GROUP_SIZE=4, BFF_LSH_REPR="proj", cudagraph="FULL_AND_PIECEWISE"),
               jobs=[dict(group=g, tag=t, env=e) for g, t, e in todo])
    json.dump(man, open(os.path.join(root, "manifest.json"), "w"), indent=2)
    say(f"suite {a.suite}: {len(todo)} jobs, gpus {pool}, root {root}")

    queue = [(g, t, e, 0) for g, t, e in todo]
    running = {}  # gpu -> (proc, group, tag, env, tries, t0)
    while queue or running:
        for gpu, (p, g, t, e, tries, t0) in list(running.items()):
            if p.poll() is None:
                continue
            del running[gpu]
            sub = os.path.join(root, g)
            if done_ok(sub, t):
                say(f"done  {g}/{t} gpu{gpu} {(time.time()-t0)/60:.1f}min")
            elif tries < 1:
                say(f"FAIL  {g}/{t} gpu{gpu} rc={p.returncode} -> requeue")
                queue.append((g, t, e, tries + 1))
            else:
                say(f"FAIL  {g}/{t} gpu{gpu} rc={p.returncode} -> giving up")
        if queue:
            mem = gpu_mem()
            for gpu in pool:
                if not queue:
                    break
                if gpu in running or mem.get(gpu, 1 << 30) > a.idle_mib:
                    continue
                g, t, e, tries = queue.pop(0)
                sub = os.path.join(root, g)
                if done_ok(sub, t):
                    say(f"skip  {g}/{t} (already done)"); continue
                os.makedirs(os.path.join(sub, "stdout"), exist_ok=True)
                env = dict(os.environ, **e, GPU=str(gpu), PORT=str(8500 + gpu), RUN_TAG=t, RESULT_DIR=sub)
                out = open(os.path.join(sub, "stdout", f"{t}.out"), "w")
                p = subprocess.Popen(["bash", os.path.join(HERE, "run_single_bff.sh")], env=env,
                                     stdout=out, stderr=subprocess.STDOUT, cwd=HERE)
                running[gpu] = (p, g, t, e, tries, time.time())
                say(f"start {g}/{t} gpu{gpu} ({len(queue)} queued)")
                time.sleep(20)  # stagger startups (model load / compile contention)
                mem = gpu_mem()
        time.sleep(15)
    say("suite finished")


if __name__ == "__main__":
    sys.exit(main())
