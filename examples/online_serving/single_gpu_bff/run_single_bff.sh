#!/bin/bash
# Single-instance (one GPU, no P/D, no KV connector) BFF benchmark.
#
#   ARM=vanilla  stock `vllm serve` (single KV-cache group, no patch)
#   ARM=fuse0    BFF layout (warmup + fusion groups) but no merging  -> layout tax only
#   ARM=fuse1    BFF nr_tree fusion of co-finishing prefills, raw mode (redirect/ref-count/free)
#
# Examples:
#   ARM=vanilla ./run_single_bff.sh
#   ARM=fuse1 BFF_THRESHOLD=0.75 MAX_CONCURRENCY=64 ./run_single_bff.sh
#   ENFORCE_EAGER=1 NUM_PROMPTS=32 ARM=fuse1 ./run_single_bff.sh      # smoke test
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
cd "$SCRIPT_DIR"

ARM=${ARM:-fuse1}
MODEL=${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}
GPU=${GPU:-1}
PORT=${PORT:-8310}
GPU_UTIL=${GPU_UTIL:-0.90}
BLOCK_SIZE=${BLOCK_SIZE:-16}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-16384}
ENFORCE_EAGER=${ENFORCE_EAGER:-0}
ASYNC_SCHED=${ASYNC_SCHED:-0}            # legacy block-table update is not async-safe yet
PREFIX_CACHING=${PREFIX_CACHING:-1}
SEED=${SEED:-1024}

# BFF knobs (legacy single-instance runner)
BFF_MERGE=${BFF_MERGE:-nr_tree}
BFF_SCALE_MODE=${BFF_SCALE_MODE:-raw}
BFF_THRESHOLD=${BFF_THRESHOLD:-0.75}
BFF_GROUP_SIZE=${BFF_GROUP_SIZE:-4}
BFF_LSH_REPR=${BFF_LSH_REPR:-proj}
BFF_LSH_PROJ_DIM=${BFF_LSH_PROJ_DIM:-512}

# Client / dataset
F1_DATASET=${F1_DATASET:-$SCRIPT_DIR/data/openmath2_4k.jsonl}
F1_SPLIT=${F1_SPLIT:-train}
F1_INPUT_KEY=${F1_INPUT_KEY:-problem}
F1_OUTPUT_KEY=${F1_OUTPUT_KEY:-generated_solution}
NUM_PROMPTS=${NUM_PROMPTS:-256}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-64}
REQUEST_RATE=${REQUEST_RATE:-inf}
BURSTINESS=${BURSTINESS:-1.0}
MIN_TOKENS=${MIN_TOKENS:-0}
MAX_IN_TOKENS=${MAX_IN_TOKENS:-0}
MAX_TOKENS=${MAX_TOKENS:-1024}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1800}
EXTRA_CLIENT_ARGS=${EXTRA_CLIENT_ARGS:-}

RESULT_DIR=${RESULT_DIR:-$SCRIPT_DIR/f1_results}
mkdir -p "$RESULT_DIR"
MTAG=$(basename "$MODEL")
case "$ARM" in
  vanilla) RUN_TAG=${RUN_TAG:-${MTAG}_vanilla_bs${BLOCK_SIZE}_mnbt${MAX_NUM_BATCHED_TOKENS}_con${MAX_CONCURRENCY}_s${SEED}} ;;
  fuse0)   RUN_TAG=${RUN_TAG:-${MTAG}_fuse0_gs${BFF_GROUP_SIZE}_bs${BLOCK_SIZE}_mnbt${MAX_NUM_BATCHED_TOKENS}_con${MAX_CONCURRENCY}_s${SEED}} ;;
  fuse1)   RUN_TAG=${RUN_TAG:-${MTAG}_fuse1_${BFF_MERGE}_${BFF_SCALE_MODE}_${BFF_LSH_REPR}_thr${BFF_THRESHOLD}_gs${BFF_GROUP_SIZE}_bs${BLOCK_SIZE}_mnbt${MAX_NUM_BATCHED_TOKENS}_con${MAX_CONCURRENCY}_s${SEED}} ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac
SERVER_LOG="$RESULT_DIR/server_${RUN_TAG}.log"

SERVE_ARGS=(--host 0.0.0.0 --port "$PORT" --seed "$SEED" --trust-remote-code
  --block-size "$BLOCK_SIZE" --max-model-len "$MAX_MODEL_LEN"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" --max-num-seqs "$MAX_CONCURRENCY"
  --gpu-memory-utilization "$GPU_UTIL" --attention-backend FLASH_ATTN)
[[ "$PREFIX_CACHING" == "1" ]] && SERVE_ARGS+=(--enable-prefix-caching) || SERVE_ARGS+=(--no-enable-prefix-caching)
[[ "$ENFORCE_EAGER" == "1" ]] && SERVE_ARGS+=(--enforce-eager)
[[ "$ASYNC_SCHED" == "1" ]] && SERVE_ARGS+=(--async-scheduling) || SERVE_ARGS+=(--no-async-scheduling)
# PROFILE=1: torch-profile PROFILE_ITERS engine steps, starting PROFILE_DELAY s into the client run.
PROFILE=${PROFILE:-0}; PROFILE_DELAY=${PROFILE_DELAY:-60}; PROFILE_ITERS=${PROFILE_ITERS:-40}
PROFILE_DIR=${PROFILE_DIR:-$SCRIPT_DIR/f1_results/profile/$RUN_TAG}
if [[ "$PROFILE" == "1" ]]; then
  mkdir -p "$PROFILE_DIR"
  SERVE_ARGS+=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROFILE_DIR\",\"max_iterations\":$PROFILE_ITERS,\"ignore_frontend\":true}")
fi

ENV_ARGS=(CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}")
[[ -n "${BFF_SHARE_AUDIT:-}" ]] && ENV_ARGS+=(BFF_SHARE_AUDIT="$BFF_SHARE_AUDIT")
[[ -n "${BFF_EXTRA_ENV:-}" ]] && read -ra _extra <<< "$BFF_EXTRA_ENV" && ENV_ARGS+=("${_extra[@]}")
if [[ "$ARM" == "vanilla" ]]; then
  LAUNCHER=vllm.entrypoints.cli.main
else
  LAUNCHER=kv_fast_fusion.fast_fusion_main
  ENV_ARGS+=(BFF_SINGLE=1 BFF_PD_FUSE=0 BFF_MERGE="$BFF_MERGE" BFF_SCALE_MODE="$BFF_SCALE_MODE"
             BFF_THRESHOLD="$BFF_THRESHOLD" BFF_GROUP_SIZE="$BFF_GROUP_SIZE"
             BFF_LSH_REPR="$BFF_LSH_REPR" BFF_LSH_PROJ_DIM="$BFF_LSH_PROJ_DIM")
  [[ "$ARM" == "fuse0" ]] && ENV_ARGS+=(BFF_SINGLE_FUSE=0) || ENV_ARGS+=(BFF_SINGLE_FUSE=1)
fi

SERVER_PID=""
cleanup() { [[ -n "$SERVER_PID" ]] && kill -INT "$SERVER_PID" 2>/dev/null; sleep 15; [[ -n "$SERVER_PID" ]] && kill -9 "$SERVER_PID" 2>/dev/null; pkill -9 -P "$SERVER_PID" 2>/dev/null; }
trap cleanup EXIT INT TERM

echo "== single-GPU BFF | ARM=$ARM | tag=$RUN_TAG"
echo "   env: ${ENV_ARGS[*]}"
echo "   serve: ${SERVE_ARGS[*]}"
# PYSPY=1: sample the server and its engine-core subprocess (ptrace_scope=1 forbids attaching later).
PYSPY_OUT=${PYSPY_OUT:-$SCRIPT_DIR/f1_results/pyspy_${RUN_TAG}.txt}
PREFIX=()
[[ "${PYSPY:-0}" == "1" ]] && PREFIX=("$REPO_ROOT/.venv/bin/py-spy" record --subprocesses --idle --rate 200 --format raw -o "$PYSPY_OUT" --)
env "${ENV_ARGS[@]}" "${PREFIX[@]}" python3 -m $LAUNCHER serve "$MODEL" "${SERVE_ARGS[@]}" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

echo -n "   waiting for server"
for i in $(seq 1 600); do
  if curl -s "localhost:$PORT/v1/models" >/dev/null 2>&1; then echo " up (${i}s)"; break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo " DIED"; tail -40 "$SERVER_LOG"; exit 1; fi
  sleep 1; echo -n "."
done

# Engagement check (diagnostics are silent when inert — verify before trusting a run).
if [[ "$ARM" != "vanilla" ]]; then
  grep -q "single-instance (graph) patch applied" "$SERVER_LOG" || { echo "   ERROR: BFF single patch NOT applied"; exit 1; }
  echo "   BFF single patch engaged"
fi

cd "$REPO_ROOT"; mkdir -p "$RESULT_DIR/raw/$RUN_TAG"
if [[ "$PROFILE" == "1" ]]; then
  ( sleep "$PROFILE_DELAY"; curl -s -X POST "localhost:$PORT/start_profile" >/dev/null; echo "   [profile] started";
    sleep 45; curl -s -X POST "localhost:$PORT/stop_profile" >/dev/null; echo "   [profile] stopped -> $PROFILE_DIR" ) &
fi
python3 -m f1_benchmark.f1_main \
  --model "$MODEL" --host localhost --port "$PORT" \
  --dataset-path "$F1_DATASET" --hf-split "$F1_SPLIT" \
  --input-key "$F1_INPUT_KEY" --output-key "$F1_OUTPUT_KEY" \
  --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONCURRENCY" \
  --request-rate "$REQUEST_RATE" --burstiness "$BURSTINESS" \
  --min-tokens "$MIN_TOKENS" --max-in-tokens "$MAX_IN_TOKENS" --max-tokens "$MAX_TOKENS" \
  --request-timeout "$REQUEST_TIMEOUT" --compute-f1 --disable-tqdm \
  --result-dir "$RESULT_DIR/raw/$RUN_TAG" --result-file "$RESULT_DIR/raw/$RUN_TAG/summary.json" \
  --label "$RUN_TAG" $EXTRA_CLIENT_ARGS 2>&1 | tee "$RESULT_DIR/f1_${RUN_TAG}.log"
# raw_outputs.json lands next to the per-run summary (no clobbering across concurrent runs).
cp "$RESULT_DIR/raw/$RUN_TAG/summary.json" "$RESULT_DIR/f1_${RUN_TAG}.json" 2>/dev/null

# Post-run: merge BFF compression + scheduler stats + run config into the result JSON.
python3 - "$RESULT_DIR/f1_${RUN_TAG}.json" "$SERVER_LOG" "$ARM" <<'PY'
import json, os, re, sys
res, log, arm = sys.argv[1:4]
summ = re.compile(r"BFF summary: (\d+) reqs, (\d+) valid → (\d+) unique")
freed = re.compile(r"Block merging freed (-?\d+) blocks")
online = re.compile(r"BFF online \| calls=(\d+) \| blocks=(\d+) \| matched tree=(\d+) registry=(\d+) .*applied=(\d+) \| confirm ok=(\d+) rejected=(\d+)")
sched = re.compile(r"BFF sched \| step=\d+ \| running=(\d+) \| waiting=(\d+) \| free_blocks=(\d+) / (\d+) \| block_usage=([\d.]+)% \| preempt\(cum\)=(\d+)")
fusion_calls = reqs = valid = unique = freed_tot = 0
reqs_per_call = []; running = []; usage = []; preempt = 0; onl = None
for line in open(log, errors="replace"):
    if m := summ.search(line):
        fusion_calls += 1; r, v, u = map(int, m.groups()); reqs += r; valid += v; unique += u; reqs_per_call.append(r)
    elif m := freed.search(line):
        freed_tot += int(m.group(1))
    elif m := online.search(line):
        onl = list(map(int, m.groups()))
    elif m := sched.search(line):
        running.append(int(m.group(1))); usage.append(float(m.group(5))); preempt = int(m.group(6))
try:
    d = json.load(open(res))
except Exception:
    print("no result JSON"); sys.exit(0)
d["bff_single"] = {
    "arm": arm, "fusion_calls": fusion_calls, "reqs_fused": reqs,
    "mean_reqs_per_call": (sum(reqs_per_call) / len(reqs_per_call)) if reqs_per_call else 0,
    "fusion_layer_valid_blocks": valid, "fusion_layer_unique_blocks": unique,
    "fusion_layer_cr": (valid / unique) if unique else 1.0,
    "sched_blocks_freed": freed_tot,
    "running_mean": (sum(running) / len(running)) if running else None,
    "running_max": max(running) if running else None,
    "block_usage_mean": (sum(usage) / len(usage)) if usage else None,
    "preempt_cum": preempt,
}
if onl:
    c, blocks, mt, mr, app, ok, rej = onl
    d["bff_single"].update({"online_calls": c, "online_blocks": blocks, "online_matched_tree": mt,
        "online_matched_registry": mr, "online_match_frac": (mt + mr) / max(1, blocks),
        "online_cr": blocks / max(1, blocks - mt - mr), "online_applied": app,
        "online_confirm_ok": ok, "online_rejected": rej})
d["run_env"] = {k: v for k, v in os.environ.items() if k.startswith(("BFF_", "MODEL", "MAX_", "BLOCK_", "ARM", "SEED", "NUM_PROMPTS", "F1_", "ENFORCE", "ASYNC", "PREFIX", "GPU"))}
json.dump(d, open(res, "w"), indent=2)
print("bff_single:", json.dumps(d["bff_single"]))
print(f"  mean_f1: {d.get('evaluation', {}).get('mean_f1')}")
for k in ("output_throughput_toks_s", "request_throughput_rps", "mean_ttft_ms", "mean_itl_ms", "mean_tpot_ms", "finish_length_pct"):
    if k in d: print(f"  {k}: {d[k]}")
PY
