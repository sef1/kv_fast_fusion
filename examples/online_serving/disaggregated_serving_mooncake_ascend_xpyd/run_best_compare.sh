#!/bin/bash
# =============================================================================
# Same scenario, mooncake against BFF, with the best settings found so far. The one way to run tests.
#
# With RUNS set, it does exactly those runs and nothing else. Without it, it runs the full sweep:
#
#   1. Baseline (mooncakev1) and BFF at each decode limit in DECODE_LIMITS, alternating.
#   2. Each arm's best limit, chosen separately: if BFF evicts less, its best limit can be higher
#      than the baseline's, and forcing one limit on both would hide exactly that.
#   3. CONFIRM_PAIRS fresh baseline/BFF pairs, each arm at its own best limit, alternating. The
#      verdict comes from these: the fastest of several noisy runs flatters whichever got lucky, so
#      the runs that chose the limit are not used to judge it.
#
# Every run is copied to its own folder under SWEEP_DIR as soon as it ends (run_benchmarks.sh wipes
# logs_latest and results_latest when the next one starts), with its console output and settings.
# Re-running with the same SWEEP_DIR skips what already finished, so a stopped sweep resumes.
#
# The client always keeps MAX_CONCURRENCY requests in flight (CLIENT_MAX_CONCURRENCY); only decode
# admission is limited, so a queue still builds behind the limit.
#
#   RUNS="bff:48 baseline:8" ./run_best_compare.sh      # just these: arm:decode limit[:folder name]
#   RUNS="bff:48" BFF_ALIAS_FUSED=0 ./run_best_compare.sh   # any setting can be changed the same way
#   DRY_RUN=1 ./run_best_compare.sh                     # print the plan, run nothing
#   ./run_best_compare.sh                               # 121 prompts >= 4096 tokens, 1024 out
#   SCENARIO=200 ./run_best_compare.sh                  # 200 prompts >= 1000 tokens, 2000 out
#   SCENARIO=families RUNS="bff:48" ./run_best_compare.sh   # one project, many files: real prompts,
#                                                       # selected in dedup's favour, sent family by family
#   SCENARIO=longbench_e RUNS="bff:48" ./run_best_compare.sh   # LongBench-E 2K-4K, dataset order; build the
#                                                       # file first with f1_benchmark/make_longbench_e.py
#   REF_DECODE_LOG=<a BFF run>/logs_latest/decode-0.txt ./run_best_compare.sh
#                                                       # limits from that log's max running
#   DECODE_LIMITS="7 8 9 48" ./run_best_compare.sh      # or name them yourself
#   SWEEP_DIR=<folder> ./run_best_compare.sh            # resume
#
# Stops after a run whose prompts are not the scenario's usual ones, since nothing could be compared
# with it. Set EXPECT_FINGERPRINT= (empty) when changing the prompts on purpose.
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
RUN_SCRIPT=${RUN_SCRIPT:-${SCRIPT_DIR}/run_benchmarks.sh}
LOGS_LATEST=${LOGS_LATEST:-${SCRIPT_DIR}/logs_latest}
RESULTS_LATEST=${RESULTS_LATEST:-${SCRIPT_DIR}/results_latest}
DRY_RUN=${DRY_RUN:-0}
CONFIRM_PAIRS=${CONFIRM_PAIRS:-2}
SWEEP_DIR=${SWEEP_DIR:-${SCRIPT_DIR}/sweeps/$(date +%Y%m%d_%H%M%S)}

MODEL=${MODEL:-/models/glm/glm/GLM-5.1-W4A8-nq8}
TP_SIZE=${TP_SIZE:-8}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-48}
SCENARIO=${SCENARIO:-121}
DATASET_ARGS=()   # empty: run_benchmarks.sh's own dataset (Magicoder)
case "$SCENARIO" in
  121) MIN_TOKENS=${MIN_TOKENS:-4096}; MAX_IN_TOKENS=${MAX_IN_TOKENS:-6000}; MAX_TOKENS=${MAX_TOKENS:-1024}; NUM_PROMPTS=${NUM_PROMPTS:-121}
       USUAL_PROMPTS="4096 6000 121 0"; USUAL_FINGERPRINT=9f5e27b8
       DEFAULT_LIMITS="8 10 ${MAX_CONCURRENCY}" ;;
  200) MIN_TOKENS=${MIN_TOKENS:-1000}; MAX_IN_TOKENS=${MAX_IN_TOKENS:-2000}; MAX_TOKENS=${MAX_TOKENS:-2000}; NUM_PROMPTS=${NUM_PROMPTS:-200}
       USUAL_PROMPTS="1000 2000 200 0"; USUAL_FINGERPRINT=4da4b383
       DEFAULT_LIMITS="16 18 24 ${MAX_CONCURRENCY}" ;;
  families) MIN_TOKENS=${MIN_TOKENS:-2000}; MAX_IN_TOKENS=${MAX_IN_TOKENS:-4096}; MAX_TOKENS=${MAX_TOKENS:-1024}; NUM_PROMPTS=${NUM_PROMPTS:-219}
       PROMPT_FAMILIES=${PROMPT_FAMILIES:-1}
       DEFAULT_LIMITS="12 16 ${MAX_CONCURRENCY}"
       USUAL_PROMPTS="2000 4096 219 1"; USUAL_FINGERPRINT=872781bc ;;
  # LongBench-E, 2K-4K tokens, in the dataset's order: a candidate for ordinary traffic. Built once with
  # f1_benchmark/make_longbench_e.py, which already filtered the lengths, so no filter here. Its first
  # use is the BFF run's "match potential" lines, which measure the real signature match.
  longbench_e) MIN_TOKENS=${MIN_TOKENS:-0}; MAX_IN_TOKENS=${MAX_IN_TOKENS:-0}; MAX_TOKENS=${MAX_TOKENS:-1024}; NUM_PROMPTS=${NUM_PROMPTS:-289}
       F1_DATASET=${F1_DATASET:-${SCRIPT_DIR}/longbench_e_2k4k.jsonl}
       DATASET_ARGS=(F1_DATASET="$F1_DATASET" F1_INPUT_KEY=prompt F1_OUTPUT_KEY=reference F1_SPLIT=train)
       if [[ "${DRY_RUN:-0}" != "1" && ! -f "$F1_DATASET" ]]; then
         echo "No prompt file at $F1_DATASET. Build it with f1_benchmark/make_longbench_e.py, or set F1_DATASET."; exit 1
       fi
       DEFAULT_LIMITS="24 ${MAX_CONCURRENCY}"
       USUAL_PROMPTS="0 0 289 0"; USUAL_FINGERPRINT=aabe1d84 ;;   # the file as built 2026-11-08 (11 tasks)
  *) echo "Unknown SCENARIO=$SCENARIO (use 121, 200, families or longbench_e)"; exit 1 ;;
esac
MAX_IN_TOKENS=${MAX_IN_TOKENS:-0}   # upper bound on input length, 0 = none
PROMPT_FAMILIES=${PROMPT_FAMILIES:-0}
if [[ "$MIN_TOKENS $MAX_IN_TOKENS $NUM_PROMPTS $PROMPT_FAMILIES" != "$USUAL_PROMPTS" ]]; then USUAL_FINGERPRINT=""; fi
EXPECT_FINGERPRINT=${EXPECT_FINGERPRINT-$USUAL_FINGERPRINT}

# The largest "Running: N reqs" in a decode log, then that value minus one, plus no limit at all. The
# largest Running is what the run reached, so a limit at it changes little: the one below it is the one
# that actually stops the admission that leads to an eviction.
if [[ -n "${REF_DECODE_LOG:-}" && -z "${DECODE_LIMITS:-}" ]]; then
  REF_MAX=$(grep -oE 'Running: [0-9]+ reqs' "$REF_DECODE_LOG" | awk '{print $2}' | sort -n | tail -1)
  if [[ -z "$REF_MAX" ]]; then echo "No 'Running:' lines in $REF_DECODE_LOG"; exit 1; fi
  echo "Largest Running in $REF_DECODE_LOG: $REF_MAX"
  DECODE_LIMITS="$((REF_MAX - 1)) ${REF_MAX} ${MAX_CONCURRENCY}"
fi
DECODE_LIMITS=${DECODE_LIMITS:-$DEFAULT_LIMITS}

# Graphs are captured up to 2 x limit tokens (MTP uses 2 per request) from this list. A limit whose double is
# not in it falls back to eager once full: limit 9 ran about 4x slower in both arms, 985 s against about 330 s.
check_limit() {
  local limit=$1 size
  [[ "${ALLOW_SLOW_LIMIT:-0}" == "1" || "$limit" -gt 32 ]] && return 0
  for size in 2 4 8 12 16 20 24 28 32 36 40 48 56 64; do
    [[ $((2 * limit)) -eq $size ]] && return 0
  done
  echo "Decode limit $limit runs about 4x slower when full: no graph for $((2 * limit)) tokens. Use 6, 8, 10, 12, 14, 16, 18,"
  echo "20, 24, 28, 32 or above 32, or set ALLOW_SLOW_LIMIT=1 to run it anyway."
  exit 1
}

# 1: an evicted answer is continued instead of restarted. Confirmed on the box; see proxy_continue_answer.py.
PROXY_CONTINUE_ANSWER=${PROXY_CONTINUE_ANSWER:-1}

COMMON=(${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"} MODEL="$MODEL" TP_SIZE="$TP_SIZE" MIN_TOKENS="$MIN_TOKENS" MAX_IN_TOKENS="$MAX_IN_TOKENS" PROMPT_FAMILIES="$PROMPT_FAMILIES" MAX_TOKENS="$MAX_TOKENS"
        PROXY_CONTINUE_ANSWER="$PROXY_CONTINUE_ANSWER"
        NUM_PROMPTS="$NUM_PROMPTS" MAX_CONCURRENCY="$MAX_CONCURRENCY"
        CLIENT_MAX_CONCURRENCY="${CLIENT_MAX_CONCURRENCY:-$MAX_CONCURRENCY}"
        DECODE_MAX_NUM_BATCHED_TOKENS="${DECODE_MAX_NUM_BATCHED_TOKENS:-64}" USE_ASCEND_STORE="${USE_ASCEND_STORE:-0}")
# The best settings. Each can be overridden from the command line.
BFF_ARGS=(BASELINE=bff_pull_v2 BFF_LSH_EXACT="${BFF_LSH_EXACT:-1}" BFF_MAX_REL_ERR="${BFF_MAX_REL_ERR:-0.5}"
          BFF_THRESHOLD="${BFF_THRESHOLD:-0.75}" BFF_EDGE_LAYERS="${BFF_EDGE_LAYERS:-20}"
          BFF_EDGE_SPLIT="${BFF_EDGE_SPLIT:-1}"
          BFF_V2_MATERIALIZE="${BFF_V2_MATERIALIZE:-1}" BFF_FREE_FLUSH="${BFF_FREE_FLUSH:-recv}"
          BFF_V2_ALLOW_FULL_GRAPH="${BFF_V2_ALLOW_FULL_GRAPH:-1}" BFF_V2_DEDUP="${BFF_V2_DEDUP:-1}"
          BFF_V2_SIG_PIGGYBACK="${BFF_V2_SIG_PIGGYBACK:-1}" BFF_V2_FREE_AT_PROMOTION="${BFF_V2_FREE_AT_PROMOTION:-1}"
          BFF_ALIAS_FUSED="${BFF_ALIAS_FUSED:-1}" BFF_V2_CLIP_FRAC="${BFF_V2_CLIP_FRAC:-0.4}"
          BFF_STEP_PROFILE="${BFF_STEP_PROFILE:-0}")
BASE_ARGS=(BASELINE=mooncakev1)

run_one() {   # run_one <folder name> <arm: baseline|bff> <decode limit> <stage: sweep|confirm>
  local name=$1 arm=$2 limit=$3 stage=$4 dest="${SWEEP_DIR}/$1" args
  if [[ "$arm" == "bff" ]]; then args=("${BFF_ARGS[@]}"); else args=("${BASE_ARGS[@]}"); fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [$name] env ${COMMON[*]} ${args[*]} DECODE_MAX_CONCURRENCY=$limit $RUN_SCRIPT"
    return 0
  fi
  if [[ -f "${dest}/DONE" ]]; then echo "=== $name: already done, skipping"; return 0; fi
  mkdir -p "$dest"
  # The inherited BFF_ settings first, so the ones this run really sets win when config.txt is read.
  { env | grep -E '^(BFF_|PROXY_CONTINUE_ANSWER=|THROUGHPUT_ONLY=|FIXED_OUTPUT_LEN=)' | sort; echo "arm=$arm"; echo "decode_limit=$limit"; echo "stage=$stage"; echo "scenario=$SCENARIO"
    printf '%s\n' "${COMMON[@]}" "${args[@]}"
    echo "git=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null)"; } > "${dest}/config.txt"
  echo "=== $name: $arm, decode limit $limit, started $(date '+%H:%M:%S')"
  env "${COMMON[@]}" "${args[@]}" DECODE_MAX_CONCURRENCY="$limit" "$RUN_SCRIPT" > "${dest}/console.txt" 2>&1
  local rc=$?
  rm -rf "${dest}/logs_latest" "${dest}/results_latest"
  cp -r "$LOGS_LATEST" "${dest}/logs_latest" 2>/dev/null
  cp -r "$RESULTS_LATEST" "${dest}/results_latest" 2>/dev/null
  # By its own last line, not its exit code: cleanup after a finished run may return non-zero.
  if ! grep -q "Benchmark done" "${dest}/console.txt"; then
    echo "=== $name FAILED (exit $rc). What it wrote is in $dest; see console.txt. Stopping."
    exit 1
  fi
  local got
  got=$(grep -rhoE 'prompt fingerprint: [0-9a-f]+' "${dest}/logs_latest" 2>/dev/null | head -1 | awk '{print $3}')
  if [[ -n "$EXPECT_FINGERPRINT" && -z "$got" ]]; then
    echo "=== $name: WARNING: no prompt fingerprint in its logs, so its prompts are unchecked."
  elif [[ -n "$EXPECT_FINGERPRINT" && "$got" != "$EXPECT_FINGERPRINT" ]]; then
    echo "=== $name used different prompts: fingerprint $got, expected $EXPECT_FINGERPRINT. Stopping, since it"
    echo "    cannot be compared with earlier runs. Saved in $dest. To allow this: EXPECT_FINGERPRINT="
    exit 1
  fi
  touch "${dest}/DONE"
  echo "=== $name: finished $(date '+%H:%M:%S'), saved to $dest"
}

compare() { (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python -m f1_benchmark.compare_runs "$SWEEP_DIR" "$@"); }

echo "Scenario $SCENARIO: $NUM_PROMPTS prompts >= $MIN_TOKENS tokens (max $MAX_IN_TOKENS, 0 = none)$([[ "$PROMPT_FAMILIES" == "1" ]] && echo ', families only, grouped'), $MAX_TOKENS out, client $MAX_CONCURRENCY"

if [[ -n "${RUNS:-}" ]]; then
  # Checked before anything runs, so a typo in the last one does not surface after the first has finished.
  for spec in $RUNS; do
    IFS=: read -r arm limit name <<< "$spec"
    if [[ ! "$arm" =~ ^(baseline|bff)$ || ! "$limit" =~ ^[0-9]+$ ]]; then
      echo "Bad entry in RUNS: '$spec'. Use arm:decode_limit or arm:decode_limit:name, with arm baseline or bff."
      exit 1
    fi
    check_limit "$limit"
  done
  echo "Runs: $RUNS | saved under $SWEEP_DIR"
  declare -A used=()
  for spec in $RUNS; do
    IFS=: read -r arm limit name <<< "$spec"
    name=${name:-${arm}_limit${limit}}
    used[$name]=$(( ${used[$name]:-0} + 1 ))
    if [[ "${used[$name]}" -gt 1 ]]; then name="${name}_${used[$name]}"; fi
    run_one "$name" "$arm" "$limit" sweep
  done
  if [[ "$DRY_RUN" != "1" ]]; then
    echo; echo "=== Comparison"; compare | tee "${SWEEP_DIR}/summary.txt"
    echo "Saved: ${SWEEP_DIR} (summary.txt, one folder per run)"
  fi
  exit 0
fi

for limit in $DECODE_LIMITS; do check_limit "$limit"; done
n_limits=$(wc -w <<< "$DECODE_LIMITS")
echo "Decode limits: $DECODE_LIMITS | runs: $((2 * n_limits + 2 * CONFIRM_PAIRS)) | saved under $SWEEP_DIR"

echo "--- Stage 1: both arms at each decode limit"
for limit in $DECODE_LIMITS; do
  run_one "sweep_baseline_limit${limit}" baseline "$limit" sweep
  run_one "sweep_bff_limit${limit}" bff "$limit" sweep
done

if [[ "$DRY_RUN" == "1" ]]; then
  BEST_BASE="<baseline best>"; BEST_BFF="<BFF best>"
else
  BEST_BASE=$(compare --best baseline); BEST_BFF=$(compare --best bff)
  if [[ -z "$BEST_BASE" || -z "$BEST_BFF" ]]; then
    echo "An arm has no run without garbage output; stopping before the confirmation runs."
    compare
    exit 1
  fi
  echo "--- Best limits: baseline $BEST_BASE, BFF $BEST_BFF"
  compare
fi

echo "--- Stage 2: $CONFIRM_PAIRS fresh pair(s), each arm at its own best limit"
for i in $(seq 1 "$CONFIRM_PAIRS"); do
  run_one "confirm${i}_bff_limit${BEST_BFF}" bff "$BEST_BFF" confirm
  run_one "confirm${i}_baseline_limit${BEST_BASE}" baseline "$BEST_BASE" confirm
done

if [[ "$DRY_RUN" != "1" ]]; then
  echo; echo "=== Final comparison"; compare | tee "${SWEEP_DIR}/summary.txt"
  echo "Saved: ${SWEEP_DIR} (summary.txt, one folder per run)"
fi
