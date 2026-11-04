#!/bin/bash
# =============================================================================
# Same scenario, mooncake against BFF, with the best settings found so far.
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
#   DRY_RUN=1 ./run_best_compare.sh                     # print the plan, run nothing
#   ./run_best_compare.sh                               # 121 prompts >= 4096 tokens, 1024 out
#   SCENARIO=200 ./run_best_compare.sh                  # 200 prompts >= 1000 tokens, 2000 out
#   REF_DECODE_LOG=<a BFF run>/logs_latest/decode-0.txt ./run_best_compare.sh
#                                                       # limits from that log's max running
#   DECODE_LIMITS="7 8 9 48" ./run_best_compare.sh      # or name them yourself
#   SWEEP_DIR=<folder> ./run_best_compare.sh            # resume
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
case "$SCENARIO" in
  121) MIN_TOKENS=${MIN_TOKENS:-4096}; MAX_TOKENS=${MAX_TOKENS:-1024}; NUM_PROMPTS=${NUM_PROMPTS:-121}
       DEFAULT_LIMITS="8 9 ${MAX_CONCURRENCY}" ;;
  200) MIN_TOKENS=${MIN_TOKENS:-1000}; MAX_IN_TOKENS=${MAX_IN_TOKENS:-2000}; MAX_TOKENS=${MAX_TOKENS:-2000}; NUM_PROMPTS=${NUM_PROMPTS:-200}
       DEFAULT_LIMITS="16 18 22 ${MAX_CONCURRENCY}" ;;
  *) echo "Unknown SCENARIO=$SCENARIO (use 121 or 200)"; exit 1 ;;
esac

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

COMMON=(MODEL="$MODEL" TP_SIZE="$TP_SIZE" MIN_TOKENS="$MIN_TOKENS" MAX_IN_TOKENS="${MAX_IN_TOKENS:-0}" MAX_TOKENS="$MAX_TOKENS"
        NUM_PROMPTS="$NUM_PROMPTS" MAX_CONCURRENCY="$MAX_CONCURRENCY"
        CLIENT_MAX_CONCURRENCY="$MAX_CONCURRENCY" DECODE_MAX_NUM_BATCHED_TOKENS=64 USE_ASCEND_STORE=0)
BFF_ARGS=(BASELINE=bff_pull_v2 BFF_LSH_EXACT=1 BFF_MAX_REL_ERR=0.5 BFF_THRESHOLD=0.75 BFF_EDGE_LAYERS=20
          BFF_V2_MATERIALIZE=1 BFF_FREE_FLUSH=recv BFF_V2_ALLOW_FULL_GRAPH=1 BFF_V2_DEDUP=1
          BFF_V2_SIG_PIGGYBACK=1 BFF_V2_FREE_AT_PROMOTION=1 BFF_ALIAS_FUSED=1
          BFF_V2_CLIP_FRAC=${BFF_V2_CLIP_FRAC:-0.4} BFF_STEP_PROFILE=0)
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
  { echo "arm=$arm"; echo "decode_limit=$limit"; echo "stage=$stage"; echo "scenario=$SCENARIO"; printf '%s\n' "${COMMON[@]}" "${args[@]}"
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
  touch "${dest}/DONE"
  echo "=== $name: finished $(date '+%H:%M:%S'), saved to $dest"
}

compare() { (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python -m f1_benchmark.compare_runs "$SWEEP_DIR" "$@"); }

n_limits=$(wc -w <<< "$DECODE_LIMITS")
echo "Scenario $SCENARIO: $NUM_PROMPTS prompts >= $MIN_TOKENS tokens, $MAX_TOKENS out, client $MAX_CONCURRENCY"
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
