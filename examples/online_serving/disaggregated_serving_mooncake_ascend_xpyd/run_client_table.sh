#!/bin/bash
# =============================================================================
# Collect every run behind the client table, then print the table.
#
#   ./run_client_table.sh                  # runs the next BATCH runs, then stops
#   DRY_RUN=1 ./run_client_table.sh        # the whole plan, run nothing
#
# Re-run the same command to continue: finished runs are skipped. It stops after BATCH runs because a
# container has hung at startup after several runs in a row, so restart the container between batches.
# A failed run stops it too, and the next call tries that run again.
#
# Plan, each arm REPEATS times, one whole repeat of everything before the next:
#   families  normal and fixed length   baseline, clip 0.7, no cap
#   121, 200  normal                    baseline, no cap
#
#   TABLE_DIR=<folder>     where runs are saved (default sweeps/client_table)
#   REPEATS=2              fewer repeats
#   ONLY="families_normal" only these cells: families_normal families_fixed 121_normal 200_normal
#   BATCH=4  FIXED_LEN=600
# =============================================================================
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
TABLE_DIR=${TABLE_DIR:-${SCRIPT_DIR}/sweeps/client_table}
REPEATS=${REPEATS:-3}
BATCH=${BATCH:-4}
FIXED_LEN=${FIXED_LEN:-600}
ONLY=${ONLY:-}
DRY_RUN=${DRY_RUN:-0}

# cell | scenario | decode limit | fixed length? | arms. The limits are above what KV can hold, so KV decides.
CELLS=("families_normal families 24 0 baseline clip07 nocap"
       "families_fixed families 24 1 baseline clip07 nocap"
       "121_normal 121 24 0 baseline nocap"
       "200_normal 200 32 0 baseline nocap")

arm_env() {
  case "$1" in
    clip07) echo "BFF_V2_CLIP_FRAC=0.7" ;;
    nocap)  echo "BFF_V2_MAX_REQ_DECLINE=1" ;;
  esac
}

show_table() {
  if [[ "$DRY_RUN" != "1" ]]; then
    (cd "$REPO_ROOT" && PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}" python -m f1_benchmark.client_table "$TABLE_DIR")
  fi
}

done_count=0; left=0; ran=0
for ((repeat = 1; repeat <= REPEATS; repeat++)); do
  for cell_spec in "${CELLS[@]}"; do
    read -r cell scenario limit fixed arms <<< "$cell_spec"
    if [[ -n "$ONLY" && " $ONLY " != *" $cell "* ]]; then continue; fi
    read -ra arm_list <<< "$arms"
    # Reverse the order on even repeats, so no arm always runs first.
    if ((repeat % 2 == 0)); then
      reversed=(); for ((i = ${#arm_list[@]} - 1; i >= 0; i--)); do reversed+=("${arm_list[i]}"); done
      arm_list=("${reversed[@]}")
    fi
    for arm in "${arm_list[@]}"; do
      name="${arm}_r${repeat}"
      dir="${TABLE_DIR}/${cell}"
      kind=$([[ "$arm" == "baseline" ]] && echo baseline || echo bff)
      if [[ -f "${dir}/${name}/DONE" ]]; then done_count=$((done_count + 1)); continue; fi
      left=$((left + 1))
      settings=(SCENARIO="$scenario" RUNS="${kind}:${limit}:${name}" SWEEP_DIR="$dir" THROUGHPUT_ONLY="$fixed")
      if [[ "$fixed" == "1" ]]; then settings+=(FIXED_OUTPUT_LEN="$FIXED_LEN"); fi
      extra=$(arm_env "$arm")
      if [[ -n "$extra" ]]; then settings+=("$extra"); fi
      if [[ "$DRY_RUN" == "1" ]]; then
        echo "  ${cell}/${name}: ${settings[*]}"
        continue
      fi
      if ((ran >= BATCH)); then continue; fi
      echo "=== run $((ran + 1)) of this batch: ${cell}/${name}"
      if ! env "${settings[@]}" bash "${SCRIPT_DIR}/run_best_compare.sh"; then
        echo "=== ${cell}/${name} FAILED. See its console.txt. Re-run this command to try it again."
        show_table
        exit 1
      fi
      ran=$((ran + 1)); left=$((left - 1)); done_count=$((done_count + 1))
    done
  done
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Plan: ${done_count} done, ${left} to run, saved under ${TABLE_DIR}"
  exit 0
fi
echo "=== ${done_count} runs done, ${left} left, saved under ${TABLE_DIR}"
if ((left > 0)); then echo "=== Restart the container, then run the same command again."; fi
show_table
