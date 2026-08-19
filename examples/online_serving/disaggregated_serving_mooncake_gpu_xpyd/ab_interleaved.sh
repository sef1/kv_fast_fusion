#!/bin/bash
# =============================================================================
# Interleaved A/B driver — settles whether the shared-core extraction cost throughput.
# =============================================================================
# Runs   legacy, v2, legacy, v2, …   alternating, NOT all-of-one-then-all-of-the-other.
#
# Interleaving is the whole point. The measured within-arm spread for the current connector is
# 1.045–1.171 req/s (n=5, ~12%), and the effect under test is ~5%. Running one arm to completion
# and then the other confounds the comparison with whatever else happens on this shared box over
# the intervening hour — a slow patch landing entirely inside one arm invents a difference, or
# hides a real one. Alternating splits that drift evenly across both arms.
#
# Usage:
#   ./ab_interleaved.sh              # 3 repeats per arm (~50 min)
#   REPEATS=2 ./ab_interleaved.sh    # fewer
#   ARMS="bff_v2_legacy bff_v2 vanilla" ./ab_interleaved.sh
#
# Results land in f1_results/f1_<tag>_r<N>.json — one file per arm per repeat, because RUN_REPEAT
# is threaded into RUN_TAG. Summarise with ./ab_summarise.py when it finishes.
# =============================================================================
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

REPEATS=${REPEATS:-3}
# Each arm is  <BASELINE>[:VAR=VAL[,VAR=VAL...]]  — the optional suffix is env applied to that arm
# only. Needed for arms that differ by a knob rather than by connector, e.g. dedup-off:
#   ARMS="bff_v2_legacy bff_v2 bff_v2:BFF_V2_DEDUP=0"
# RUN_TAG already encodes BFF_V2_DEDUP/BFF_V2_RESIDENT, so such arms separate into their own files
# without further help; anything that does NOT appear in the tag would collide, so check first.
ARMS=${ARMS:-"bff_v2_legacy bff_v2"}

# Everything that must be IDENTICAL across arms. Exported so the inner script cannot pick up a
# different default for one arm and not the other — the comparison is worthless if it does.
export BFF_THRESHOLD=${BFF_THRESHOLD:-0.8}
export BFF_MAX_REL_ERR=${BFF_MAX_REL_ERR:-0.3}
export BFF_PD_CROSS_INDEX=${BFF_PD_CROSS_INDEX:-lsh}
export NUM_PREFILL=${NUM_PREFILL:-1}
export NUM_DECODE=${NUM_DECODE:-1}
export MAX_CONCURRENCY=${MAX_CONCURRENCY:-150}
export NUM_PROMPTS=${NUM_PROMPTS:-500}

echo "Interleaved A/B: arms=[$ARMS] repeats=$REPEATS"
echo "  thr=$BFF_THRESHOLD rel_err=$BFF_MAX_REL_ERR conc=$MAX_CONCURRENCY n=$NUM_PROMPTS"
echo "  order is alternating; do not reorder into blocks."
echo ""

started=$(date +%s)
for r in $(seq 1 "$REPEATS"); do
    for spec in $ARMS; do
        arm=${spec%%:*}                       # BASELINE
        extra=${spec#"$arm"}; extra=${extra#:} # optional VAR=VAL,VAR=VAL
        env_args=()
        if [[ -n "$extra" ]]; then
            IFS=',' read -ra kvs <<< "$extra"
            env_args=("${kvs[@]}")
        fi
        echo "=============================================================="
        echo "  repeat $r/$REPEATS   arm=$arm ${env_args[*]}  ($(date +%H:%M:%S))"
        echo "=============================================================="
        env "${env_args[@]}" RUN_REPEAT="$r" BASELINE="$arm" ./disagg_bff_mooncake_gpu.sh
        rc=$?
        # Keep going on failure: a lost cell is recoverable, an aborted sweep costs the whole hour.
        # It is reported at the end so a partial matrix is never mistaken for a complete one.
        if [[ $rc -ne 0 ]]; then
            echo "  !! arm=$spec repeat=$r exited $rc — continuing"
            echo "$spec r$r rc=$rc" >> ab_failures.txt
        fi
        sleep 10   # let the driver release GPU memory before the next launch
    done
done

echo ""
echo "Done in $(( ($(date +%s) - started) / 60 )) min."
[[ -s ab_failures.txt ]] && { echo "FAILED CELLS:"; cat ab_failures.txt; }
echo "Summarise with: ./ab_summarise.py"
