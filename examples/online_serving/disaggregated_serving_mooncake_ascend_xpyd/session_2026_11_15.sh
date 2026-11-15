#!/bin/bash
# One box session, three experiments in priority order (commit f6aeb14faa or later). A group that fails stops
# itself; the next one still runs. Every run's folder lands under sweeps/session_2026_11_15/.
#
#   1. families_mixed           grouped families + 1 short request each, limit 32       sim: +8%
#   2. families_unsorted        reordered prefill (PROXY_REORDER=1), limit 24, 3+3     sim: +3% (box ~+1-2%)
#   3. families_mixed_unsorted  unsorted families + 1 short request each, limit 32     sim: ~0 (checks the sim)
#
# Baselines are in every group: each group changes the prompts or the order, so saved baselines do not apply.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
S=$PWD/sweeps/session_2026_11_15
BFF=(BFF_V2_CLIP_FRAC=0.7 BFF_EDGE_LAYERS=10 BFF_EDGE_SPLIT=3 BFF_EDGE_AT_END=1)

echo "=== 1/3 families_mixed $(date '+%H:%M')"
env "${BFF[@]}" SCENARIO=families_mixed SWEEP_DIR=$S/mixed_grouped \
  RUNS="baseline:32 bff:32:end baseline:32 bff:32:end" ./run_best_compare.sh

echo "=== 2/3 families_unsorted with reordering $(date '+%H:%M')"
env "${BFF[@]}" PROXY_REORDER=1 SCENARIO=families_unsorted SWEEP_DIR=$S/unsorted_reorder \
  RUNS="baseline:24 bff:24:end baseline:24 bff:24:end baseline:24 bff:24:end" ./run_best_compare.sh

echo "=== 3/3 families_mixed_unsorted $(date '+%H:%M')"
env "${BFF[@]}" SCENARIO=families_mixed_unsorted SWEEP_DIR=$S/mixed_unsorted \
  RUNS="baseline:32 bff:32:end baseline:32 bff:32:end" ./run_best_compare.sh

echo "=== session done $(date '+%H:%M'): zip sweeps/session_2026_11_15"
