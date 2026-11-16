#!/bin/bash
# Box session after the last-block fix (commit cde133ed29 or later). A group that fails stops itself; the
# next one still runs. Every run's folder lands under sweeps/session_2026_11_16/.
#
#   1. families_mixed       BFF x2, limit 32   no garbled short answers any more; does +6.8% hold
#   2. families             BFF x2, limit 24   does +12.8% hold without last-block sharing
#   3. families_unsorted    reordered, 3+3     the reordering test again, with the fixed tokenizer
#
# Groups 1 and 2 reuse saved baselines (the fix does not touch the baseline arm): 1 against the two
# session_2026_11_15/mixed_grouped baselines, 2 against the three families baselines at limit 24.
# Group 3 needs its own: reordering changes both arms.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
S=$PWD/sweeps/session_2026_11_16
BFF=(BFF_V2_CLIP_FRAC=0.7 BFF_EDGE_LAYERS=10 BFF_EDGE_SPLIT=3 BFF_EDGE_AT_END=1)

echo "=== 1/3 families_mixed, BFF only $(date '+%H:%M')"
env "${BFF[@]}" SCENARIO=families_mixed SWEEP_DIR=$S/mixed_grouped_fix \
  RUNS="bff:32:end bff:32:end" ./run_best_compare.sh

echo "=== 2/3 families, BFF only $(date '+%H:%M')"
env "${BFF[@]}" SCENARIO=families SWEEP_DIR=$S/families_fix \
  RUNS="bff:24:end bff:24:end" ./run_best_compare.sh

echo "=== 3/3 families_unsorted with reordering $(date '+%H:%M')"
env "${BFF[@]}" PROXY_REORDER=1 SCENARIO=families_unsorted SWEEP_DIR=$S/unsorted_reorder \
  RUNS="baseline:24 bff:24:end baseline:24 bff:24:end baseline:24 bff:24:end" ./run_best_compare.sh

echo "=== session done $(date '+%H:%M'): zip sweeps/session_2026_11_16"
