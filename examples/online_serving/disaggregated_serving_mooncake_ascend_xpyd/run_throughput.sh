#!/usr/bin/env bash
# Throughput-only harness — measure the PURE concurrency/throughput potential of KV freeing, with accuracy
# DISREGARDED. It sets THROUGHPUT_ONLY=1 so run_benchmarks.sh:
#   * drops --compute-f1 / --compute-code-metrics (no accuracy scoring), and
#   * runs f1_main with --ignore-eos + --max-tokens=FIXED_OUTPUT_LEN, so EVERY request decodes exactly the
#     same number of tokens.
# Fixing the output length removes generation-length (rambling) as a confound: with quality irrelevant and
# work-per-request constant, any rps / output-tok-s difference between configs is the pure effect of KV
# freeing on concurrency and per-step cost. Everything else (cluster launch, BFF env, capacity/occupancy
# reports) is exactly run_benchmarks.sh — all its env knobs apply unchanged.
#
# Usage (A/B the freeing extremes at a fixed output length):
#   BASELINE=bff_pull_v2 BFF_V2_DEDUP=1 BFF_MAX_REL_ERR=0.2 NUM_PREFILL=1 NUM_DECODE=1 \
#     MAX_CONCURRENCY=512 FIXED_OUTPUT_LEN=1024 ./run_throughput.sh          # ~no freeing
#   BASELINE=bff_pull_v2 BFF_V2_DEDUP=1 BFF_MAX_REL_ERR=0.5 ... ./run_throughput.sh   # heavy freeing
#   BASELINE=bff_pull_v2 BFF_V2_DEDUP=0 ... ./run_throughput.sh                       # freeing off
# Read output_throughput_toks_s / request_throughput_rps + peak Running + occupancy from the logs; F1 is
# intentionally absent.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export THROUGHPUT_ONLY=1
export FIXED_OUTPUT_LEN="${FIXED_OUTPUT_LEN:-1024}"
# FREE_FLUSH (full-device npu.synchronize per block-free) only fixes the F1 write race, which this
# accuracy-disregarding harness does not measure — default it OFF so throughput reflects freeing's true
# potential, not the per-step sync. Override with BFF_FREE_FLUSH=1 to measure the sync's cost.
export BFF_FREE_FLUSH="${BFF_FREE_FLUSH:-0}"
exec bash "${here}/run_benchmarks.sh" "$@"
