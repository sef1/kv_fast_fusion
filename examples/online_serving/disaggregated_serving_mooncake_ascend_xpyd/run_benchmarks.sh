#!/bin/bash
# =============================================================================
# BFF (KV-Cache Fast Fusion) Disaggregated Serving — Ascend / Mooncake nPmD
# =============================================================================
# Ascend/NPU analogue of ../disaggregated_serving_p2p_nccl_xpyd/disagg_bff_p2p_nccl_xpyd.sh.
# Everything is env-overridable (see the block below); nothing host-specific is hard-coded.
#
# Servers launch via `kv_fast_fusion.fast_fusion_main serve` so importing the package runs the
# Ascend BFF patch (kv_fast_fusion_ascend) — it patches NPUModelRunner/scheduler and registers the
# group-aware `MooncakeLayerwiseConnectorFF`. The KV path is a MultiConnector wrapping the chosen
# Mooncake mover + AscendStoreConnector, exactly like the stock Ascend setup.
#
# BASELINE selects the mover:
#   bff        (default) → MooncakeLayerwiseConnectorFF + BFF_PD_FUSE=1 (fusion + D-side sharing)
#   bff_v2               → MooncakeLayerwiseConnectorFFv2: P ships per-block signatures, the DECODE
#                          replies with what it does not need, and those blocks are never written.
#                          Tune with BFF_THRESHOLD + BFF_MAX_REL_ERR (see the v2 knobs below).
#   bff_pull             → MooncakeConnectorFF: the NON-layerwise (pull) transport with BFF's
#                          multi-group KV layout and NO dedup. This is the transport the GPU
#                          numbers were measured on, so it is what makes NPU and GPU comparable.
#                          Its control arm is `mooncakev1` (same transport, no BFF), and the gate
#                          is ACCURACY, not throughput: a wrong per-group block table does not
#                          raise, it silently transfers the wrong KV. Compare CodeBLEU/N-gram
#                          against mooncakev1 before reading a single req/s figure.
#   bff_pull_v2          → MooncakeConnectorFFv2: the same pull transport with the merge decision
#                          moved to the DECODE. D asks P for signatures of the blocks it is about
#                          to read and simply does not read the ones it can satisfy locally — so
#                          unlike bff_pull (which saves KV capacity only) this saves wire bandwidth
#                          too, and needs no producer forward-path work. Its control arm is
#                          `bff_pull`; BFF_V2_DEDUP=0 is the within-arm ablation.
#   layerwise            → stock MooncakeLayerwiseConnector (layerwise transfer, no fusion)
#   mooncakev1           → stock MooncakeConnectorV1 (whole-request transfer)
#   vanilla              → true stock: launches via `vllm.entrypoints.cli.main`, so
#                          `kv_fast_fusion`/`kv_fast_fusion_ascend` are never imported (no patches,
#                          no FF connector registration at all). Connector choice via
#                          VANILLA_CONNECTOR=MooncakeLayerwiseConnector (default) | MooncakeConnectorV1.
#
# SSD offload (opt-in): ENABLE_SSD_OFFLOAD=1 spills the AscendStore KV pool to SSD via Mooncake. It
# only takes effect with USE_ASCEND_STORE=1 and requires the SSD patches applied on the host
# (ENABLE_SSD=1 <repo>/patch/apply_all.sh). Tune the path with SSD_OFFLOAD_PATH.
#
# Example runs:
#   ./run_benchmarks.sh                                   # 2P1D, bff, defaults
#   NUM_PREFILL=2 NUM_DECODE=1 BASELINE=bff BFF_THRESHOLD=0.85 ./run_benchmarks.sh
#   BASELINE=layerwise ./run_benchmarks.sh                # fusion ablation (BFF layout off entirely)
#   ENABLE_SSD_OFFLOAD=1 USE_ASCEND_STORE=1 BASELINE=layerwise ./run_benchmarks.sh   # KV pool → SSD
#   BASELINE=vanilla ./run_benchmarks.sh                  # true stock, no BFF code imported at all
#   ./run_benchmarks.sh -k                                # kill a previous cluster and exit
# =============================================================================

set -ex

# ============================================================
# Config — everything overridable via env (VAR=... ./run_benchmarks.sh)
# ============================================================
# ---- Model / topology ----
MODEL=${MODEL:-Qwen/Qwen2.5-7B}
NUM_PREFILL=${NUM_PREFILL:-2}
NUM_DECODE=${NUM_DECODE:-1}
TP_SIZE=${TP_SIZE:-1}                 # tensor-parallel size per instance (connector prefill/decode tp_size)
DP_SIZE=${DP_SIZE:-1}                 # data-parallel size per instance (connector prefill/decode dp_size)

# ---- Ports ----
MASTER_PORT=${MASTER_PORT:-57788}     # mooncake_master
PROXY_PORT=${PROXY_PORT:-8770}
PREFILL_PORT_BASE=${PREFILL_PORT_BASE:-8771}
DECODE_PORT_BASE=${DECODE_PORT_BASE:-8773}
PREFILL_KV_PORT_BASE=${PREFILL_KV_PORT_BASE:-31000}
DECODE_KV_PORT_BASE=${DECODE_KV_PORT_BASE:-30000}

# ---- Engine sizing ----
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-65536}   # was 64 in the original (a typo) — 64 starves prefill
MAX_NUM_SEQS=${MAX_NUM_SEQS:-48}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
BLOCK_SIZE=${BLOCK_SIZE:-128}         # BFF requires 128
SEED=${SEED:-1024}

# ---- Baseline / connector selection ----
BASELINE=${BASELINE:-bff}             # bff | layerwise | mooncakev1 | vanilla
VANILLA_CONNECTOR=${VANILLA_CONNECTOR:-MooncakeLayerwiseConnector}  # vanilla only: MooncakeLayerwiseConnector | MooncakeConnectorV1
case "$BASELINE" in
  bff)         CONNECTOR="MooncakeLayerwiseConnectorFF"; LAYER_WISE=true;  BFF_ON=1; LAUNCHER="kv_fast_fusion.fast_fusion_main" ;;
  bff_v2)      CONNECTOR="MooncakeLayerwiseConnectorFFv2"; LAYER_WISE=true; BFF_ON=1; LAUNCHER="kv_fast_fusion.fast_fusion_main" ;;
  layerwise)   CONNECTOR="MooncakeLayerwiseConnector";   LAYER_WISE=true;  BFF_ON=0; LAUNCHER="kv_fast_fusion.fast_fusion_main" ;;
  mooncakev1)  CONNECTOR="MooncakeConnectorV1";          LAYER_WISE=false; BFF_ON=0; LAUNCHER="kv_fast_fusion.fast_fusion_main" ;;
  bff_pull)    CONNECTOR="MooncakeConnectorFF";          LAYER_WISE=false; BFF_ON=1; LAUNCHER="kv_fast_fusion.fast_fusion_main" ;;
  bff_pull_v2) CONNECTOR="MooncakeConnectorFFv2";        LAYER_WISE=false; BFF_ON=1; LAUNCHER="kv_fast_fusion.fast_fusion_main" ;;
  vanilla)
    CONNECTOR="$VANILLA_CONNECTOR"
    [[ "$CONNECTOR" == "MooncakeLayerwiseConnector" ]] && LAYER_WISE=true || LAYER_WISE=false
    BFF_ON=0; LAUNCHER="vllm.entrypoints.cli.main" ;;
  *) echo "Unknown BASELINE=$BASELINE (use bff|bff_v2|bff_pull|bff_pull_v2|layerwise|mooncakev1|vanilla)"; exit 1 ;;
esac

# Wrap the mover in MultiConnector + AscendStoreConnector (external KV pool)?
# Default: OFF for bff, ON for the stock baselines. BFF needs HMA (multi-group), but the deployed
# AscendMultiConnector/AscendStoreConnector may not implement SupportsHMA — so bff runs the FF mover
# standalone (it IS HMA-capable). For an apples-to-apples comparison set USE_ASCEND_STORE=0 on the
# stock baselines too.
if [[ "$BASELINE" == "bff" || "$BASELINE" == "bff_v2" || "$BASELINE" == "bff_pull" \
      || "$BASELINE" == "bff_pull_v2" ]]; then
  USE_ASCEND_STORE=${USE_ASCEND_STORE:-0}
else
  USE_ASCEND_STORE=${USE_ASCEND_STORE:-1}
fi

# ---- BFF knobs (only take effect when BASELINE=bff) ----
BFF_PD_DEBUG=${BFF_PD_DEBUG:-0}            # 
BFF_FF_REP_SAFE=${BFF_FF_REP_SAFE:-1}            # rep-lifetime fix: resolve rep only from live state (default ON) 
BFF_FF_AUDIT=${BFF_FF_AUDIT:-1}            # 
# BFF_FF_GROUPS=${BFF_FF_GROUPS:-None}            # 
BFF_FF_RID_LIVE=${BFF_FF_RID_LIVE:-0}            # 
BFF_PD_FUSE=${BFF_PD_FUSE:-1}            # connector-level fusion + redirect propagation to D
BFF_SCALE_MODE=${BFF_SCALE_MODE:-raw}    # NPU supports raw only (ratio needs a CUDA Triton kernel)
BFF_PD_MERGE=${BFF_PD_MERGE:-cc}         # within-batch clustering: cc | nr_tree
BFF_PD_REPR=${BFF_PD_REPR:-proj}         # block repr for similarity: full | proj | mean
BFF_THRESHOLD=${BFF_THRESHOLD:-0.85}     # cosine merge threshold (0..1)
BFF_GROUP_SIZE=${BFF_GROUP_SIZE:-4}      # fusion layers packed per KV-cache group
BFF_PD_ENCODED_BATCH_SIZE=${BFF_PD_ENCODED_BATCH_SIZE:-8}   # cross-batch registry window (0=within-batch only)

# ---- v2 knobs (BASELINE=bff_v2 only) ----
# v2 moves the merge decision to the DECODE: P ships per-block signatures, D replies with the blocks
# it does not need, and those are never written. On GPU this took throughput 1.13 -> 1.42 req/s with
# ngram_match inside the undamaged band, at BFF_THRESHOLD=0.8 BFF_MAX_REL_ERR=0.3.
BFF_V2_DEDUP=${BFF_V2_DEDUP:-1}          # 0 disables the whole mechanism (signature exchange too)
BFF_V2_RESIDENT=${BFF_V2_RESIDENT:-1}    # alias to blocks left over from earlier transfers
BFF_SIG_DIM=${BFF_SIG_DIM:-128}          # signature width; ~256 B/block against ~1.6 MB of KV
BFF_V2_MAX_RESIDENT=${BFF_V2_MAX_RESIDENT:-32768}
BFF_V2_SIG_TIMEOUT=${BFF_V2_SIG_TIMEOUT:-2}   # no answer in this long -> send the group whole
# bff_pull_v2's own budget, and much larger than the layerwise one above on purpose: there the
# DECODE answers from a dict it already holds, here the PRODUCER has to gather blocks on the NPU and
# sync them to the host, on a side thread of a node saturated with prefill. The exchange runs on the
# recv thread, never the forward path, so a generous budget costs one request's KV latency while a
# tight one silently costs the whole optimisation.
BFF_PULL_V2_SIG_TIMEOUT=${BFF_PULL_V2_SIG_TIMEOUT:-10}
# Ceiling on how much of ONE request may be served from other requests' blocks. BFF_MAX_REL_ERR is a
# per-BLOCK bar; this is the per-REQUEST one, and they are not substitutes. At max_rel_err=0.3 the
# cosine floor is 0.954 — sane for one block, catastrophic for nineteen of twenty, because the model
# then attends to a coherent prompt that is not the one it was asked. 1.0 disables it.
BFF_V2_MAX_REQ_DECLINE=${BFF_V2_MAX_REQ_DECLINE:-0.5}
# Refuse to alias a block the decode has not finished writing (its last prompt block is partially
# filled). 1 = guard on and count; 0 = reproduce the old behaviour, still counted.
BFF_V2_PROTECT_HOT_BLOCKS=${BFF_V2_PROTECT_HOT_BLOCKS:-1}
# Ceiling on the relative substitution error a merge may inject:
#   rel_err = ||k_owner - k_rep|| / ||k_owner|| = sqrt(1 + r^2 - 2*r*cos),  r = |k_rep|/|k_owner|
# This, not BFF_THRESHOLD, governs accuracy — cosine is scale-free, so a pair can clear any cosine
# bar and still be a bad substitution. 1.0 is inert. Note min_r rel_err = sqrt(1-cos^2), so a 0.3
# budget implies cos >= 0.954 whatever the norms do.
BFF_MAX_REL_ERR=${BFF_MAX_REL_ERR:-1.0}
# Which layers of a fusion group feed the signature. "first" (default) uses only the group's first
# layer, which is all that exists when layerwise wants to send it, and preserves the compute/
# transfer overlap. "group" matches the GPU signature exactly but holds the group's transfer until
# its last layer is written.
BFF_SIG_LAYERS=${BFF_SIG_LAYERS:-first}
# What vLLM does when a connector reports blocks whose KV never arrived. v2 reports these BY DESIGN:
# a block the decode declined and could not then alias holds nothing, so it must be recomputed.
# vLLM's own default is "fail", which ERRORS those requests instead — turning the designed fallback
# into lost requests (and, because a request failed before promotion still carries the
# num_cached_tokens=-1 sentinel, into a negative Prometheus counter that takes down the API server's
# output handler). The GPU script has always passed "recompute"; this one silently inherited "fail".
KV_LOAD_FAILURE_POLICY=${KV_LOAD_FAILURE_POLICY:-recompute}

# ---- Benchmark knobs ----
NUM_PROMPTS=${NUM_PROMPTS:-512}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-530}
PREFILL_MAX_CONCURRENCY=${PREFILL_MAX_CONCURRENCY:-${MAX_CONCURRENCY}}
DECODE_MAX_CONCURRENCY=${DECODE_MAX_CONCURRENCY:-${MAX_CONCURRENCY}}
REQUEST_RATE=${REQUEST_RATE:-64}
BURSTINESS=${BURSTINESS:-0.1}
MIN_TOKENS=${MIN_TOKENS:-2048}
# Generation cap. NOT the same knob as MIN_TOKENS, which filters INPUT length. It decides which
# regime the benchmark measures: the GPU comparison runs used 1024, where ~72% of requests hit the
# cap, so generation length is pinned and the run measures pure token throughput. This default of
# 6000 caps only ~9%, which leaves length free — and therefore exposes any effect BFF has on when
# the model stops, which the GPU runs truncate away. Match it to 1024 to compare the two platforms.
MAX_TOKENS=${MAX_TOKENS:-6000}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-6000.0}
# F1_DATASET=${F1_DATASET:-m-a-p/CodeFeedback-Filtered-Instruction}
# F1_DATASET=${F1_DATASET:-codeparrot_f1_benchmark.jsonl}
F1_DATASET=${F1_DATASET:-ise-uiuc/Magicoder-Evol-Instruct-110K}

F1_SPLIT=${F1_SPLIT:-train}
F1_INPUT_KEY=${F1_INPUT_KEY:-instruction}
F1_OUTPUT_KEY=${F1_OUTPUT_KEY:-response}

# ---- SSD offload (opt-in; only meaningful when USE_ASCEND_STORE=1) ----
# Spills the AscendStore KV pool to SSD via Mooncake. Requires the SSD patches applied on the host:
#   ENABLE_SSD=1 <path>/vllm_ff/patch/apply_all.sh   (adds enable_ssd_offload to mooncake_backend.py)
ENABLE_SSD_OFFLOAD=${ENABLE_SSD_OFFLOAD:-0}
SSD_OFFLOAD_PATH=${SSD_OFFLOAD_PATH:-/data/mooncake_offload}   # ssd_offload_path + MOONCAKE_OFFLOAD_FILE_STORAGE_PATH

# ---- mooncake.json base fields (auto-generated unless MOONCAKE_CONFIG_PATH is overridden) ----
MC_METADATA_SERVER=${MC_METADATA_SERVER:-P2PHANDSHAKE}
MC_PROTOCOL=${MC_PROTOCOL:-ascend}
MC_DEVICE_NAME=${MC_DEVICE_NAME:-}
MC_MASTER_ADDR=${MC_MASTER_ADDR:-127.0.0.1}                    # master_server_address = MC_MASTER_ADDR:MASTER_PORT
MC_GLOBAL_SEGMENT_SIZE=${MC_GLOBAL_SEGMENT_SIZE:-100GB}
MC_LOCAL_BUFFER_SIZE=${MC_LOCAL_BUFFER_SIZE:-1GB}

# ---- MOONCAKE_OFFLOAD_* tunables (wiki defaults; only exported when ENABLE_SSD_OFFLOAD=1) ----
MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR=${MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR:-bucket_storage_backend}
MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY=${MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY:-lru}
MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES=${MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES:-8589934592}   # 8 GiB
MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS=${MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS:-3}
MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES=${MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES:-1649267441664}   # ~1.5 TiB
MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE=${MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE:-1484340654899}     # ~1.35 TiB
MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES=${MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES:-536870912}     # 512 MiB
MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT=${MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT:-1000}
MOONCAKE_OFFLOAD_TOTAL_KEYS_LIMIT=${MOONCAKE_OFFLOAD_TOTAL_KEYS_LIMIT:-120000000}

# ---- Paths / infra ----
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
VLLM_ASCEND_ROOT=${VLLM_ASCEND_ROOT:-/vllm-workspace/vllm-ascend}
PROXY_DIR=${PROXY_DIR:-${VLLM_ASCEND_ROOT}/examples/disaggregated_prefill_v1}
# If the user set MOONCAKE_CONFIG_PATH, respect it; otherwise write_mooncake_config() generates one.
MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-}
LOG_ROOT=${LOG_ROOT:-logs}
RESULT_ROOT=${RESULT_ROOT:-results}
MIN_FREE_MEMORY_MB=${MIN_FREE_MEMORY_MB:-50000}
KILL_ONLY=false

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

# ============================================================
# Parse args (flags override env)
# ============================================================
parse_args() {
  while [[ $# -gt 0 ]]; do
    case $1 in
      -p|--num-prefill) NUM_PREFILL="$2"; shift 2 ;;
      -d|--num-decode)  NUM_DECODE="$2"; shift 2 ;;
      -m|--model)       MODEL="$2"; shift 2 ;;
      -b|--baseline)    BASELINE="$2"; shift 2 ;;
      --max-model-len)  MAX_MODEL_LEN="$2"; shift 2 ;;
      --proxy-port)     PROXY_PORT="$2"; shift 2 ;;
      -k|--kill-only)   KILL_ONLY=true; shift 1 ;;
      -h|--help)        grep "^#" "$0" | sed '1d; s/^# \{0,1\}//'; exit 0 ;;
      *) echo "Unknown option: $1"; exit 1 ;;
    esac
  done
}

# ============================================================
# Host IP
# ============================================================
resolve_host_ip() {
  if [ -n "${VLLM_HOST_IP:-}" ]; then return; fi
  if command -v hostname &>/dev/null; then
    VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
  elif command -v ip &>/dev/null; then
    VLLM_HOST_IP=$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7}')
  fi
  export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}
  export no_proxy="localhost,127.0.0.1,${VLLM_HOST_IP},${no_proxy}"
  export NO_PROXY="localhost,127.0.0.1,${VLLM_HOST_IP},${NO_PROXY}"
}

# ============================================================
# Cluster lifecycle
# ============================================================
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s http://${VLLM_HOST_IP}:${port}/health > /dev/null; do sleep 1; done" \
    && return 0 || return 1
}

get_related_pids() {
  local pid=${1}; [ -z "$pid" ] && echo ""
  ps -ef | grep "$pid" | grep -v 'grep' | awk '{print $2}' | tr '\n' ' '
}

destroy_node_by_port_and_pattern() {
  local port=$1 pattern=$2
  local target_pids
  target_pids=$(ps -ef | grep -v 'grep' | grep "${pattern}" | grep -E "\<${port}\>" | awk '{print $2}')
  local timeout_pids
  timeout_pids=$(ps -ef | grep -v 'grep' | grep "timeout" | grep -E ":${port}(/|\>)" | awk '{print $2}')
  local all_targets="${target_pids} ${timeout_pids}"
  all_targets=$(echo "${all_targets}" | xargs -n1 2>/dev/null | sort -u | xargs)
  [ -z "${all_targets}" ] && return 0
  for target_pid in ${all_targets}; do
    local related_pids
    related_pids=$(get_related_pids "${target_pid}")
    for pid in ${related_pids}; do related_pids="${related_pids} $(get_related_pids "$pid")"; done
    local kill_pool
    kill_pool=$(echo "${target_pid} ${related_pids}" | xargs -n1 | sort -u | xargs)
    [ -n "${kill_pool}" ] && kill -9 ${kill_pool} 2>/dev/null || true
  done
}

get_free_npus() {
  local min_free_mb=${1:-40000} output
  output=$(npu-smi info 2>/dev/null)
  if [ -z "$output" ]; then echo "0,1,2,3,4,5,6,7"; return; fi
  echo "$output" | awk -F'|' -v min_free="$min_free_mb" '
    $3 ~ /0000:/ {
      clean_c2=$2; gsub(/[^0-9]+/," ",clean_c2); split(clean_c2,c2," "); phy_id=c2[2]
      clean_c4=$4; gsub(/[^0-9]+/," ",clean_c4); split(clean_c4,c4," ")
      if (length(c4)>=2) { used=c4[length(c4)-1]; total=c4[length(c4)]
        if (phy_id!="" && phy_id!="0" && total>0 && (total-used)>=min_free)
          free_list=(free_list==""?phy_id:free_list","phy_id) } }
    END { print free_list }'
}

assign_npu_for_node() {
  echo "$AVAILABLE_NPUS" | cut -d',' -f$(( $1 + 1 )) 2>/dev/null | tr -d ' '
}

kill_all_nodes() {
  echo "Wiping existing cluster..."
  destroy_node_by_port_and_pattern ${PROXY_PORT} "proxy"
  destroy_node_by_port_and_pattern ${MASTER_PORT} "mooncake_master"
  for ((i=0; i<NUM_PREFILL; i++)); do destroy_node_by_port_and_pattern $((PREFILL_PORT_BASE + i)) "$LAUNCHER"; done
  for ((i=0; i<NUM_DECODE; i++));  do destroy_node_by_port_and_pattern $((DECODE_PORT_BASE + i))  "$LAUNCHER"; done
  curl -X POST http://${VLLM_HOST_IP}:${MASTER_PORT}/admin/clear_all 2>/dev/null || true
  for ((i=0; i<NUM_PREFILL; i++)); do rm -f "/tmp/lookup_rpc_port_$((PREFILL_PORT_BASE+i))_dp_rank0" 2>/dev/null || true; done
  for ((i=0; i<NUM_DECODE; i++));  do rm -f "/tmp/lookup_rpc_port_$((DECODE_PORT_BASE+i))_dp_rank0"  2>/dev/null || true; done
  sleep 2
}

# ============================================================
# Shared Ascend env (exported into every engine process)
# ============================================================
export_ascend_env() {
  export HCCL_OP_EXPANSION_MODE="AIV"
  export HCCL_IF_IP=127.0.0.1
  export GLOO_SOCKET_IFNAME="lo" TP_SOCKET_IFNAME="lo" HCCL_SOCKET_IFNAME="lo"
  export OMP_PROC_BIND=false OMP_NUM_THREADS=10
  export VLLM_USE_V1=1 HCCL_BUFFSIZE=200
  export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
  export TASK_QUEUE_ENABLE=1 CPU_AFFINITY_CONF=1 ASCEND_AGGREGATE_ENABLE=1
  export ASCEND_TRANSPORT_PRINT=1 ACL_OP_INIT_MODE=1 HCCL_INTRA_ROCE_ENABLE=1
  export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
  export PYTHONHASHSEED=0
  export MOONCAKE_CONFIG_PATH ASCEND_BUFFER_POOL=4:8 MC_STORE_ENABLE_HTTP_SERVER=1
  # COMPILATION_CONFIG (cudagraph) is DECODE-ONLY (FULL_DECODE_ONLY); prefill launches without it.
  export COMPILATION_CONFIG='{"cudagraph_capture_sizes":[1,4,8,12,16,20,24,28,32,36,40,48,56,64,80,96],"cudagraph_mode":"FULL_DECODE_ONLY"}'
  export_ssd_offload_env   # no-op unless ENABLE_SSD_OFFLOAD=1
  unset HCCL_INTRA_ROCE_ENABLE
}

# Export the MOONCAKE_OFFLOAD_* env consumed by the Mooncake store/master when SSD offload is on.
# No-op when ENABLE_SSD_OFFLOAD=0 so the default path is unchanged. Called on both the master and the
# engines (superset is harmless — the master reads only the storage/heartbeat subset).
export_ssd_offload_env() {
  [[ "$ENABLE_SSD_OFFLOAD" != "1" ]] && return 0
  export MOONCAKE_OFFLOAD_FILE_STORAGE_PATH="$SSD_OFFLOAD_PATH"
  export MOONCAKE_OFFLOAD_STORAGE_BACKEND_DESCRIPTOR MOONCAKE_OFFLOAD_BUCKET_EVICTION_POLICY \
         MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES MOONCAKE_OFFLOAD_HEARTBEAT_INTERVAL_SECONDS \
         MOONCAKE_OFFLOAD_TOTAL_SIZE_LIMIT_BYTES MOONCAKE_OFFLOAD_BUCKET_MAX_TOTAL_SIZE \
         MOONCAKE_OFFLOAD_BUCKET_SIZE_LIMIT_BYTES MOONCAKE_OFFLOAD_BUCKET_KEYS_LIMIT \
         MOONCAKE_OFFLOAD_TOTAL_KEYS_LIMIT
}

# Guard the SSD-offload prerequisites and prepare the offload directory. Offload only flows through
# AscendStoreConnector -> MooncakeBackend, so it is a no-op (and misleading) without USE_ASCEND_STORE=1,
# and it needs the SSD backend patch applied on the host (adds enable_ssd_offload to mooncake_backend.py).
validate_ssd_offload() {
  [[ "$ENABLE_SSD_OFFLOAD" != "1" ]] && return 0
  if [[ "$USE_ASCEND_STORE" != "1" ]]; then
    echo "ERROR: ENABLE_SSD_OFFLOAD=1 requires USE_ASCEND_STORE=1 — offload flows through the" >&2
    echo "       AscendStore KV pool (MooncakeBackend) and has no effect on the standalone mover." >&2
    echo "       Re-run with USE_ASCEND_STORE=1 (note: bff also needs a SupportsHMA AscendMultiConnector)." >&2
    exit 1
  fi
  local backend_py="${VLLM_ASCEND_ROOT}/vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py"
  if [ -f "$backend_py" ] && ! grep -q "enable_ssd_offload" "$backend_py"; then
    echo "WARNING: ${backend_py} has no 'enable_ssd_offload' support — offload will be ignored." >&2
    echo "         Apply the SSD patches first:  ENABLE_SSD=1 ${REPO_ROOT}/patch/apply_all.sh" >&2
  fi
  mkdir -p "$SSD_OFFLOAD_PATH" || true
  echo "SSD offload: path=${SSD_OFFLOAD_PATH} (requires patch/vllm-ascend/0004,0005 applied on host)"
}

# Generate mooncake.json (read only by AscendStoreConnector's MooncakeBackend, i.e. USE_ASCEND_STORE=1)
# into the run dir and point MOONCAKE_CONFIG_PATH at it — unless the user supplied their own path.
# The enable_ssd_offload/ssd_offload_path keys are emitted only when ENABLE_SSD_OFFLOAD=1.
write_mooncake_config() {
  if [ -n "$MOONCAKE_CONFIG_PATH" ]; then
    echo "Using user-provided MOONCAKE_CONFIG_PATH=$MOONCAKE_CONFIG_PATH"
    export MOONCAKE_CONFIG_PATH
    return 0
  fi
  MOONCAKE_CONFIG_PATH="${logs_root}/mooncake.json"
  local ssd_keys=""
  if [[ "$ENABLE_SSD_OFFLOAD" == "1" ]]; then
    ssd_keys="
  \"enable_ssd_offload\": true,
  \"ssd_offload_path\": \"${SSD_OFFLOAD_PATH}\","
  fi
  cat > "$MOONCAKE_CONFIG_PATH" <<JSON
{
  "metadata_server": "${MC_METADATA_SERVER}",
  "protocol": "${MC_PROTOCOL}",
  "device_name": "${MC_DEVICE_NAME}",
  "master_server_address": "${MC_MASTER_ADDR}:${MASTER_PORT}",
  "global_segment_size": "${MC_GLOBAL_SEGMENT_SIZE}",
  "local_buffer_size": "${MC_LOCAL_BUFFER_SIZE}",${ssd_keys}
  "_generated_by": "run_benchmarks.sh"
}
JSON
  echo "Wrote mooncake.json → $MOONCAKE_CONFIG_PATH (ssd_offload=$ENABLE_SSD_OFFLOAD)"
  export MOONCAKE_CONFIG_PATH
}

# Export BFF env into the current process env (inherited by the launched engine).
# For non-bff baselines, explicitly CLEAR BFF_PD_FUSE so an inherited env var can't re-enable the
# Ascend patch (the group split): apply_fast_fusion_ascend_patch is gated on BFF_PD_FUSE==1.
export_bff_env() {
  local role=$1   # kv_producer | kv_consumer
  if [[ "$BFF_ON" != "1" ]]; then
    export BFF_PD_FUSE=0
    unset BFF_PD_STATS_DIR
    return
  fi
  export BFF_PD_FUSE=$BFF_PD_FUSE BFF_SCALE_MODE=$BFF_SCALE_MODE BFF_PD_MERGE=$BFF_PD_MERGE \
         BFF_PD_REPR=$BFF_PD_REPR BFF_THRESHOLD=$BFF_THRESHOLD BFF_GROUP_SIZE=$BFF_GROUP_SIZE \
         BFF_PD_ENCODED_BATCH_SIZE=$BFF_PD_ENCODED_BATCH_SIZE
  # v2 knobs. Exported for every BFF arm, not just bff_v2: BFF_MAX_REL_ERR also gates v1's merges
  # (both go through pd_lsh.probe), so an A/B at the same error budget is one variable apart.
  export BFF_MAX_REL_ERR=$BFF_MAX_REL_ERR BFF_V2_DEDUP=$BFF_V2_DEDUP \
         BFF_V2_RESIDENT=$BFF_V2_RESIDENT BFF_SIG_DIM=$BFF_SIG_DIM \
         BFF_V2_MAX_RESIDENT=$BFF_V2_MAX_RESIDENT BFF_V2_SIG_TIMEOUT=$BFF_V2_SIG_TIMEOUT \
         BFF_PULL_V2_SIG_TIMEOUT=$BFF_PULL_V2_SIG_TIMEOUT BFF_SIG_LAYERS=$BFF_SIG_LAYERS \
         BFF_V2_MAX_REQ_DECLINE=$BFF_V2_MAX_REQ_DECLINE \
         BFF_V2_PROTECT_HOT_BLOCKS=$BFF_V2_PROTECT_HOT_BLOCKS
  # BOTH roles dump fuse stats: the producer counts blocks + redirects shipped, the decode side counts
  # the redirects that actually landed and the REAL freed-block delta (the measured compression).
  export BFF_PD_STATS_DIR="$results_root"
}

# Build the KV_TRANSFER_CONFIG for a given role + kv_port.
#   USE_ASCEND_STORE=1 → MultiConnector wrapping [<mover>, AscendStoreConnector].
#   USE_ASCEND_STORE=0 → the mover alone (top-level connector; kv_port at top level). Required for
#                        bff on a deployment whose AscendMultiConnector isn't SupportsHMA.
build_kv_transfer_config() {
  local role=$1 kv_port=$2
  if [[ "$USE_ASCEND_STORE" == "1" ]]; then
    cat <<JSON
{
  "kv_connector": "MultiConnector",
  "kv_role": "${role}",
  "kv_load_failure_policy": "${KV_LOAD_FAILURE_POLICY}",
  "kv_connector_extra_config": {
    "connectors": [
      {
        "kv_connector": "${CONNECTOR}",
        "kv_role": "${role}",
        "kv_port": ${kv_port},
        "kv_connector_extra_config": {
          "use_ascend_direct": true,
          "prefill": { "dp_size": ${DP_SIZE}, "tp_size": ${TP_SIZE} },
          "decode":  { "dp_size": ${DP_SIZE}, "tp_size": ${TP_SIZE} }
        }
      },
      {
        "kv_connector": "AscendStoreConnector",
        "kv_role": "${role}",
        "kv_connector_extra_config": { "lookup_rpc_port": "0", "backend": "mooncake" }
      }
    ]
  }
}
JSON
  else
    cat <<JSON
{
  "kv_connector": "${CONNECTOR}",
  "kv_role": "${role}",
  "kv_port": ${kv_port},
  "kv_load_failure_policy": "${KV_LOAD_FAILURE_POLICY}",
  "kv_connector_extra_config": {
    "use_ascend_direct": true,
    "prefill": { "dp_size": ${DP_SIZE}, "tp_size": ${TP_SIZE} },
    "decode":  { "dp_size": ${DP_SIZE}, "tp_size": ${TP_SIZE} }
  }
}
JSON
  fi
}

# vLLM args shared by P and D. BFF needs block-size 128 + prefix caching + hybrid KV manager.
# $1 = tag (prefill|decode). --compilation-config (cudagraph) is DECODE-ONLY.
common_args() {
  local tag=$1 extra=""
  local max_concurrency
  if [[ "$tag" == "prefill" ]]; then
    max_concurrency=${PREFILL_MAX_CONCURRENCY}
  else
    max_concurrency=${DECODE_MAX_CONCURRENCY}
  fi
  if [[ "$BFF_ON" == "1" ]]; then
    extra="--enable-prefix-caching --no-disable-hybrid-kv-cache-manager"
  fi
  if [[ "$tag" == "decode" ]]; then
    extra="${extra} --compilation-config '${COMPILATION_CONFIG}'"
  fi
  echo "--host ${VLLM_HOST_IP} \
    --tensor-parallel-size ${TP_SIZE} \
    --seed ${SEED} \
    --trust-remote-code \
    --block-size ${BLOCK_SIZE} \
    --max-model-len ${MAX_MODEL_LEN} \
    --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} \
    --max-num-seqs ${max_concurrency} \
    ${extra}"
    # --num_gpu_blocks_override 6000 \
}

# ============================================================
# Launchers
# ============================================================
launch_mooncake_master() {
  echo "Launching mooncake_master on port ${MASTER_PORT}..."
  local offload_flag=""
  if [[ "$ENABLE_SSD_OFFLOAD" == "1" ]]; then
    offload_flag="--enable_offload=true"
    export_ssd_offload_env   # master reads MOONCAKE_OFFLOAD_FILE_STORAGE_PATH / heartbeat / eviction
    echo "  SSD offload ENABLED (path=${SSD_OFFLOAD_PATH})"
  fi
  nohup mooncake_master --port ${MASTER_PORT} --eviction_high_watermark_ratio 0.95 \
    --eviction_ratio 0.05 --rpc_thread_num 128 --promotion_on_hit=true \
    --promotion_admission_threshold=3 --default_kv_lease_ttl=30s --client_ttl=30 \
    ${offload_flag} \
    > ${logs_root}/mooncake_master.log 2>&1 &
  sleep 3
}

launch_engines() {
  local role=$1 count=$2 port_base=$3 kv_base=$4 npu_offset=$5 tag=$6
  local max_concurrency
  if [[ "$tag" == "prefill" ]]; then
    max_concurrency=${PREFILL_MAX_CONCURRENCY}
  else
    max_concurrency=${DECODE_MAX_CONCURRENCY}
  fi
  echo "Launching ${count} ${tag} node(s) (max_concurrency=${max_concurrency})..."
  for ((i=0; i<count; i++)); do
    local port=$((port_base + i)) kv_port=$((kv_base + i))
    local npu; npu=$(assign_npu_for_node $((npu_offset + i)))
    echo "  ${tag} $i: NPU ${npu}, HTTP ${port}, KV ${kv_port}, max_num_seqs ${max_concurrency}"

    export_ascend_env
    export_bff_env "$role"
    export ASCEND_RT_VISIBLE_DEVICES=$npu
    local kv_cfg; kv_cfg=$(build_kv_transfer_config "$role" "$kv_port")

    nohup bash -c "python -m ${LAUNCHER} serve \"${MODEL}\" \
        $(common_args "$tag") \
        --port ${port} \
        --gpu-memory-utilization ${GPU_MEM_UTIL} \
        --kv-transfer-config '${kv_cfg}' \
        > \"${logs_root}/${tag}-${i}.txt\" 2>&1" &
    echo "    PID: $!"
  done
}

launch_proxy() {
  echo "Launching disaggregated proxy (layer_wise=${LAYER_WISE})..."
  local prefill_hosts="" prefill_ports="" decode_ports=""
  for ((i=0; i<NUM_PREFILL; i++)); do
    prefill_hosts="${prefill_hosts}${VLLM_HOST_IP} "
    prefill_ports="${prefill_ports}$((PREFILL_PORT_BASE + i)) "
  done
  for ((i=0; i<NUM_DECODE; i++)); do decode_ports="${decode_ports}$((DECODE_PORT_BASE + i)) "; done

  local proxy_script
  if [[ "$LAYER_WISE" == "true" ]]; then
    proxy_script="${PROXY_DIR}/load_balance_proxy_layerwise_server_example.py"
  else
    proxy_script="${PROXY_DIR}/load_balance_proxy_server_example.py"
  fi
  nohup python "$proxy_script" \
    --prefiller-hosts ${prefill_hosts} \
    --prefiller-ports ${prefill_ports} \
    --decoder-hosts ${VLLM_HOST_IP} \
    --decoder-ports ${decode_ports} \
    --host ${VLLM_HOST_IP} \
    --port ${PROXY_PORT} \
    > ${logs_root}/proxy.txt 2>&1 &
  echo "  Proxy PID: $!"
  sleep 3
}

wait_for_all_nodes() {
  for ((i=0; i<NUM_PREFILL; i++)); do wait_for_server $((PREFILL_PORT_BASE + i)); done
  for ((i=0; i<NUM_DECODE; i++));  do wait_for_server $((DECODE_PORT_BASE + i));  done
  echo "All nodes ready."
}

run_benchmark() {
  local target_index=$((NUM_PREFILL + NUM_DECODE))
  local npu; npu=$(assign_npu_for_node $target_index); [ -z "$npu" ] && npu=$(assign_npu_for_node 0)
  export ASCEND_RT_VISIBLE_DEVICES=$npu
  export VLLM_WORKER_MULTIPROC_METHOD="spawn" VLLM_USE_V1="1"
  echo "Running F1 benchmark (prefill_concurrency=${PREFILL_MAX_CONCURRENCY}, decode_concurrency=${DECODE_MAX_CONCURRENCY}, prompts=${NUM_PROMPTS}) against proxy ${PROXY_PORT}..."
  python -m f1_benchmark.f1_main \
    --dataset-path "${F1_DATASET}" --hf-split "${F1_SPLIT}" \
    --input-key "${F1_INPUT_KEY}" --output-key "${F1_OUTPUT_KEY}" \
    --num-prompts ${NUM_PROMPTS} --request-rate ${REQUEST_RATE} --burstiness ${BURSTINESS} \
    --max-concurrency ${DECODE_MAX_CONCURRENCY} --request-timeout ${REQUEST_TIMEOUT} \
    --min-tokens ${MIN_TOKENS} --max-tokens ${MAX_TOKENS} --compute-f1 --compute-code-metrics \
    --model "${MODEL}" --host ${VLLM_HOST_IP} --port ${PROXY_PORT} \
    --result-dir "${results_root}" \
    > "${logs_root}/${BASELINE}-${NUM_PREFILL}Px${NUM_DECODE}D-con${DECODE_MAX_CONCURRENCY}-serving.txt" 2>&1
}

# Post-run BFF stats (producer dumps bff_stats_<pid>.json into results_root when BASELINE=bff).
# Did the decode ever actually run out of KV blocks? BFF's ONLY product is freed KV capacity, so if
# capacity never binds, fusion cannot improve throughput no matter how well it compresses. Printed for
# every run (vanilla included) so this precondition is never invisible again.
report_capacity_bound() {
  python3 - "$logs_root" "${MAX_CONCURRENCY:-0}" <<'PY' || true
import glob, os, re, sys

d = sys.argv[1]
max_conc = int(sys.argv[2]) if len(sys.argv) > 2 else 0
files = sorted(glob.glob(os.path.join(d, "decode-*.txt")))
if not files:
    print("  capacity: no decode logs found"); raise SystemExit

# vLLM v1 logs: "<prefix>Avg prompt throughput: .., Avg generation throughput: .., Running: N reqs,
# Waiting: N reqs[, Preemptions: N], GPU KV cache usage: X%, Prefix cache hit rate: Y%"
# (Preemptions only appears when >0).
#
# Parsed as ONE regex per line so running/waiting/kv are known to CO-OCCUR. The previous version
# matched each field independently and compared their independent maxima, which is unsound: it
# reported "peak running 150, peak waiting 67" for moments that never happened together, and drew a
# verdict from the pair. `.` does not match newline, so each match is confined to one log line.
LINE = re.compile(r"Running: (\d+) reqs.*?Waiting: (\d+) reqs.*?GPU KV cache usage: ([0-9.]+)%")

rows = []                       # (running, waiting, kv_pct), co-occurring
preempt_samples = preempt_max = 0
for fp in files:
    try:
        txt = open(fp, errors="replace").read()
    except Exception:
        continue
    rows += [(int(a), int(b), float(c)) for a, b, c in LINE.findall(txt)]
    for m in re.finditer(r"Preemptions: (\d+)", txt):
        preempt_samples += 1
        preempt_max = max(preempt_max, int(m.group(1)))

if not rows:
    print("  capacity: no scheduler stats in decode logs (is log-stats disabled?)"); raise SystemExit

samples = len(rows)
pr = max(r[0] for r in rows)
pw = max(r[1] for r in rows)
pk = max(r[2] for r in rows)

# THE signal, and the one the old rule got wrong: vLLM does not preempt under admission pressure —
# it declines to admit and the queue grows. So "preemptions > 0" is sufficient evidence of capacity
# binding but NOT necessary, and requiring it made the check tell users to raise MAX_CONCURRENCY in
# runs that were already KV-bound. A sample with the cache full AND requests queued is the scheduler
# saying, in the only way it can, that it wants blocks it does not have.
KV_FULL = 95.0
kv_blocked = sum(1 for r in rows if r[2] >= KV_FULL and r[1] > 0)
kv_blocked_frac = kv_blocked / samples

# Concurrency-capped means the CLIENT is the limit: running is at the cap and nothing is queued
# behind it. Both conditions must hold in the SAME sample — running at the cap while a queue exists
# means the queue is blocked on something else, which is the KV case above.
capped = (sum(1 for r in rows if r[0] >= max_conc and r[1] == 0) / samples) if max_conc > 0 else 0.0

if preempt_samples > 0:
    verdict = "YES"
elif kv_blocked_frac >= 0.10:
    verdict = "YES"
elif capped >= 0.50:
    verdict = "NO (concurrency-capped)"
elif pk >= 90.0:
    verdict = "MARGINAL"
else:
    verdict = "MARGINAL" if pk >= 70.0 else "NO"

print(f"  capacity-bound: {verdict} (peak KV {pk:.1f}%, peak waiting {int(pw)}, "
      f"peak running {int(pr)}, preemptions {preempt_max} in {preempt_samples} samples"
      + (f", MAX_CONCURRENCY={max_conc}" if max_conc else "") + ")")
print(f"    KV>={KV_FULL:.0f}% with a non-empty queue: {kv_blocked}/{samples} samples "
      f"({100*kv_blocked_frac:.1f}%) — the admission-pressure signal")

if verdict == "YES" and preempt_samples == 0:
    print(f"    -> KV is blocking ADMISSION: the scheduler is queueing rather than preempting, so "
          f"preemptions stay 0. Freed blocks should raise the sustained Running count here.")
elif verdict.startswith("NO (conc"):
    print(f"    -> Running sits at MAX_CONCURRENCY={max_conc} with an EMPTY queue in "
          f"{100*capped:.0f}% of samples: the CLIENT's cap binds, not KV "
          f"(KV peaks at {pk:.1f}%, i.e. {100-pk:.0f}% headroom).")
    print(f"    -> raise MAX_CONCURRENCY until the queue stays non-empty while KV is full.")
elif verdict.startswith("NO"):
    print(f"    -> KV cache never filled ({pk:.1f}%): capacity is NOT the bottleneck, so freeing "
          f"blocks cannot raise throughput here.")
    print(f"    -> to make fusion matter: lower GPU_MEM_UTIL (shrink the pool) or raise "
          f"MAX_CONCURRENCY/MAX_NUM_SEQS/prompt length until this says YES.")
else:
    print(f"    -> KV peaks high ({pk:.1f}%) but rarely blocks admission "
          f"({100*kv_blocked_frac:.1f}% of samples): fusion has little room to prove itself.")
PY
}

collect_bff_stats() {
  [[ "$BFF_ON" != "1" ]] && return 0
  python3 - "$results_root" <<'PY' || true
import glob, json, os, sys
d = sys.argv[1]


def load(pat):
    out = []
    for fp in sorted(glob.glob(os.path.join(d, pat))):
        try:
            out.append(json.load(open(fp)))
        except Exception:
            pass
    return out


prod = load("bff_stats_*.json")           # producer: blocks seen + redirects SHIPPED
dec = load("bff_decode_stats_*.json")     # decode scheduler: REAL freed-block delta
app = load("bff_apply_stats_*.json")      # decode worker: which redirects landed, and why not
if not prod:
    print("  bff stats: none found (fusion may not have engaged)"); raise SystemExit

# v2 writes into the SAME bff_stats_*.json but with a completely different schema (it has no
# redirects at all — a block it declines is simply never transferred). Split them off first: fed to
# the v1 reader below they trip the schema-mismatch guard, which then refuses to report anything,
# and that is what a bff_v2 / bff_pull_v2 run has been silently doing.
v2 = [s for s in prod if s.get("bff_version") == 2]
prod = [s for s in prod if s.get("bff_version") != 2]

if v2:
    P = sum(s.get("blocks_planned", 0) for s in v2)
    D = sum(s.get("blocks_not_requested", 0) for s in v2)
    RES = sum(s.get("blocks_not_requested_resident", 0) for s in v2)
    SAME = sum(s.get("blocks_not_requested_same_pull", 0) for s in v2)
    AP = sum(s.get("aliases_applied", 0) for s in v2)
    RC = sum(s.get("aliases_recomputed", 0) for s in v2)
    EX = sum(s.get("exchanges", 0) for s in v2)
    FAIL = sum(s.get("signature_phase_failed", 0) for s in v2)
    print(f"  bff v2 wire saving: {100.0*D/P if P else 0.0:.1f}% of blocks never requested "
          f"({D} of {P}; {RES} from resident, {SAME} within the same pull) over {len(v2)} node(s)")
    # An all-zero saving reads two completely different ways, and only these separate them:
    # exchanges>0 means v2 ran and found nothing worth merging; exchanges==0 means it never ran.
    print(f"    exchanges: {EX} ({FAIL} fell back to a full read) | aliases applied {AP}, "
          f"recomputed {RC}"
          + (f" ({100.0*AP/(AP+RC):.1f}% resolved)" if (AP + RC) else ""))
    RT = sum(s.get("sig_batches") or 0 for s in v2)
    if RT:
        # The batched protocol shipped INERT once: 512 requests in 512 round trips, because the
        # thread it runs on handles one at a time. This ratio is the only thing that says otherwise.
        print(f"    signature round trips: {RT} for {EX} request(s) "
              f"({EX/RT:.1f} per exchange, cap BFF_PULL_V2_SIG_BATCH)")
        if EX / RT < 1.5:
            print("    -> BATCHING DID NOT ENGAGE: the recv queue is never deep enough to drain. "
                  "Nothing downstream of this is worth tuning until it is.")
    if EX == 0:
        print("    -> v2 NEVER ASKED: the mechanism did not engage at all. Check BFF_V2_DEDUP and "
              "the producer's 'signature server (REP) bound on' line in the prefill log.")
    elif FAIL >= 0.5 * EX:
        print(f"    -> {100.0*FAIL/EX:.0f}% of exchanges failed. Grep the PREFILL log: no 'bound on'"
              f" line = the producer never listened; 'bound on' but no 'served' = the decode never "
              f"reached it; both = it answered too slowly (raise BFF_PULL_V2_SIG_TIMEOUT).")
    # Did the KV that landed on the decode match what the producer had? The only CONTENT check in
    # the whole transfer path; everything else is structural (lengths, coverage, reachability).
    VC = sum(s.get("verify_checked") or 0 for s in v2)
    if VC:
        VM = sum(s.get("verify_mismatched") or 0 for s in v2)
        WC = min([s.get("verify_worst_cos") for s in v2 if s.get("verify_worst_cos") is not None]
                 or [1.0])
        print(f"    transfer verify: {VM} of {VC} block(s) MISMATCHED (worst cos {WC:.5f})")
        if not VM:
            print("    -> the KV arriving on the decode is the KV the producer had. Any quality "
                  "loss is downstream of the transfer.")
        else:
            # NOT "the transfer is wrong" — that was overclaimed once already. A mismatch says the
            # decode's KV differs from the SIGNATURE the producer gave for it, which has three
            # causes; the verdicts are the re-check that tells them apart.
            VV = {}
            for s in v2:
                for k, n in (s.get("verify_verdicts") or {}).items():
                    VV[k] = VV.get(k, 0) + n
            print(f"    -> the decode's KV does not match the signature the producer gave for it. "
                  f"Cause: {VV or 'not diagnosed'}")
            if VV.get("transfer_wrong"):
                print("    -> TRANSFER_WRONG: the producer was stable and the decode still differs."
                      " Everything else is downstream of this.")
                VL = {}
                for s in v2:
                    for k, n in (s.get("verify_localised") or {}).items():
                        VL[k] = VL.get(k, 0) + n
                if VL:
                    # What the wrong block actually held. source/destination/group are descriptor
                    # arithmetic and localise to an index; foreign is block ownership instead, and
                    # the fix for that is not in the descriptor loop. `degraded` is neither — it is
                    # a correct block whose magnitude drifted, and it must not be read as corruption.
                    print(f"       what it held: {VL}")
                    if VL.get("degraded"):
                        print(f"       -> {VL['degraded']} of these still matched their OWN row: "
                              "content correct, magnitude drifted. Not wrong blocks.")
                    if VL.get("foreign_static"):
                        # STATIC alone does not mean "never written" — a block overwritten ONCE by
                        # another owner reads the same. The replay verdicts below are what decide.
                        print(f"       -> {VL['foreign_static']} STATIC across a re-read.")
                    if VL.get("retry_same"):
                        print(f"       -> {VL['retry_same']} RETRY_SAME: replaying the block's own "
                              "descriptor returned identical content. The remote address does not "
                              "point at the intended block — this is OUR arithmetic. Read the "
                              "logged descriptor in the decode log.")
                    if VL.get("retry_fixed"):
                        print(f"       -> {VL['retry_fixed']} RETRY_FIXED: the same descriptor "
                              "delivered correct KV on a second attempt. The first write was LOST "
                              "inside batch_transfer_sync_read — upstream, and shared with the "
                              "stock connector.")
                    if VL.get("foreign_changing"):
                        print(f"       -> {VL['foreign_changing']} CHANGED across a re-read: "
                              "another writer owns the block. That is a block-ownership bug, not "
                              "the transfer.")
                    if VL.get("foreign") and not any(
                            VL.get(k) for k in ("source_permuted", "destination_permuted",
                                                "group_permuted")):
                        print("       -> FOREIGN: the content belongs to no row of the request. "
                              "That is block OWNERSHIP, not the descriptor arithmetic — check "
                              "whether the stock connector has it too before changing ours.")
            elif VV.get("producer_moved"):
                print("    -> PRODUCER_MOVED: the transfer is faithful, the signature was stale. "
                      "Dedup decides what to skip from those same signatures.")
            elif VV.get("transient"):
                print("    -> TRANSIENT: everything agreed on re-check, so the first read saw the "
                      "block before it settled. Points at the reader, not the transfer.")

    # Where dedup's per-step cost lives. At con512 the group split alone costs +4.5% per decode step
    # and dedup a further +26%, i.e. ~18ms/step; apply_ms is ~1.5s of that, so the rest is elsewhere.
    # hook_ms bounds the forward path, the others are recv-thread work competing with it for the GIL.
    HOOK = sum(s.get("hook_ms_total") or 0 for s in v2)
    APPLY_MS = sum(s.get("apply_ms_total") or 0 for s in v2)
    PLAN = sum(s.get("plan_ms_total") or 0 for s in v2)
    SDEC = sum(s.get("sig_decode_ms_total") or 0 for s in v2)
    EXMS = sum(s.get("exchange_ms_total") or 0 for s in v2)
    AUD = sum(s.get("audit_ms_total") or 0 for s in v2)
    if HOOK or PLAN or EXMS:
        print(f"    dedup time: forward-path hook {HOOK/1000:.1f}s (apply {APPLY_MS/1000:.1f}s, "
              f"audit {AUD/1000:.1f}s) | recv thread: plan {PLAN/1000:.1f}s "
              f"(decode {SDEC/1000:.1f}s), exchange {EXMS/1000:.1f}s")
        if not PLAN:
            # Dedup off. The exchange then runs ONLY while the verification budget lasts, so
            # exchange_ms covers a handful of requests and comparing it against a whole run's hook
            # time is comparing two different denominators. A previous version printed the
            # recv-thread verdict from 45.9s over 33 requests, which meant nothing.
            print(f"    -> dedup is OFF: the {EXMS/1000:.1f}s of exchange above covers only the "
                  f"{EX} verified request(s), not the run. Not comparable to the hook time.")
        elif PLAN + EXMS > 5 * HOOK and PLAN + EXMS > 30000:
            print("    -> the cost is on the RECV THREAD, not the forward path: that work shares a "
                  "process and the GIL with the decode loop, so it steals from every step.")

    reasons = {}
    for s in v2:
        for k, n in (s.get("alias_failure_reasons") or {}).items():
            reasons[k] = reasons.get(k, 0) + n
    if reasons:
        print("    alias failures: " + " ".join(f"{k}={n}" for k, n in sorted(reasons.items())))

    # THE distribution. A run-level saving of a few percent reads identically whether every request
    # gave up a little or a handful gave up nearly everything — and only the second answers the
    # wrong prompt. Printed next to the saving so the two are never read apart again.
    BUCKETS = ("0-10%", "10-25%", "25-50%", "50-75%", "75-90%", "90-100%")
    hist = {b: sum((s.get("request_decline_frac") or {}).get(b, 0) for s in v2) for b in BUCKETS}
    n_req = sum(hist.values())
    if n_req:
        print("    per-request decline: "
              + " ".join(f"{b}={hist[b]}" for b in BUCKETS if hist[b]))
        heavy = hist["75-90%"] + hist["90-100%"]
        capped = sum(s.get("requests_capped", 0) for s in v2)
        if capped:
            print(f"    -> {capped} request(s) exceeded BFF_V2_MAX_REQ_DECLINE and were read in "
                  f"full: that much substitution answers a neighbouring prompt.")
        if heavy:
            print(f"    -> {heavy} request(s) ({100.0*heavy/n_req:.1f}%) had OVER 75% of their KV "
                  f"replaced. Compare against the length(=max_tokens) share in the serving log — "
                  f"a matching fraction means those are the damaged requests.")
    hot = sum(s.get("hot_block_aliases", 0) for s in v2)
    if hot:
        guarded = any(s.get("hot_block_guarded") for s in v2)
        print(f"    hot-block aliases: {hot} "
              f"({'refused' if guarded else 'APPLIED — BFF_V2_PROTECT_HOT_BLOCKS=0'}), "
              f"{sum(s.get('hot_block_free_slots', 0) for s in v2)} free slots at stake")
    if not prod:
        raise SystemExit

B = sum(s.get("total_blocks", 0) for s in prod)

# Fusion overhead MUST be reported from the forward-thread counters. overhead_avg_group_dedup_ms is
# accumulated inside run_fusion_task when the async worker is on, i.e. on the WORKER thread — so
# across BFF_FF_ASYNC=0/1 it measures two different regions and comparing it is meaningless. It read
# 23.693 -> 2.241 ms/group across an A/B in which the real per-hook forward cost was flat at
# 9.11 -> 9.13 ms. forward_* is the honest cost of BFF to prefill; dedup is shown separately and
# labelled by where it ran.
live = [s for s in prod if s.get("steps")]
fwd = [s["forward_total_ms"] / s["forward_calls"] for s in live
       if s.get("forward_calls") and "forward_total_ms" in s]
fwd_g = [s["forward_avg_per_group_ms"] for s in live if s.get("forward_avg_per_group_ms")]
ov = [s.get("overhead_avg_group_dedup_ms", 0.0) for s in live]
_async = any(s.get("async_worker") for s in live)
if fwd:
    ov_s = (f" | fusion overhead {sum(fwd)/len(fwd):.3f} ms/layer-hook"
            + (f", {sum(fwd_g)/len(fwd_g):.3f} ms/group" if fwd_g else "") + " [forward thread]")
    if ov:
        ov_s += (f" (+{sum(ov)/len(ov):.3f} ms/group "
                 + ("on the fusion worker, off the critical path)" if _async else "clustering, inline)"))
elif ov:
    ov_s = (f" | fusion overhead {sum(ov)/len(ov):.3f} ms/group [dedup only"
            + (" — WORKER thread, not comparable to the sync arm]" if _async else " — inline]"))
else:
    ov_s = ""


def emitted(s):
    """Redirect rows the producer shipped. 'freed' is the legacy name for the same counter (it was
    never blocks freed). Tolerate either, and return None if the producer speaks neither dialect —
    a silent .get(key, 0) there renders a version skew as a believable '1.000x', which is exactly
    how a stale run_benchmarks.sh vs a newer connector went unnoticed."""
    for k in ("redirects_emitted", "freed"):
        if k in s:
            return s[k]
    return None


vals = [emitted(s) for s in prod]
if any(v is None for v in vals):
    keys = sorted({k for s in prod for k in s})
    print(f"  bff stats: SCHEMA MISMATCH — no redirect count in producer stats (keys seen: {keys}).")
    print(f"    -> run_benchmarks.sh is out of sync with the connector; re-sync and rerun. "
          f"Refusing to report a compression number.")
    raise SystemExit
E = sum(vals)

if dec:
    # MEASURED: blocks the decode block-pool actually reclaimed.
    R = sum(s.get("blocks_freed_total", 0) for s in dec)
    ME = sum(s.get("merge_events", 0) for s in dec)
    # Cross-check against the apply side. Each applied redirect frees ~1 block, and merge_events
    # should equal promo_merge_calls — so a large shortfall means the decode ledger was TRUNCATED by
    # its dump cadence, not that fusion failed. This exact case reported freed=16 / 1 event against a
    # real 785 / 38, turning a ~1.8x compression into a reported 1.007x.
    MC = sum(s.get("promo_merge_calls", 0) for s in app)
    PA = sum(s.get("promo_applied", 0) for s in app)
    if MC and ME < MC:
        print(f"  bff compression: NOT REPORTED — decode ledger looks truncated "
              f"(merge_events={ME} but promo_merge_calls={MC}; freed={R} vs promo_applied={PA}).")
        print(f"    -> bff_decode_stats_*.json is stale; grep 'Block merging freed' in the decode "
              f"log for the true total. Overhead below is still valid.{ov_s}")
    else:
        print(f"  bff compression: {B/max(1,B-R):.3f}x smaller (blocks={B} freed={R}) "
              f"[measured, decode-side]{ov_s}")
        if R > E:
            # A decode side that freed more blocks than the producer says it shipped means the
            # PRODUCER ledger is short — its own dump cadence stopped before the run did. Printing
            # "realized 212%" instead of saying so is how a truncated ledger passes for a result.
            print(f"    producer potential: NOT REPORTED — producer ledger looks truncated "
                  f"(redirects_emitted={E} < freed={R}); blocks={B} is short for the same reason.")
        else:
            rate = f"{100.0*R/E:.1f}%" if E else "n/a"
            print(f"    producer potential: {B/max(1,B-E):.3f}x (redirects_emitted={E}) "
                  f"→ realized {rate}")
else:
    print(f"  bff compression: {B/max(1,B-E):.3f}x POTENTIAL only (blocks={B} redirects_emitted={E}) "
          f"over {len(prod)} producer(s){ov_s}")
    print("    (no decode-side stats: rerun with BFF_PD_STATS_DIR exported for the consumer role)")

if app:
    a = {k: sum(s.get(k, 0) for s in app) for k in
         ("applied", "reps_unresolved", "owner_unresident", "owners_deferred",
          "owners_dropped_post_decode")}
    print("    apply (legacy worker-side path): " + " ".join(f"{k}={v}" for k, v in a.items()))
    # The promotion-time path is the live one (BFF_FF_PROMO_APPLY=1), so the legacy counters above
    # are all zero in a normal run and say nothing. These are the ones that matter.
    p = {k: sum(s.get(k, 0) for s in app) for k in
         ("promo_applied", "promo_unresolved", "promo_unres_rep_loading", "promo_unres_rep_gone",
          "promo_rows_late", "promo_merge_calls")}
    tot = p["promo_applied"] + p["promo_unresolved"]
    res = f"{100.0*p['promo_applied']/tot:.1f}%" if tot else "n/a"
    print(f"    apply (promotion path): resolution {res} of {tot} rows | "
          + " ".join(f"{k[6:]}={v}" for k, v in p.items()))
    if p["promo_rows_late"]:
        print(f"    WARNING: {p['promo_rows_late']} redirect rows arrived after their owner was "
              f"promoted and could not be applied — fusion is lagging the KV transfer.")
    rg = {k[8:]: sum(s.get(k, 0) for s in app) for k in
          ("repgone_revive_live", "repgone_revive_cached", "repgone_truly_gone",
           "repgone_nohist_missing", "repgone_nohist_gi_oob", "repgone_nohist_slot_oob",
           "repgone_nohist_badblock")}
    if any(rg.values()):
        print("    rep-gone causes: " + " ".join(f"{k}={v}" for k, v in rg.items() if v))
PY
}

# ============================================================
# Main
# ============================================================
main() {
  parse_args "$@"
  resolve_host_ip

  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  logs_root="${SCRIPT_DIR}/${LOG_ROOT}_latest"
  results_root="${SCRIPT_DIR}/${RESULT_ROOT}_latest"

  if [ "$KILL_ONLY" = true ]; then kill_all_nodes; echo "Cleanup done."; exit 0; fi

  echo "BFF Ascend config: BASELINE=$BASELINE launcher=$LAUNCHER connector=$CONNECTOR ${NUM_PREFILL}Px${NUM_DECODE}D tp=$TP_SIZE use_ascend_store=$USE_ASCEND_STORE"
  [[ "$BFF_ON" == "1" ]] && echo "  BFF: fuse=$BFF_PD_FUSE scale=$BFF_SCALE_MODE merge=$BFF_PD_MERGE repr=$BFF_PD_REPR thr=$BFF_THRESHOLD gs=$BFF_GROUP_SIZE eb=$BFF_PD_ENCODED_BATCH_SIZE max_rel_err=$BFF_MAX_REL_ERR"
  [[ "$BASELINE" == "bff_v2" ]] && echo "  BFF v2: dedup=$BFF_V2_DEDUP resident=$BFF_V2_RESIDENT sig_dim=$BFF_SIG_DIM sig_layers=$BFF_SIG_LAYERS sig_timeout=${BFF_V2_SIG_TIMEOUT}s"
  [[ "$BASELINE" == "bff_pull_v2" ]] && echo "  BFF pull-v2: dedup=$BFF_V2_DEDUP resident=$BFF_V2_RESIDENT sig_dim=$BFF_SIG_DIM sig_timeout=${BFF_PULL_V2_SIG_TIMEOUT}s (D asks P; sig port = kv_port+22000)"

  rm -rf "${logs_root}" "${results_root}"; mkdir -p "${logs_root}" "${results_root}"
  rm -f "${results_root}"/bff_stats_*.json

  validate_ssd_offload
  write_mooncake_config

  kill_all_nodes
  echo "Detecting free NPUs..."; AVAILABLE_NPUS=$(get_free_npus ${MIN_FREE_MEMORY_MB}); echo "  $AVAILABLE_NPUS"

  ulimit -l unlimited || true
  launch_mooncake_master
  launch_engines "kv_producer" "$NUM_PREFILL" "$PREFILL_PORT_BASE" "$PREFILL_KV_PORT_BASE" 0 "prefill"
  launch_engines "kv_consumer" "$NUM_DECODE"  "$DECODE_PORT_BASE"  "$DECODE_KV_PORT_BASE"  "$NUM_PREFILL" "decode"
  wait_for_all_nodes
  launch_proxy

  run_benchmark
  report_capacity_bound
  collect_bff_stats

  echo "Benchmark done. Logs: ${logs_root}  Results: ${results_root}"
  kill_all_nodes
}

main "$@"
