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
#   layerwise            → stock MooncakeLayerwiseConnector (layerwise transfer, no fusion)
#   mooncakev1           → stock MooncakeConnectorV1 (whole-request transfer)
#
# Example runs:
#   ./run_benchmarks.sh                                   # 2P1D, bff, defaults
#   NUM_PREFILL=2 NUM_DECODE=1 BASELINE=bff BFF_THRESHOLD=0.85 ./run_benchmarks.sh
#   BASELINE=layerwise ./run_benchmarks.sh                # fusion ablation (BFF layout off entirely)
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
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}   # was 64 in the original (a typo) — 64 starves prefill
MAX_NUM_SEQS=${MAX_NUM_SEQS:-48}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}
BLOCK_SIZE=${BLOCK_SIZE:-128}         # BFF requires 128
SEED=${SEED:-1024}

# ---- Baseline / connector selection ----
BASELINE=${BASELINE:-bff}             # bff | layerwise | mooncakev1
case "$BASELINE" in
  bff)         CONNECTOR="MooncakeLayerwiseConnectorFF"; LAYER_WISE=true;  BFF_ON=1 ;;
  layerwise)   CONNECTOR="MooncakeLayerwiseConnector";   LAYER_WISE=true;  BFF_ON=0 ;;
  mooncakev1)  CONNECTOR="MooncakeConnectorV1";          LAYER_WISE=false; BFF_ON=0 ;;
  *) echo "Unknown BASELINE=$BASELINE (use bff|layerwise|mooncakev1)"; exit 1 ;;
esac

# ---- BFF knobs (only take effect when BASELINE=bff) ----
BFF_PD_FUSE=${BFF_PD_FUSE:-1}            # connector-level fusion + redirect propagation to D
BFF_SCALE_MODE=${BFF_SCALE_MODE:-raw}    # NPU supports raw only (ratio needs a CUDA Triton kernel)
BFF_PD_MERGE=${BFF_PD_MERGE:-cc}         # within-batch clustering: cc | nr_tree
BFF_PD_REPR=${BFF_PD_REPR:-proj}         # block repr for similarity: full | proj | mean
BFF_THRESHOLD=${BFF_THRESHOLD:-0.75}     # cosine merge threshold (0..1)
BFF_GROUP_SIZE=${BFF_GROUP_SIZE:-4}      # fusion layers packed per KV-cache group
BFF_PD_ENCODED_BATCH_SIZE=${BFF_PD_ENCODED_BATCH_SIZE:-128}   # cross-batch registry window (0=within-batch only)

# ---- Benchmark knobs ----
NUM_PROMPTS=${NUM_PROMPTS:-1024}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-256}
REQUEST_RATE=${REQUEST_RATE:-64}
BURSTINESS=${BURSTINESS:-0.1}
MIN_TOKENS=${MIN_TOKENS:-500}
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-6000.0}
F1_DATASET=${F1_DATASET:-m-a-p/CodeFeedback-Filtered-Instruction}
F1_SPLIT=${F1_SPLIT:-train}
F1_INPUT_KEY=${F1_INPUT_KEY:-query}
F1_OUTPUT_KEY=${F1_OUTPUT_KEY:-answer}

# ---- Paths / infra ----
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}
VLLM_ASCEND_ROOT=${VLLM_ASCEND_ROOT:-/vllm-workspace/vllm-ascend}
PROXY_DIR=${PROXY_DIR:-${VLLM_ASCEND_ROOT}/examples/disaggregated_prefill_v1}
MOONCAKE_CONFIG_PATH=${MOONCAKE_CONFIG_PATH:-"$(pwd)/mooncake.json"}
LOG_ROOT=${LOG_ROOT:-logs}
RESULT_ROOT=${RESULT_ROOT:-results}
MIN_FREE_MEMORY_MB=${MIN_FREE_MEMORY_MB:-40000}
KILL_ONLY=false

export no_proxy="localhost,127.0.0.1,${no_proxy}"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY}"
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
        if (phy_id!="" && total>0 && (total-used)>=min_free)
          free_list=(free_list==""?phy_id:free_list","phy_id) } }
    END { print free_list }'
}

assign_npu_for_node() {
  echo "$AVAILABLE_NPUS" | cut -d',' -f$(( $1 + 1 )) 2>/dev/null | tr -d ' '
}

kill_all_nodes() {
  echo "Wiping existing cluster..."
  destroy_node_by_port_and_pattern ${PROXY_PORT} "proxy_server"
  destroy_node_by_port_and_pattern ${MASTER_PORT} "mooncake_master"
  for ((i=0; i<NUM_PREFILL; i++)); do destroy_node_by_port_and_pattern $((PREFILL_PORT_BASE + i)) "fast_fusion_main"; done
  for ((i=0; i<NUM_DECODE; i++));  do destroy_node_by_port_and_pattern $((DECODE_PORT_BASE + i))  "fast_fusion_main"; done
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
  # Only the producer dumps fuse stats; point it at the results dir for post-run collection.
  if [[ "$role" == "kv_producer" ]]; then export BFF_PD_STATS_DIR="$results_root"; else unset BFF_PD_STATS_DIR; fi
}

# Build the MultiConnector KV_TRANSFER_CONFIG for a given role + kv_port.
build_kv_transfer_config() {
  local role=$1 kv_port=$2
  cat <<JSON
{
  "kv_connector": "MultiConnector",
  "kv_role": "${role}",
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
}

# vLLM args shared by P and D. BFF needs block-size 128 + prefix caching + hybrid KV manager.
# $1 = tag (prefill|decode). --compilation-config (cudagraph) is DECODE-ONLY.
common_args() {
  local tag=$1 extra=""
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
    --max-num-seqs ${MAX_NUM_SEQS} \
    ${extra}"
}

# ============================================================
# Launchers
# ============================================================
launch_mooncake_master() {
  echo "Launching mooncake_master on port ${MASTER_PORT}..."
  nohup mooncake_master --port ${MASTER_PORT} --eviction_high_watermark_ratio 0.95 \
    --eviction_ratio 0.05 --rpc_thread_num 128 --promotion_on_hit=true \
    --promotion_admission_threshold=3 --default_kv_lease_ttl=30s --client_ttl=30 \
    > ${logs_root}/mooncake_master.log 2>&1 &
  sleep 3
}

launch_engines() {
  local role=$1 count=$2 port_base=$3 kv_base=$4 npu_offset=$5 tag=$6
  echo "Launching ${count} ${tag} node(s)..."
  for ((i=0; i<count; i++)); do
    local port=$((port_base + i)) kv_port=$((kv_base + i))
    local npu; npu=$(assign_npu_for_node $((npu_offset + i)))
    echo "  ${tag} $i: NPU ${npu}, HTTP ${port}, KV ${kv_port}"

    export_ascend_env
    export_bff_env "$role"
    export ASCEND_RT_VISIBLE_DEVICES=$npu
    local kv_cfg; kv_cfg=$(build_kv_transfer_config "$role" "$kv_port")

    nohup bash -c "python -m kv_fast_fusion.fast_fusion_main serve \"${MODEL}\" \
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
  echo "Running F1 benchmark (concurrency=${MAX_CONCURRENCY}, prompts=${NUM_PROMPTS}) against proxy ${PROXY_PORT}..."
  python -m f1_benchmark.f1_main \
    --dataset-path "${F1_DATASET}" --hf-split "${F1_SPLIT}" \
    --input-key "${F1_INPUT_KEY}" --output-key "${F1_OUTPUT_KEY}" \
    --num-prompts ${NUM_PROMPTS} --request-rate ${REQUEST_RATE} --burstiness ${BURSTINESS} \
    --max-concurrency ${MAX_CONCURRENCY} --request-timeout ${REQUEST_TIMEOUT} \
    --min-tokens ${MIN_TOKENS} --compute-f1 \
    --model "${MODEL}" --host ${VLLM_HOST_IP} --port ${PROXY_PORT} \
    --result-dir "${results_root}" \
    > "${logs_root}/${BASELINE}-${NUM_PREFILL}Px${NUM_DECODE}D-con${MAX_CONCURRENCY}-serving.txt" 2>&1
}

# Post-run BFF stats (producer dumps bff_stats_<pid>.json into results_root when BASELINE=bff).
collect_bff_stats() {
  [[ "$BFF_ON" != "1" ]] && return 0
  python3 - "$results_root" <<'PY' || true
import glob, json, os, sys
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "bff_stats_*.json")))
if not files:
    print("  bff stats: none found (fusion may not have engaged)"); raise SystemExit
B = F = 0; ov = []
for fp in files:
    try: s = json.load(open(fp))
    except Exception: continue
    B += s.get("total_blocks", 0); F += s.get("freed", 0)
    if s.get("steps"): ov.append(s.get("overhead_avg_group_dedup_ms", 0.0))
factor = B / max(1, B - F)
print(f"  bff compression: {factor:.3f}x smaller (blocks={B} freed={F}) over {len(files)} producer(s)"
      + (f" | fusion overhead {sum(ov)/len(ov):.3f} ms/group" if ov else ""))
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

  echo "BFF Ascend config: BASELINE=$BASELINE connector=$CONNECTOR ${NUM_PREFILL}Px${NUM_DECODE}D tp=$TP_SIZE"
  [[ "$BFF_ON" == "1" ]] && echo "  BFF: fuse=$BFF_PD_FUSE scale=$BFF_SCALE_MODE merge=$BFF_PD_MERGE repr=$BFF_PD_REPR thr=$BFF_THRESHOLD gs=$BFF_GROUP_SIZE eb=$BFF_PD_ENCODED_BATCH_SIZE"

  rm -rf "${logs_root}" "${results_root}"; mkdir -p "${logs_root}" "${results_root}"
  rm -f "${results_root}"/bff_stats_*.json

  kill_all_nodes
  echo "Detecting free NPUs..."; AVAILABLE_NPUS=$(get_free_npus ${MIN_FREE_MEMORY_MB}); echo "  $AVAILABLE_NPUS"

  ulimit -l unlimited || true
  launch_mooncake_master
  launch_engines "kv_producer" "$NUM_PREFILL" "$PREFILL_PORT_BASE" "$PREFILL_KV_PORT_BASE" 0 "prefill"
  launch_engines "kv_consumer" "$NUM_DECODE"  "$DECODE_PORT_BASE"  "$DECODE_KV_PORT_BASE"  "$NUM_PREFILL" "decode"
  wait_for_all_nodes
  launch_proxy

  run_benchmark
  collect_bff_stats

  echo "Benchmark done. Logs: ${logs_root}  Results: ${results_root}"
  kill_all_nodes
}

main "$@"
