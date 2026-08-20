#!/bin/bash

# example usage: NUM_PREFILL=2 NUM_DECODE=1 \
#   ./examples/online_serving/disaggregated_serving_mooncake_gpu_xpyd/disagg_bff_mooncake_gpu.sh
# Fusion ablation (BFF layout, no merge):
#   NUM_PREFILL=2 NUM_DECODE=1 BFF_PD_FUSE=0 ./…/disagg_bff_mooncake_gpu.sh
# True vanilla (stock vLLM MooncakeConnector, single KV-cache group):
#   NUM_PREFILL=2 NUM_DECODE=1 BASELINE=vanilla ./…/disagg_bff_mooncake_gpu.sh
# =============================================================================
# BFF (KV-Cache Fast Fusion) Disaggregated Serving — GPU Mooncake XpYd
# =============================================================================
# Sibling of disagg_bff_p2p_nccl_xpyd.sh: same model, same BFF knobs, same F1
# benchmark and stats collection — but the KV moves over the Mooncake Transfer
# Engine (decode PULLS via RDMA/TCP) instead of P2P NCCL (prefill PUSHES).
#
# Differences from the NCCL script, all forced by the transport:
#   * --kv_connector is MooncakeConnectorFF (registered by kv_fast_fusion's P/D
#     patch alongside P2pNcclConnectorFF).
#   * Each instance needs a unique `engine_id`; each PREFILL runs its own HTTP
#     bootstrap server (VLLM_MOONCAKE_BOOTSTRAP_PORT), which the decode queries
#     to find that prefill's workers. The proxy is told these statically —
#     Mooncake has no ZMQ service discovery, so there is no PROXY ZMQ port and
#     no ___decode_addr_ request-id rewriting.
#   * No NCCL env (VLLM_NCCL_SO_PATH / NCCL_P2P_DISABLE / custom all-reduce),
#     and no kv_buffer_size — Mooncake reads registered memory directly, so
#     there is no recv-buffer threshold to tune on D.
#   * BFF_SCALE_MODE is raw only: the redirect maps ride the transfer ACK, which
#     carries no float payload, so `ratio` has no channel here (use the NCCL
#     connector for ratio mode).
#
# Everything else — the 128 block size, prefix caching, hybrid KV-cache manager,
# the BFF group split and the F1 benchmark against the proxy — is identical, so
# a Mooncake run is directly comparable to an NCCL run of the same config.
# =============================================================================

# ---- Model / topology --------------------------------------------------------
MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-1200}
PROXY_HTTP_PORT=${PROXY_HTTP_PORT:-10001}   # proxy HTTP serving port (benchmark target)
KV_IP=${KV_IP:-10.10.10.174}
HF_HOME=${HF_HOME:-"/data/models/huggingface"}
HF_HUB_CACHE=${HF_HUB_CACHE:-"/data/models/huggingface/hub"}

NUM_PREFILL=${NUM_PREFILL:-1}        # n
NUM_DECODE=${NUM_DECODE:-1}          # m
TP=${TP:-1}                          # tensor-parallel size PER instance
HTTP_PORT_BASE=${HTTP_PORT_BASE:-20003}
# Each prefill runs its own bootstrap server; they must not collide on this host.
BOOTSTRAP_PORT_BASE=${BOOTSTRAP_PORT_BASE:-8998}
# Mooncake transport. "tcp" works on any box; "rdma" is faster but REQUIRES an RoCE/IB HCA.
# On a host with no HCA the Transfer Engine still initializes ("Found 0 HCAs") and then fails
# transfers with -1 under load, which strands decode requests in WAITING_FOR_REMOTE_KVS forever —
# the run looks like a hang, not an error. _check_rdma_capable() below refuses that combination.
MOONCAKE_PROTOCOL=${MOONCAKE_PROTOCOL:-tcp}

# Refuse rdma on a box with no RDMA device (see above). Override with
# MOONCAKE_ALLOW_RDMA_WITHOUT_HCA=1 if you know the HCA appears by another route.
_check_rdma_capable() {
    [[ "$MOONCAKE_PROTOCOL" != "rdma" ]] && return 0
    [[ "${MOONCAKE_ALLOW_RDMA_WITHOUT_HCA:-0}" == "1" ]] && return 0
    local n=0
    [[ -d /sys/class/infiniband ]] && n=$(ls -1 /sys/class/infiniband 2>/dev/null | wc -l)
    if [[ "$n" -eq 0 ]]; then
        echo "ERROR: MOONCAKE_PROTOCOL=rdma but this host has no RDMA device"
        echo "       (/sys/class/infiniband is empty — Mooncake logs 'No RDMA devices found,"
        echo "       check your device installation' / 'Found 0 HCAs' and then fails every"
        echo "       transfer with -1, hanging the decode)."
        echo "  Fix: MOONCAKE_PROTOCOL=tcp $0"
        exit 1
    fi
    echo "  RDMA: found $n device(s) in /sys/class/infiniband"
}

# Build "start,start+1,...,start+count-1".
_seq_csv() { local start=$1 count=$2 out="" k; for ((k=0; k<count; k++)); do out+="$((start+k)),"; done; echo "${out%,}"; }

# GPUs are allocated TP-per-instance, packed contiguously (same layout as the NCCL script):
# prefill i → [i*TP .. i*TP+TP-1], decode j → [NUM_PREFILL*TP + j*TP .. +TP-1].
PREFILL_GPUS=${PREFILL_GPUS:-$(_seq_csv 0 "$((NUM_PREFILL * TP))")}
DECODE_GPUS=${DECODE_GPUS:-$(_seq_csv "$((NUM_PREFILL * TP))" "$((NUM_DECODE * TP))")}
PREFILL_PORTS=${PREFILL_PORTS:-$(_seq_csv "$HTTP_PORT_BASE" "$NUM_PREFILL")}
DECODE_PORTS=${DECODE_PORTS:-$(_seq_csv "$((HTTP_PORT_BASE + NUM_PREFILL))" "$NUM_DECODE")}

# ---- Baseline mode -----------------------------------------------------------
# BASELINE=bff (default) → BFF launcher + group-aware MooncakeConnectorFF (system under test).
# BASELINE=bff_v2        → same launcher, MooncakeConnectorFFv2: the producer ships per-block
#                          signatures and the DECODE decides what not to pull, so a deduplicated
#                          block is never transferred. v1 merges on P and transfers everything
#                          anyway, which is what wedged the producer at 99.6% KV for a whole run.
# BASELINE=vanilla       → stock `vllm serve` + stock MooncakeConnector, single KV-cache group,
#                          NO BFF patches/group split — the end-to-end reference.
# NOTE: BFF_PD_FUSE=0 is the *fusion ablation* (BFF layout, no merge), NOT vanilla.
BASELINE=${BASELINE:-bff}
if [[ "$BASELINE" == "vanilla" ]]; then
    LAUNCHER="vllm.entrypoints.cli.main"
    CONNECTOR="MooncakeConnector"
    HYBRID_FLAG=""                      # stock single-group default
elif [[ "$BASELINE" == "bff_v2" ]]; then
    LAUNCHER="kv_fast_fusion.fast_fusion_main"
    CONNECTOR="MooncakeConnectorFFv2"
    HYBRID_FLAG="--no-disable-hybrid-kv-cache-manager"
elif [[ "$BASELINE" == "bff_v2_legacy" ]]; then
    # The PRE-EXTRACTION v2 (773 lines, commit 6122e3126) kept verbatim beside the current one.
    # Same knobs, same everything — the ONLY difference is the connector, which is the point:
    # extracting the shared core for Ascend cut this file to 360 lines and that refactor has never
    # been measured. Run this against BASELINE=bff_v2 to settle whether it cost the 1.42.
    LAUNCHER="kv_fast_fusion.fast_fusion_main"
    CONNECTOR="MooncakeConnectorFFv2Legacy"
    HYBRID_FLAG="--no-disable-hybrid-kv-cache-manager"
else
    LAUNCHER="kv_fast_fusion.fast_fusion_main"
    CONNECTOR="MooncakeConnectorFF"
    HYBRID_FLAG="--no-disable-hybrid-kv-cache-manager"
fi
ENABLE_CHUNKED=${ENABLE_CHUNKED:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-}
if [[ "${VLLM_BATCH_INVARIANT:-0}" == "1" && -z "$ATTENTION_BACKEND" ]]; then
    ATTENTION_BACKEND=FLASH_ATTN
fi
if [[ "$ENABLE_CHUNKED" == "1" ]]; then
    CHUNKED_FLAG="--enable-chunked-prefill"
else
    CHUNKED_FLAG="--no-enable-chunked-prefill"
    # With chunked prefill OFF, vLLM rejects max_num_batched_tokens < max_model_len.
    if (( MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
        MAX_NUM_BATCHED_TOKENS=$MAX_MODEL_LEN
    fi
fi

# ---- BFF knobs ---------------------------------------------------------------
BFF_PD_MERGE=${BFF_PD_MERGE:-cc}         # within-batch clustering (cc, nr_tree)
BFF_SCALE_MODE=raw                       # the only mode this transport supports (see header)
BFF_PD_REPR=${BFF_PD_REPR:-proj}         # full | proj (LSH) | mean
BFF_PD_FUSE=${BFF_PD_FUSE:-1}            # connector-level fusion + redirect propagation
BFF_GROUP_SIZE=${BFF_GROUP_SIZE:-4}      # fusion layers packed per KV cache group
BFF_THRESHOLD=${BFF_THRESHOLD:-0.75}     # fusion threshold (0.0-1.0)
BFF_PD_ENCODED_BATCH_SIZE=${BFF_PD_ENCODED_BATCH_SIZE:-32}   # cross-batch window (0 = within-batch)
# Restrict fusion to the KV-cache groups that actually pay (comma list, e.g. "1,2,3"). Unset = all.
# The 2026-08-13 run measured g1=171.6x g2=1.9x g3=4.8x but g4=1.01x g5=1.45x g6=1.01x — the deep
# groups pay full per-layer repr + clustering cost for nothing. "none" keeps the group split and all
# BFF patches active while fusing nothing (control arm isolating split cost from fusion cost).
BFF_FF_GROUPS=${BFF_FF_GROUPS:-}
# Cross-request index: "matrix" (dense FIFO window of BFF_PD_ENCODED_BATCH_SIZE requests, default)
# or "lsh" (SimHash banded index, pool bounded by BFF_LSH_MAX_ENTRIES instead of a request window,
# so far more requests can match; TP=1 only, falls back to matrix under TP>1).
BFF_PD_CROSS_INDEX=${BFF_PD_CROSS_INDEX:-matrix}
BFF_LSH_TABLES=${BFF_LSH_TABLES:-16}                 # banded sub-hashes per block
BFF_LSH_BITS_PER_TABLE=${BFF_LSH_BITS_PER_TABLE:-20} # bits per band (16x20: ~87% recall at cos 0.95)
BFF_LSH_MAX_ENTRIES=${BFF_LSH_MAX_ENTRIES:-10000}    # reps per group before LRU-evicting the oldest half
BFF_LSH_MAX_CANDIDATES=${BFF_LSH_MAX_CANDIDATES:-0}  # 0 = verify every bucket candidate
# Max distinct owning REQUESTS in the index. The real bound: a rep is only usable while its KV is
# still resident on D, and D reports finished ids back on the pull request.
# Neither mechanism does much until the pool actually grows: measured at 18 reps / 17 owners, this
# cap never fires, and only 1 rep was retired by D's feedback all run — every pull request is issued
# during the initial admission burst while requests finish much later, so with
# NUM_PROMPTS <= MAX_CONCURRENCY there is no steady stream of pulls to piggyback on. Both become
# live once the threshold is raised enough to unfreeze registration (see BFF_THRESHOLD_G) and
# NUM_PROMPTS exceeds concurrency. The channel is not broken; it has nothing to carry yet.
# (a literal, not $MAX_CONCURRENCY — that is defined further down; keep it >= decode concurrency)
BFF_LSH_MAX_LIVE_OWNERS=${BFF_LSH_MAX_LIVE_OWNERS:-512}
# Per-group merge thresholds ("1:0.97,2:0.90"), overriding BFF_THRESHOLD for those groups. Each
# group has its own similarity FLOOR (group 1 = the first attention layers, whose keys share a large
# common component and sit near cosine 0.9 even for unrelated blocks), so one global bar merges
# everything in some groups and nothing in others. Measure the floor with BFF_PD_AUDIT=1 first.
BFF_THRESHOLD_G=${BFF_THRESHOLD_G:-}
# ---- v2 knobs (BASELINE=bff_v2 only) -----------------------------------------
# Master switch. 0 disables the WHOLE mechanism including the signature phase — the pull goes back
# to one round trip, so this is the "BFF group split, no fusion" control arm. It is NOT a
# measurement of what the signature exchange costs; for that, keep this at 1 and set
# BFF_THRESHOLD=1.01 so the exchange happens and every candidate is rejected.
BFF_V2_DEDUP=${BFF_V2_DEDUP:-1}
# Alias to blocks left over from EARLIER pulls, not just duplicates within the current one. This is
# where cross-request reuse comes from; it is safe because the decode's block pool notifies the
# index on every release (preemption included), so a block leaves it before it can be reallocated.
BFF_V2_RESIDENT=${BFF_V2_RESIDENT:-1}
# Signature width in fp16 dims. ~256 B per block against ~1.6 MB of KV, i.e. 0.02% of the wire.
BFF_SIG_DIM=${BFF_SIG_DIM:-128}
# Ceiling on the relative substitution error a merge may inject:
#   rel_err = ||k_owner - k_rep|| / ||k_owner|| = sqrt(1 + r^2 - 2*r*cos),  r = |k_rep|/|k_owner|
# This, not BFF_THRESHOLD, is what governs accuracy — cosine is scale-free, so a pair can clear any
# cosine bar and still be a bad substitution because the magnitudes differ. 1.0 is inert (pure
# cosine, what every run before this used). Note min_r rel_err = sqrt(1-cos^2), so a 0.20 budget
# implies cos >= 0.98 no matter what the norms do.
# Measured on the thr0.75 run: only 3% of its 36,498 merges were under 0.20 and 5% under 0.30,
# while ngram_match fell 45% — the merges were mostly noise, and this is the knob that says no.
BFF_MAX_REL_ERR=${BFF_MAX_REL_ERR:-1.0}
# Blocks held per group in the decode's dedup index (an "owner" here is a block, not a request).
BFF_V2_MAX_RESIDENT=${BFF_V2_MAX_RESIDENT:-32768}
# Phase 1 is best-effort: a producer slower than this gets its request pulled in full, never stalls.
BFF_V2_SIG_TIMEOUT=${BFF_V2_SIG_TIMEOUT:-10}
BFF_V2_READY_TIMEOUT=${BFF_V2_READY_TIMEOUT:-10}
# Similarity audit: sample random cross-request block pairs for the first N fusion steps per group
# and report their cosine quantiles. Off by default; costs nothing after those steps.
BFF_PD_AUDIT=${BFF_PD_AUDIT:-0}
BFF_PD_AUDIT_STEPS=${BFF_PD_AUDIT_STEPS:-8}
BFF_PD_AUDIT_PAIRS=${BFF_PD_AUDIT_PAIRS:-512}

# ---- GPU memory --------------------------------------------------------------
# No kv_buffer_size here: Mooncake transfers straight out of registered KV memory, so D has no
# recv-buffer threshold to tune (the NCCL script's DECODE_KV_BUFFER has no counterpart).
PREFILL_GPU_UTIL=${PREFILL_GPU_UTIL:-0.85}
DECODE_GPU_UTIL=${DECODE_GPU_UTIL:-0.85}

# ---- F1 accuracy + latency benchmark knobs -----------------------------------
F1_DATASET=${F1_DATASET:-ise-uiuc/Magicoder-Evol-Instruct-110K} #m-a-p/CodeFeedback-Filtered-Instruction
F1_SPLIT=${F1_SPLIT:-train}
F1_INPUT_KEY=${F1_INPUT_KEY:-instruction}
F1_OUTPUT_KEY=${F1_OUTPUT_KEY:-response}
NUM_PROMPTS=${NUM_PROMPTS:-500}
MAX_CONCURRENCY=${MAX_CONCURRENCY:-150}
REQUEST_RATE=${REQUEST_RATE:-300}
BURSTINESS=${BURSTINESS:-0.3}
MIN_TOKENS=${MIN_TOKENS:-1536}
MAX_TOKENS=${MAX_TOKENS:-8192}
if (( MAX_TOKENS >= MAX_MODEL_LEN )); then
    _safe=$(( MAX_MODEL_LEN - 8192 )); (( _safe < 1024 )) && _safe=1024
    echo "  WARNING: MAX_TOKENS ($MAX_TOKENS) >= max_model_len ($MAX_MODEL_LEN): the server rejects"
    echo "           EVERY request (no room for the prompt). Clamping MAX_TOKENS to $_safe."
    MAX_TOKENS=$_safe
fi
# Compression only pays when the KV cache is FULL, so a run that never holds saturation measures
# nothing about it. With NUM_PROMPTS <= MAX_CONCURRENCY every request is admitted at once and the
# whole run is ramp + drain, with no steady state at all — whole-run means then look far below the
# peak (a 500/512 run held >=95% for 10 of 17 samples but averaged 43%). Not clamped: a longer run
# costs real time, so the choice stays yours.
if (( NUM_PROMPTS < 2 * MAX_CONCURRENCY )); then
    echo "  WARNING: NUM_PROMPTS ($NUM_PROMPTS) < 2x MAX_CONCURRENCY ($MAX_CONCURRENCY): the run is"
    echo "           mostly ramp-up and drain. For saturation numbers use NUM_PROMPTS>=$((4 * MAX_CONCURRENCY))."
fi
REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-1200}
RESULT_DIR=${RESULT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/f1_results}
if [[ "$BASELINE" == "vanilla" ]]; then
    RUN_TAG=${RUN_TAG:-mooncake_vanilla_con_${MAX_CONCURRENCY}_${NUM_PREFILL}Px${NUM_DECODE}D${RUN_SET:+_s${RUN_SET}}${RUN_REPEAT:+_r${RUN_REPEAT}}}
elif [[ "$BASELINE" == "bff_v2" || "$BASELINE" == "bff_v2_legacy" ]]; then
    # Both v2 variants share the tag shape; `_legacy` distinguishes them so the A/B lands in two
    # files. Without it the pre- and post-extraction runs would overwrite each other, which is the
    # exact failure that lost the 1.42 evidence in the first place.
    [[ "$BASELINE" == "bff_v2_legacy" ]] && _V2_VARIANT="_legacy" || _V2_VARIANT=""
    # v2's knobs are different ones (no merge mode, no encoded batch — the decode decides per pull),
    # so it gets its own tag shape rather than a v1 tag that would misdescribe the run.
    # `_re` is BFF_MAX_REL_ERR and is NOT optional: it governs the whole accuracy/compression trade
    # (0.3 -> 6% saving, 1.0 -> 61%), so two runs that differ only in it are different experiments.
    # It was missing until 2026-08-19, which means every thr0.8 run silently overwrote its
    # predecessor's .json and .log — the most likely reason the 1.42 result no longer exists on disk.
    RUN_TAG=${RUN_TAG:-mooncake_v2${_V2_VARIANT}_thr${BFF_THRESHOLD}_re${BFF_MAX_REL_ERR}_gs${BFF_GROUP_SIZE}_sig${BFF_SIG_DIM}_dedup${BFF_V2_DEDUP}_res${BFF_V2_RESIDENT}${BFF_FF_GROUPS:+_g${BFF_FF_GROUPS//,/}}_con_${MAX_CONCURRENCY}_${NUM_PREFILL}Px${NUM_DECODE}D${RUN_SET:+_s${RUN_SET}}${RUN_REPEAT:+_r${RUN_REPEAT}}}
else
    RUN_TAG=${RUN_TAG:-mooncake_${BFF_PD_MERGE}_${BFF_PD_REPR}_thr${BFF_THRESHOLD}_gs${BFF_GROUP_SIZE}_eb${BFF_PD_ENCODED_BATCH_SIZE}_${BFF_PD_CROSS_INDEX}${BFF_FF_GROUPS:+_g${BFF_FF_GROUPS//,/}}_con_${MAX_CONCURRENCY}_${NUM_PREFILL}Px${NUM_DECODE}D${RUN_SET:+_s${RUN_SET}}${RUN_REPEAT:+_r${RUN_REPEAT}}}
fi

# ---- Required BFF / HF environment (CLAUDE.md) -------------------------------
REPO_ROOT=${REPO_ROOT:-/data/users/sefi/from_git/vllm_013/vllm_ff}
export HF_HOME=${HF_HOME:-/data/models/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-/data/models/huggingface/hub}
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
export VLLM_USE_V1=1
export PYTHONPATH=${REPO_ROOT}:${PYTHONPATH}

# ---- Mooncake TCP: keep the KV plane on loopback ------------------------------------------
# Mooncake's TCP transport opens a fresh connection per transfer and does not pool them. At high
# concurrency that burns the ephemeral port range, and every further connect() fails:
#     E tcp_transport.cpp] TcpTransport::getConnection failed to create connection to <ip>:<port>.
#         Error: connect: Cannot assign requested address
#   -> batch_transfer_sync_write returns -1 -> "pulling kv_caches for [...] failed".
# (2026-08-13: 41,111 of these in one prefill log; the run wedged.) This box has ~28k ephemeral
# ports (32768-60999) with a 60s TIME_WAIT, and net.ipv4.tcp_tw_reuse=2 — which recycles TIME_WAIT
# ports for LOOPBACK destinations ONLY. P and D are always on the same host here, so pinning the KV
# plane to 127.0.0.1 (instead of the routable KV_IP) is what makes that reuse apply.
# NOTE: this only affects vLLM's own get_ip() (transfer engine + worker side channel). The proxy
# still reaches the servers on $KV_IP, and the bootstrap servers bind 0.0.0.0.
# Set VLLM_HOST_IP yourself for a genuine multi-host run — then see the sysctl note below instead.
export VLLM_HOST_IP=${VLLM_HOST_IP:-127.0.0.1}
# Bound the producer's concurrent transfers (vLLM default 10 workers / 20 in-flight tasks). Fewer
# workers = fewer simultaneous connections = less port pressure, at some transfer parallelism.
MOONCAKE_NUM_WORKERS=${MOONCAKE_NUM_WORKERS:-10}
# What to do when a KV pull fails. vLLM defaults to "fail", which drops the affected requests
# outright; "recompute" makes the decode prefill those prompts locally instead, so a transport
# hiccup costs latency rather than completions or accuracy — strictly better for a benchmark.
KV_LOAD_FAILURE_POLICY=${KV_LOAD_FAILURE_POLICY:-recompute}

_check_ephemeral_ports() {
    [[ "$MOONCAKE_PROTOCOL" != "tcp" ]] && return 0
    local range lo hi n reuse
    range=$(sysctl -n net.ipv4.ip_local_port_range 2>/dev/null) || return 0
    lo=$(echo "$range" | awk '{print $1}'); hi=$(echo "$range" | awk '{print $2}')
    n=$((hi - lo)); reuse=$(sysctl -n net.ipv4.tcp_tw_reuse 2>/dev/null)
    echo "  TCP: ${n} ephemeral ports (${lo}-${hi}), tcp_tw_reuse=${reuse}, KV plane on $VLLM_HOST_IP"
    if [[ "$VLLM_HOST_IP" != 127.* && "$reuse" != "1" ]]; then
        echo "  WARNING: KV plane is NOT on loopback and tcp_tw_reuse != 1 — Mooncake's per-transfer"
        echo "           connections can exhaust the ephemeral range under load and fail with -1."
        echo "           Root fix: sysctl -w net.ipv4.tcp_tw_reuse=1"
        echo "                     sysctl -w net.ipv4.ip_local_port_range='1024 65535'"
    fi
}

echo "BFF Mooncake (GPU) Disaggregated Configuration:"
echo "  Model:        $MODEL"
echo "  Topology:     ${NUM_PREFILL}P x ${NUM_DECODE}D   (TP=$TP per instance)"
echo "  Prefill GPUs: $PREFILL_GPUS, Ports: $PREFILL_PORTS  (bootstrap ${BOOTSTRAP_PORT_BASE}+)"
echo "  Decode GPUs:  $DECODE_GPUS, Ports: $DECODE_PORTS"
echo "  Proxy:        HTTP $PROXY_HTTP_PORT   host $KV_IP"
echo "  Transport:    mooncake protocol=$MOONCAKE_PROTOCOL"
echo "  Baseline:     $BASELINE   (launcher=$LAUNCHER  connector=$CONNECTOR  chunked_prefill=$ENABLE_CHUNKED)"
echo "  Traffic:      MAX_CONCURRENCY=$MAX_CONCURRENCY  REQUEST_RATE=$REQUEST_RATE  BURSTINESS=$BURSTINESS"
if [[ "$BASELINE" == "vanilla" ]]; then
echo "  BFF:          (disabled — stock vLLM single-group reference)"
else
echo "  BFF:          BFF_PD_MERGE=$BFF_PD_MERGE  BFF_SCALE_MODE=$BFF_SCALE_MODE  BFF_PD_REPR=$BFF_PD_REPR  BFF_PD_FUSE=$BFF_PD_FUSE  BFF_THRESHOLD=$BFF_THRESHOLD  BFF_GROUP_SIZE=$BFF_GROUP_SIZE  ENCODED_BATCH=$BFF_PD_ENCODED_BATCH_SIZE"
echo "  BFF fusion:   cross_index=$BFF_PD_CROSS_INDEX  ff_groups=${BFF_FF_GROUPS:-<all>}  lsh=${BFF_LSH_TABLES}x${BFF_LSH_BITS_PER_TABLE} max_entries=$BFF_LSH_MAX_ENTRIES"
fi
echo "  GPU util:     P=$PREFILL_GPU_UTIL  D=$DECODE_GPU_UTIL"
echo ""

PIDS=()
cd "$(dirname "${BASH_SOURCE[0]}")"

check_required_files() {
    if [[ ! -f "disagg_proxy_mooncake_gpu.py" ]]; then
        echo "Required file disagg_proxy_mooncake_gpu.py not found in $(pwd)"
        exit 1
    fi
    if ! python3 -c "import mooncake.engine" 2>/dev/null; then
        echo "The 'mooncake' transfer engine is not importable by python3."
        echo "Install it (pip install mooncake-transfer-engine) or run the NCCL script instead."
        exit 1
    fi
}

check_num_gpus() {
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    local need=$(((NUM_PREFILL + NUM_DECODE) * TP))
    if [ "$num_gpus" -lt "$need" ]; then
        echo "You need at least $need GPUs ((${NUM_PREFILL}P + ${NUM_DECODE}D) × TP=$TP); found $num_gpus."
        exit 1
    fi
    echo "Found $num_gpus GPUs (using $need: (${NUM_PREFILL}P + ${NUM_DECODE}D) × TP=$TP)."
}

_kill_stale_gpu_procs() {
    # vLLM WorkerProcs (TP>1, spawn) can survive `kill -- -$$` once their parent dies first — they
    # get reparented to init, squat GPU memory, and the NEXT launch fails its startup free-memory
    # check. Scope strictly to THIS repo: the box is shared and other users' vLLM servers must
    # never be touched, so match /proc/<pid>/exe OR cwd under $REPO_ROOT rather than by name.
    local pid exe cwd killed=""
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
        exe=$(readlink "/proc/$pid/exe" 2>/dev/null)
        cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null)
        if [[ "$exe" == "$REPO_ROOT"/* || "$cwd" == "$REPO_ROOT"/* ]]; then
            echo "Killing stale GPU process $pid (exe=$exe cwd=$cwd)"
            kill -9 "$pid" 2>/dev/null
            killed+=" $pid"
        fi
    done
    # kill -9 returns before the driver tears the CUDA context down; launching immediately races
    # the next run's startup free-memory check. Wait for the pids to vanish, then let it settle.
    if [ -n "$killed" ]; then
        local waited=0 live still=""
        while [ "$waited" -lt 30 ]; do
            live=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
            still=""
            for pid in $killed; do
                grep -qw "$pid" <<< "$live" && still+=" $pid"
            done
            [ -z "$still" ] && break
            sleep 1
            waited=$((waited + 1))
        done
        [ -n "$still" ] && echo "WARNING: GPU memory of killed pids not released after ${waited}s:${still}"
        sleep 2
    fi
}

cleanup() {
    echo "Stopping everything…"
    trap - INT TERM
    pkill -9 -f "disagg_proxy_mooncake_gpu.py"
    # Sweep BEFORE the group kill: `kill -- -$$` also SIGTERMs this script.
    _kill_stale_gpu_procs
    kill -- -$$
    wait
    exit 0
}

wait_for_server() {
    local port=$1
    local start_time=$(date +%s)
    echo "Waiting for server on port $port..."
    while true; do
        if curl -s "localhost:${port}/v1/completions" > /dev/null; then
            echo "Server on port $port is ready."
            return 0
        fi
        local now=$(date +%s)
        if (( now - start_time >= TIMEOUT_SECONDS )); then
            echo "Timeout waiting for server on port $port"
            return 1
        fi
        sleep 1
    done
}

common_args() {
    echo "--host 0.0.0.0 \
        --tensor-parallel-size $TP \
        --seed 1024 \
        --trust-remote-code \
        --block-size 128 \
        --enable-prefix-caching \
        $HYBRID_FLAG \
        $CHUNKED_FLAG \
        --max-model-len $MAX_MODEL_LEN \
        --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
        ${ATTENTION_BACKEND:+--attention-backend $ATTENTION_BACKEND} \
        --max-num-seqs $MAX_CONCURRENCY"
}

main() {
    check_required_files
    check_num_gpus
    _check_rdma_capable
    _check_ephemeral_ports

    trap cleanup INT
    trap cleanup USR1
    trap cleanup TERM

    echo "Launching BFF Mooncake disaggregated serving components..."
    echo "Logs: prefill*.log / decode*.log / proxy.log"

    _kill_stale_gpu_procs

    # Rotate last run's logs instead of truncating: a crashed run's root cause must survive
    # one relaunch.
    local f
    for f in prefill*.log decode*.log proxy.log; do
        [ -f "$f" ] && mv -f "$f" "$f.prev"
    done

    IFS=',' read -ra PREFILL_GPU_ARRAY <<< "$PREFILL_GPUS"
    IFS=',' read -ra DECODE_GPU_ARRAY <<< "$DECODE_GPUS"
    IFS=',' read -ra PREFILL_PORT_ARRAY <<< "$PREFILL_PORTS"
    IFS=',' read -ra DECODE_PORT_ARRAY <<< "$DECODE_PORTS"

    # ---- Prefill servers (producers) -----------------------------------------
    # Producers dump their cumulative fuse overhead + compression to bff_stats_<pid>.json here
    # (read back post-run). Clean stale files first.
    mkdir -p "$RESULT_DIR"
    rm -f "$RESULT_DIR"/bff_stats_*.json
    echo ""
    echo "Starting ${#PREFILL_PORT_ARRAY[@]} prefill server(s)..."
    local proxy_prefill_addrs="" proxy_engine_ids="" proxy_bootstraps=""
    for ((i=0; i<${#PREFILL_PORT_ARRAY[@]}; i++)); do
        local gpu_id=$(IFS=','; echo "${PREFILL_GPU_ARRAY[*]:i*TP:TP}")
        local port=${PREFILL_PORT_ARRAY[$i]}
        local bport=$((BOOTSTRAP_PORT_BASE + i))
        local eid="prefill-$i"

        echo "  Prefill $((i+1)): GPU $gpu_id, HTTP $port, bootstrap $bport, engine_id $eid"
        proxy_prefill_addrs+="${KV_IP}:${port},"
        proxy_engine_ids+="${eid},"
        proxy_bootstraps+="http://${KV_IP}:${bport},"

        CUDA_VISIBLE_DEVICES=$gpu_id \
        VLLM_MOONCAKE_BOOTSTRAP_PORT=$bport \
        BFF_PD_MERGE=$BFF_PD_MERGE BFF_SCALE_MODE=$BFF_SCALE_MODE BFF_PD_REPR=$BFF_PD_REPR BFF_PD_FUSE=$BFF_PD_FUSE \
        BFF_GROUP_SIZE=$BFF_GROUP_SIZE BFF_THRESHOLD=$BFF_THRESHOLD \
        BFF_PD_ENCODED_BATCH_SIZE=$BFF_PD_ENCODED_BATCH_SIZE \
        BFF_FF_GROUPS=$BFF_FF_GROUPS BFF_PD_CROSS_INDEX=$BFF_PD_CROSS_INDEX \
        BFF_LSH_TABLES=$BFF_LSH_TABLES BFF_LSH_BITS_PER_TABLE=$BFF_LSH_BITS_PER_TABLE \
        BFF_LSH_MAX_ENTRIES=$BFF_LSH_MAX_ENTRIES BFF_LSH_MAX_CANDIDATES=$BFF_LSH_MAX_CANDIDATES \
        BFF_LSH_MAX_LIVE_OWNERS=$BFF_LSH_MAX_LIVE_OWNERS BFF_THRESHOLD_G=$BFF_THRESHOLD_G \
        BFF_PD_AUDIT=$BFF_PD_AUDIT BFF_PD_AUDIT_STEPS=$BFF_PD_AUDIT_STEPS \
        BFF_PD_AUDIT_PAIRS=$BFF_PD_AUDIT_PAIRS \
        BFF_V2_DEDUP=$BFF_V2_DEDUP BFF_V2_RESIDENT=$BFF_V2_RESIDENT BFF_SIG_DIM=$BFF_SIG_DIM \
        BFF_V2_MAX_RESIDENT=$BFF_V2_MAX_RESIDENT \
        BFF_V2_SIG_TIMEOUT=$BFF_V2_SIG_TIMEOUT BFF_V2_READY_TIMEOUT=$BFF_V2_READY_TIMEOUT \
        BFF_MAX_REL_ERR=$BFF_MAX_REL_ERR \
        BFF_PD_STATS_DIR="$RESULT_DIR" \
        python3 -m $LAUNCHER serve $MODEL \
        $(common_args) \
        --port $port \
        --gpu-memory-utilization $PREFILL_GPU_UTIL \
        --kv-transfer-config \
        "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"kv_producer\",\"engine_id\":\"$eid\",\"kv_load_failure_policy\":\"$KV_LOAD_FAILURE_POLICY\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$MOONCAKE_PROTOCOL\",\"num_workers\":$MOONCAKE_NUM_WORKERS}}" \
        > prefill$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # ---- Decode servers (consumers) ------------------------------------------
    echo ""
    echo "Starting ${#DECODE_PORT_ARRAY[@]} decode server(s)..."
    local proxy_decode_addrs=""
    for ((i=0; i<${#DECODE_PORT_ARRAY[@]}; i++)); do
        local gpu_id=$(IFS=','; echo "${DECODE_GPU_ARRAY[*]:i*TP:TP}")
        local port=${DECODE_PORT_ARRAY[$i]}
        local eid="decode-$i"

        echo "  Decode $((i+1)): GPU $gpu_id, HTTP $port, engine_id $eid"
        proxy_decode_addrs+="${KV_IP}:${port},"

        CUDA_VISIBLE_DEVICES=$gpu_id \
        BFF_PD_MERGE=$BFF_PD_MERGE BFF_SCALE_MODE=$BFF_SCALE_MODE BFF_PD_REPR=$BFF_PD_REPR BFF_PD_FUSE=$BFF_PD_FUSE \
        BFF_GROUP_SIZE=$BFF_GROUP_SIZE BFF_THRESHOLD=$BFF_THRESHOLD \
        BFF_PD_ENCODED_BATCH_SIZE=$BFF_PD_ENCODED_BATCH_SIZE \
        BFF_FF_GROUPS=$BFF_FF_GROUPS BFF_PD_CROSS_INDEX=$BFF_PD_CROSS_INDEX \
        BFF_LSH_TABLES=$BFF_LSH_TABLES BFF_LSH_BITS_PER_TABLE=$BFF_LSH_BITS_PER_TABLE \
        BFF_LSH_MAX_ENTRIES=$BFF_LSH_MAX_ENTRIES BFF_LSH_MAX_CANDIDATES=$BFF_LSH_MAX_CANDIDATES \
        BFF_V2_DEDUP=$BFF_V2_DEDUP BFF_V2_RESIDENT=$BFF_V2_RESIDENT BFF_SIG_DIM=$BFF_SIG_DIM \
        BFF_V2_MAX_RESIDENT=$BFF_V2_MAX_RESIDENT \
        BFF_V2_SIG_TIMEOUT=$BFF_V2_SIG_TIMEOUT BFF_V2_READY_TIMEOUT=$BFF_V2_READY_TIMEOUT \
        BFF_MAX_REL_ERR=$BFF_MAX_REL_ERR \
        BFF_PD_STATS_DIR="$RESULT_DIR" \
        python3 -m $LAUNCHER serve $MODEL \
        $(common_args) \
        --port $port \
        --gpu-memory-utilization $DECODE_GPU_UTIL \
        --kv-transfer-config \
        "{\"kv_connector\":\"$CONNECTOR\",\"kv_role\":\"kv_consumer\",\"engine_id\":\"$eid\",\"kv_load_failure_policy\":\"$KV_LOAD_FAILURE_POLICY\",\"kv_connector_extra_config\":{\"mooncake_protocol\":\"$MOONCAKE_PROTOCOL\",\"num_workers\":$MOONCAKE_NUM_WORKERS}}" \
        > decode$((i+1)).log 2>&1 &
        PIDS+=($!)
    done

    # ---- Wait for all servers ------------------------------------------------
    echo ""
    echo "Waiting for all servers to start..."
    for port in "${PREFILL_PORT_ARRAY[@]}" "${DECODE_PORT_ARRAY[@]}"; do
        if ! wait_for_server $port; then
            echo "Failed to start server on port $port"
            cleanup
            exit 1
        fi
    done

    # ---- Proxy ----------------------------------------------------------------
    # Started AFTER the servers: it is configured statically with their addresses / engine ids /
    # bootstrap addresses (Mooncake has no service discovery to wait on).
    echo ""
    echo "Starting proxy server (HTTP $PROXY_HTTP_PORT)..."
    PREFILL_ADDRS="${proxy_prefill_addrs%,}" \
    PREFILL_ENGINE_IDS="${proxy_engine_ids%,}" \
    PREFILL_BOOTSTRAPS="${proxy_bootstraps%,}" \
    DECODE_ADDRS="${proxy_decode_addrs%,}" \
    PROXY_HTTP_PORT="$PROXY_HTTP_PORT" \
    python3 disagg_proxy_mooncake_gpu.py > proxy.log 2>&1 &
    PIDS+=($!)
    sleep 3

    echo ""
    echo "All servers up. Running F1 benchmark against proxy HTTP $PROXY_HTTP_PORT..."
    echo "  run tag: $RUN_TAG   →   $RESULT_DIR/f1_${RUN_TAG}.{json,log}"
    mkdir -p "$RESULT_DIR"

    # ---- F1 + latency benchmark (targets the PROXY, not a server) ------------
    python3 -m f1_benchmark.f1_main \
        --model $MODEL --host $KV_IP --port $PROXY_HTTP_PORT \
        --dataset-path $F1_DATASET --hf-split $F1_SPLIT \
        --input-key $F1_INPUT_KEY --output-key $F1_OUTPUT_KEY \
        --num-prompts $NUM_PROMPTS --max-concurrency $MAX_CONCURRENCY \
        --request-rate $REQUEST_RATE --burstiness $BURSTINESS \
        --min-tokens $MIN_TOKENS --max-tokens $MAX_TOKENS \
        --request-timeout $REQUEST_TIMEOUT \
        --compute-f1 --compute-code-metrics --result-dir "$RESULT_DIR" \
        --result-file "$RESULT_DIR/f1_${RUN_TAG}.json" \
        --label "$RUN_TAG" 2>&1 | tee "$RESULT_DIR/f1_${RUN_TAG}.log"

    # ---- Collect BFF metrics into the JSON (all POST-run → ZERO impact on the measurement) ----
    # Both transports emit the same stats shapes, so this is the NCCL script's collector, lifted
    # into a shared module rather than duplicated as a second heredoc.
    # Invoked by PATH, not `-m`: importing the kv_fast_fusion package would re-apply the whole
    # BFF patch (and pull in vLLM) just to parse a few JSON files.
    python3 "$REPO_ROOT/kv_fast_fusion/tools/collect_bff_stats.py" \
        "$RESULT_DIR/f1_${RUN_TAG}.json" "$RESULT_DIR" prefill*.log decode*.log

    echo "Benchmarking done. Cleaning up..."
    cleanup
}

main
