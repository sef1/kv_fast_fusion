#!/bin/bash
# LOCAL fix (not a wiki patch): make the stock `P2pNcclConnector` usable as a P/D baseline.
#
# Part A — stable cross-P/D tensor id. vLLM v1 `InputProcessor.assign_request_id` sets
#   `request_id = f"{external_req_id}-{random_uuid():.8}"` on EACH server independently. The disagg
#   proxy gives P and D the same `external_req_id` (X-Request-Id), but each tacks on its OWN random
#   suffix, so a tensor_id keyed on the full id NEVER matches across P/D. `P2pNcclEngine.recv_tensor`
#   under PUT/PUT_ASYNC is an unbounded `while tensor_id not in recv_store: cv.wait()` — so the
#   consumer's FIRST forward step blocks forever: zero engine-stats lines, producer KV piling up
#   undrained ("Out Of Threshold"), client hangs to its request timeout.
#   (Ported from the v0.14.0 local patch. `P2pNcclConnectorFF._pd_key` is the same fix.)
#
# Part B — chunked-prefill continuation. Stock asserts `new_block_ids is not None` in the chunked
#   continuation branches; at high concurrency prompts share the batched-token budget and genuinely
#   split, and a continuation step that allocates no new block passes None -> AssertionError kills
#   EngineCore. Mirrors P2pNcclConnectorFF.build_connector_meta.
#
# Idempotent (self-detects prior patch). Override the target with VLLM_DIR; by default this probes
# the local .venv site-packages first, then the container layout /vllm-workspace/vllm.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL="vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py"

if [ -n "$VLLM_DIR" ]; then
    CONNECTOR_PATH="${VLLM_DIR}/${REL}"
else
    # Probe: repo-local venv (site-packages has no `vllm/` source-tree prefix quirk — the package
    # dir IS `vllm/`, so the same REL works), then the container's vLLM source tree.
    CONNECTOR_PATH=""
    for cand in "$HERE"/../../.venv/lib/python*/site-packages/"$REL" \
                /vllm-workspace/vllm/"$REL"; do
        if [ -f "$cand" ]; then
            CONNECTOR_PATH="$cand"
            break
        fi
    done
fi

if [ -z "$CONNECTOR_PATH" ] || [ ! -f "$CONNECTOR_PATH" ]; then
    echo "Error: p2p_nccl_connector.py not found (set VLLM_DIR to the vLLM root)."
    exit 1
fi

echo "Target: $CONNECTOR_PATH"

if [ ! -f "${CONNECTOR_PATH}.bak_stable_id" ]; then
    cp "$CONNECTOR_PATH" "${CONNECTOR_PATH}.bak_stable_id"
    echo "Created backup at ${CONNECTOR_PATH}.bak_stable_id"
fi

CONNECTOR_PATH="$CONNECTOR_PATH" python3 << 'PYTHON_EOF'
import os
import sys

PATH = os.environ["CONNECTOR_PATH"]

with open(PATH) as f:
    content = f.read()

if "_pd_stable_id" in content:
    print("File appears to already be patched!")
    raise SystemExit(0)

failed = []


def sub(old, new, label):
    """Replace `old` with `new` exactly once; record a failure if the anchor is missing/ambiguous."""
    global content
    n = content.count(old)
    if n != 1:
        failed.append(f"{label} (anchor matched {n} times, expected 1)")
        return
    content = content.replace(old, new)
    print(f"Applied: {label}")


# ---------------------------------------------------------------- Part A: stable cross-P/D id

sub(
    '''@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):''',
    '''def _pd_stable_id(request_id: str) -> str:
    """Strip the per-server random 8-hex suffix that `InputProcessor.assign_request_id` appends
    (`request_id = f"{external_req_id}-{random_uuid():.8}"`). The disagg proxy gives P and D the
    same `external_req_id`, but each server tacks on its OWN suffix — so a tensor_id keyed on the
    full id never matches across P/D and the consumer's recv_tensor blocks forever. Key transfers
    on the stable (shared) id instead. (Mirrors `P2pNcclConnectorFF._pd_key`.)"""
    return request_id.rsplit("-", 1)[0]


@dataclass
class P2pNcclConnectorMetadata(KVConnectorMetadata):''',
    "A1/4 _pd_stable_id helper",
)

sub(
    '''                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    request.request_id + "#" + layer_name, remote_address
                )''',
    '''                kv_cache = self.p2p_nccl_engine.recv_tensor(
                    _pd_stable_id(request.request_id) + "#" + layer_name, remote_address
                )''',
    "A2/4 recv_tensor keyed on stable id",
)

sub(
    '''            self.p2p_nccl_engine.send_tensor(
                request_id + "#" + layer_name, kv_cache, remote_address
            )''',
    '''            self.p2p_nccl_engine.send_tensor(
                _pd_stable_id(request_id) + "#" + layer_name, kv_cache, remote_address
            )''',
    "A3/4 send_tensor keyed on stable id",
)

sub(
    '''        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        return self.p2p_nccl_engine.get_finished(finished_req_ids, no_compile_layers)''',
    '''        no_compile_layers = self._vllm_config.compilation_config.static_forward_context
        # Free recv_store/pool entries under the SAME stable id used for send/recv, else the
        # engine reconstructs `full_id#layer` (never stored) and the spilled KV leaks → OOM.
        stable_ids = {_pd_stable_id(r) for r in finished_req_ids}
        return self.p2p_nccl_engine.get_finished(stable_ids, no_compile_layers)''',
    "A4/4 get_finished frees under stable id",
)

# -------------------------------------------------------- Part B: chunked-prefill continuation

sub(
    '''                assert req_id in self.chunked_prefill
                assert new_block_ids is not None
                block_ids = new_block_ids[0]
                if not resumed_from_preemption:
                    block_ids = self.chunked_prefill[req_id][0] + block_ids''',
    '''                # A cached producer req we never stashed = a post-prefill decode step (e.g. the
                # 1-token decode before finish): nothing to transfer-accumulate → skip.
                if req_id not in self.chunked_prefill:
                    continue
                prev = self.chunked_prefill[req_id][0]
                if new_block_ids is None:
                    # No new blocks allocated this chunk (fits in already-allocated / sliding-
                    # window blocks) — carry the accumulated ids forward unchanged.
                    block_ids = prev
                elif resumed_from_preemption:
                    block_ids = new_block_ids[0]
                else:
                    block_ids = prev + new_block_ids[0]''',
    "B1/2 producer continuation tolerates new_block_ids=None",
)

sub(
    '''                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                assert new_block_ids is not None
                block_ids = new_block_ids[0]''',
    '''                # NOTE(rob): For resumed req, new_block_ids is all
                # of the block_ids for the request.
                if new_block_ids is None:
                    continue  # nothing new to load this step
                block_ids = new_block_ids[0]''',
    "B2/2 consumer resume tolerates new_block_ids=None",
)

if failed:
    print("\nERROR: anchors did not match — vLLM has changed. NOT writing.", file=sys.stderr)
    for f_ in failed:
        print(f"  - {f_}", file=sys.stderr)
    raise SystemExit(1)

with open(PATH, "w") as f:
    f.write(content)

print("\nSuccessfully applied the p2p stable-id + chunked-continuation fix!")
PYTHON_EOF

python3 -m py_compile "$CONNECTOR_PATH"
echo "py_compile OK"
