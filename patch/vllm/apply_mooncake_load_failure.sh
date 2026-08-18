#!/bin/bash
# LOCAL fix (not a wiki patch): stop a FAILED Mooncake KV pull from wedging the decode engine.
#
# Symptom (observed 2026-08-13 on this box, BASELINE=vanilla, 1P1D, concurrency 256):
#     ERROR mooncake_connector.py:1112] pulling kv_caches for ['chatcmpl-...'] failed:
#         Mooncake transfer engine returned -1
#   ...118 times, and then the decode engine settles permanently at
#     Running: 0 reqs, Waiting: 141 reqs, GPU KV cache usage: 30.1%
#   and the benchmark never finishes.
#
# Cause: `MooncakeConnectorWorker.process_pulling_result` credits `response.ok_reqs` toward
#   `finished_recving_reqs` but merely LOGS `response.err_reqs`. A request whose pull failed is
#   therefore never reported as done-recving, so the scheduler leaves it in
#   `WAITING_FOR_REMOTE_KVS` forever — holding its allocated (never-filled) D-side blocks and a
#   concurrency slot. Once enough accumulate the engine has nothing runnable left and simply stops.
#   A single transient transfer error is thus unrecoverable, and it looks like a hang rather than
#   an error. (This is what makes the underlying transport failure fatal instead of merely slow;
#   the transport failure itself — e.g. TCP ephemeral-port exhaustion under load — is separate.)
#
# Fix: route a failed pull into vLLM's EXISTING KV-load-failure path, which already knows how to
#   recover. The connector reports the blocks that were never written via
#   `get_block_ids_with_load_errors()`; the scheduler's `_handle_invalid_blocks` then either
#   recomputes the request locally (`kv_load_failure_policy=recompute`, the default) or fails it
#   outright (`=fail`). Both beat hanging, and neither runs a request on KV that never arrived.
#   The request is also marked done-recving so it actually leaves WAITING_FOR_REMOTE_KVS.
#
# `MooncakeConnectorFF` (kv_fast_fusion/connectors/mooncake_connector_ff.py) implements the same
# recovery natively — this patch is what gives the STOCK connector, i.e. the BASELINE=vanilla
# reference, the same property, so the two are comparable.
#
# Idempotent (self-detects prior patch). Override the target with VLLM_DIR; by default this probes
# the local .venv site-packages first, then the container layout /vllm-workspace/vllm.

set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REL="vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py"

if [ -n "$VLLM_DIR" ]; then
    CONNECTOR_PATH="${VLLM_DIR}/${REL}"
else
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
    echo "Error: mooncake_connector.py not found (set VLLM_DIR to the vLLM root)."
    exit 1
fi

echo "Target: $CONNECTOR_PATH"

if [ ! -f "${CONNECTOR_PATH}.bak_load_failure" ]; then
    cp "$CONNECTOR_PATH" "${CONNECTOR_PATH}.bak_load_failure"
    echo "Created backup at ${CONNECTOR_PATH}.bak_load_failure"
fi

CONNECTOR_PATH="$CONNECTOR_PATH" python3 << 'PYTHON_EOF'
import os

PATH = os.environ["CONNECTOR_PATH"]

with open(PATH) as f:
    content = f.read()

if "_failed_load_block_ids" in content:
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


# ---- 1/4: worker state holding the blocks of failed pulls ----------------------------------
sub(
    '''        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()''',
    '''        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()
        # Blocks belonging to pulls that FAILED. Drained by get_block_ids_with_load_errors() so the
        # scheduler can recompute (or fail) the owning requests instead of waiting forever on KV
        # that will never arrive. See patch/vllm/apply_mooncake_load_failure.sh.
        self._failed_load_block_ids: set[int] = set()''',
    "1/4 worker state for failed-load blocks",
)

# ---- 2/4: turn a failed pull into a reported KV-load failure --------------------------------
sub(
    '''        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )''',
    '''        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )
            # Do NOT just log: a failed pull that is never reported as done-recving strands its
            # request in WAITING_FOR_REMOTE_KVS forever, holding its blocks and a concurrency slot,
            # until the engine has nothing runnable left. Report the never-written blocks as load
            # errors and release the request, so Scheduler._handle_invalid_blocks recovers it.
            for req_id in response.err_reqs:
                pull_meta = pull_metas.get(req_id)
                if pull_meta is None:
                    continue
                # local_block_ids is a flat list[int] for this connector, but the BFF subclass
                # (MooncakeConnectorFF) carries PER-GROUP list[list[int]] and calls up into here,
                # where set.update() on nested lists raises "unhashable type: 'list'". Flatten
                # defensively so this is correct for either shape. Every step below is idempotent,
                # so it is also harmless when the subclass has already handled these err_reqs.
                for group in pull_meta.local_block_ids:
                    if isinstance(group, (list, tuple)):
                        self._failed_load_block_ids.update(int(b) for b in group)
                    else:
                        self._failed_load_block_ids.add(int(group))
                pull_meta.pull_tasks_count = 0
                self.finished_recving_reqs.add(pull_meta.d_req_id)''',
    "2/4 failed pull -> load error + release",
)

# ---- 3/4: worker accessor -------------------------------------------------------------------
sub(
    '''    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):''',
    '''    def take_block_ids_with_load_errors(self) -> set[ReqId]:
        """Blocks of failed pulls since the last call (drained)."""
        out = self._failed_load_block_ids
        self._failed_load_block_ids = set()
        return out

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):''',
    "3/4 worker accessor",
)

# ---- 4/4: connector-level hook vLLM actually calls -------------------------------------------
sub(
    '''    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()''',
    '''    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Blocks whose remote pull failed. vLLM turns these into `invalid_block_ids` ->
        `Scheduler._handle_invalid_blocks`, which recomputes or fails the owning requests.
        Without this a failed transfer is a silent, permanent stall."""
        if self.connector_worker is None:
            return set()
        return self.connector_worker.take_block_ids_with_load_errors()''',
    "4/4 connector get_block_ids_with_load_errors",
)

if failed:
    print("\\nFAILED to apply (vLLM source has drifted):")
    for f in failed:
        print(f"  - {f}")
    print("No changes written.")
    raise SystemExit(1)

with open(PATH, "w") as f:
    f.write(content)
print("\\nPatch applied successfully.")
PYTHON_EOF

python3 -c "import ast,sys; ast.parse(open('$CONNECTOR_PATH').read())" \
    && echo "Syntax check passed."
echo "Revert with: cp ${CONNECTOR_PATH}.bak_load_failure $CONNECTOR_PATH"
