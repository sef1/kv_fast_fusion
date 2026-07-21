# Wiki patch set rebased to v0.19.1

The Huawei wiki patches were authored for **v0.18.rc1**; these are rebased to the deployed
**v0.19.1** stack (vllm-ascend `v0.19.1rc1`, vLLM `0.19.1`, Mooncake `v0.3.11`). Apply on the
container. Paths are overridable via `VLLM_ASCEND_DIR` / `VLLM_DIR` / `MOONCAKE_DIR`
(defaults `/vllm-workspace/{vllm-ascend,vllm}` and the Mooncake source tree).

## What changed vs. the original wiki set

| Wiki patch | Here | Status on v0.19.1 |
|---|---|---|
| 0003 P/D Mooncake MTP accuracy | `vllm-ascend/0001-*MTP-accuracy*.patch` | rebased (1-line slice `remote_block_ids[:num_prompt_blocks]`) |
| 0004 P/D transmit kv cache failure | `vllm-ascend/0002-*transmit-kv-cache-failure*.patch` | rebased onto refactored `_handle_request` |
| 0001 mtp+kv pool | `vllm-ascend/0003-*ascend_store-deferred-finish*.patch` | **only the ascend_store part** applies; the model_runner parts are already covered by v0.19.1's `finalize_kv_connector` / `maybe_get_kv_connector_output(defer_finalize)` |
| 0002 mtp+kv pool 2 | — | **NOT NEEDED** on v0.19.1 (subsumed by the context-manager finalize above) |
| 0009 SSD offload | `vllm-ascend/0004-Support-ssd-offload.patch` | rebased (optional; needs Mooncake v0.3.11+ setup kwargs) |
| 0010 SSD dp-rank | `vllm-ascend/0005-*dp-rank*.patch` | rebased (optional) |
| 0008 Mooncake SSD metrics | `mooncake/0001-*SSD-metrics*.patch` | applies clean (optional; **needs a C++ rebuild** of Mooncake) |
| apply_fix.sh (loggers) | `vllm/apply_fix.sh` | matches v0.19.1 verbatim; run as-is |
| 0005 GLM chat template | `vllm/apply_glm_and_chat.sh` | rebased (string-replace) |
| 0006 + 0011 GLM reasoning parser | `vllm/apply_glm_and_chat.sh` + `vllm/glm_reasoning_parser.py` | 0006 and 0011 **folded** into one bundled parser file |

## Not from the wiki: `vllm/apply_p2p_stable_id.sh` (local fix, **required** for stock-connector P/D)

Applied by `apply_all.sh` alongside the loggers guard. Two independent fixes to
`vllm/distributed/kv_transfer/kv_connector/v1/p2p/p2p_nccl_connector.py`, both needed before the stock
`P2pNcclConnector` can serve as a P/D baseline at any real concurrency:

- **Stable cross-P/D tensor id.** `InputProcessor.assign_request_id` appends a *per-server* random
  8-hex suffix (`f"{external_req_id}-{random_uuid():.8}"`). The proxy gives P and D the same
  `external_req_id`, but each server generates its own suffix, so a tensor_id keyed on the full id
  never matches. `P2pNcclEngine.recv_tensor` waits on an unbounded condition variable, so the
  consumer's **first forward step blocks forever** — the decode logs zero engine stats while producer
  KV piles up undrained (`Out Of Threshold`) and clients hang to their timeout. Keys send/recv **and
  `get_finished`** on the stripped id; the `get_finished` hunk is not cosmetic — under the full id the
  engine pops `recv_store` keys that were never stored, leaking spilled KV until OOM.
  (Carried since v0.14.0. `P2pNcclConnectorFF._pd_key` is the same fix, which is why BFF was immune.)
- **Chunked-prefill continuation.** Stock asserts `new_block_ids is not None` in its continuation
  branches; a continuation step that allocates no new block passes `None` → `AssertionError` → EngineCore
  dies. Mirrors `P2pNcclConnectorFF.build_connector_meta`. Without this, vanilla must run
  `--no-enable-chunked-prefill` and is no longer config-comparable to a BFF run.

`VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` also makes the ids match with no code change, but vLLM marks
it deprecated and it disables the uniqueness guard process-wide — useful as a quick check, not as the fix.

Relevance to the current setup (Qwen2.5-7B, no MTP, BFF runs the FF mover standalone): the two that
fix the P/D transfer slowness are **vllm-ascend/0001 (MTP-accuracy) and 0002 (transmit-kv-cache-failure)**.
The ascend_store (0003), SSD (0004/0005 + mooncake), and GLM (vllm) patches are situational.

## vLLM patches are scripts, not `git am`

The vLLM changes ship as idempotent scripts (`apply_fix.sh`, `apply_glm_and_chat.sh`) + a bundled
`glm_reasoning_parser.py`, because they were rebased against the installed vLLM (not a git tree). They
string-match v0.19.1 anchors and self-detect prior application.

## Apply

```bash
# core P/D transfer fixes (recommended)
VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend VLLM_DIR=/vllm-workspace/vllm \
  ./apply_all.sh

# include the optional SSD offload (Mooncake rebuild required) + GLM serving fixes
ENABLE_SSD=1 ENABLE_GLM=1 MOONCAKE_DIR=/path/to/Mooncake \
VLLM_ASCEND_DIR=/vllm-workspace/vllm-ascend VLLM_DIR=/vllm-workspace/vllm \
  ./apply_all.sh
```

If a `git am` hunk rejects because the container tree differs from this rebase target, retry that
repo with `git am -3` (3-way merge). After applying, restart the P/D servers.
