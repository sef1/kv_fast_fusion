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
