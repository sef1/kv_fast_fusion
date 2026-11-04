# BFF (KV fast fusion): project status, 2026-11-04

**Goal.** Free decode KV pages by deduplicating near-identical KV blocks across requests, and turn
the freed pages into higher throughput without damaging output quality.

**Hardware.** One Ascend box with 16 NPUs, so one prefill and one decode node at TP8 each. vLLM
0.18.0 and vllm-ascend 0.18.0 on the box. GLM-5.1 W4A8, 79 layers including the MTP layer, MLA with DSA.

Removal work is in [PLAN.md](PLAN.md). This file says what exists and what is true today.

## 1. Where things stand

- **Correctness is solved for the current path.** BFF runs at baseline quality with 0 garbage
  outputs, with dedup, the clip and free-at-promotion all on.
- **In the same scenario, BFF already wins on prefill work and network.** About 30% less re-prefill
  work, about 2.8% less KV pulled, about 214 decode blocks freed per run.
- **BFF has not beaten mooncake on throughput.** On the 121-prompt set it is level, within noise. It
  frees about 2.3% of a prompt's KV; the queue simulator says a visible win (+5%) needs about 10%.
- **No public dataset scanned so far reaches that.** Magicoder's long prompts are still the best
  found. LongBench and SWE-smith could not be loaded and remain unscanned.
- **The repo carries far more than the path in use.** Your runs load about 13,000 of 31,428
  production lines, and not all of those execute. See section 4 and PLAN.md.

**Pending.** One run: baseline at decode limit 8. It tests whether the two systems have different
best limits.

## 2. Results

GLM-5.1 W4A8, 1P1D, TP8, concurrency 48. Every run in a table uses the same prompts.

**121 prompts of 4,096-6,000 tokens, 1,024 output tokens** (fingerprint `9f5e27b8`)

| Run | Build | Garbage | F1 | CodeBLEU | AST | tok/s | req/s | wall s | Prefill tokens | Freed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mooncake baseline | - | 0 | 0.656 | 46.45 | 41.3 | 197.1 | 0.36 | 335 | 659k | 0 |
| BFF connector, dedup off | 03a- | 0 | 0.657 | 45.80 | 40.5 | 196.5 | 0.36 | 334 | 662k | 0 |
| dedup, nothing freed | 03a | 0 | 0.666 | 46.44 | 43.0 | 200.7 | 0.38 | 323 | 636k | 0 |
| free, alias off | 03b | 0 | 0.649 | 45.69 | 43.8 | 188.5 | 0.34 | 355 | 646k | 104 |
| free, clip 0.25 | 03d | 0 | 0.644 | 44.61 | 42.2 | 199.7 | 0.36 | 340 | 645k | 215 |
| free, alias on (bug) | 03d | 14 | 0.566 | 38.54 | 34.7 | 174.5 | 0.28 | 432 | - | - |
| free, clip 0.4, alias fixed | 03e | 0 | 0.655 | 45.88 | 43.0 | 197.7 | 0.36 | 333 | 622k | 230 |
| free, clip 0.5 | 03f | 0 | 0.654 | 46.53 | 39.7 | 192.8 | 0.35 | 342 | 633k | 229 |
| free, clip 0.4 | 03g | 0 | 0.647 | 45.98 | 41.3 | 196.6 | 0.36 | 340 | 651k | - |
| free, clip 0.4 | 03h | 0 | 0.649 | 45.02 | - | 199.9 | 0.36 | 335 | 640k | - |
| free, clip 0.4 | 03i | 0 | 0.648 | 45.54 | 41.3 | 195.6 | 0.36 | 339 | 653k | 275 |
| **free, clip 0.4 (best so far)** | **03j** | **0** | **0.656** | **46.62** | **39.7** | **198.3** | **0.37** | **329** | **636k** | **214** |

Clean clip 0.4 runs average 197.6 tok/s, against 197.1 for the baseline. Identical configurations
differ by about 2.7% in wall time and about 1 point of CodeBLEU.

**200 prompts of 1,000-2,000 tokens, 2,000 output tokens** (fingerprint `4da4b383`)

| Run | Limit | tok/s | Evictions | F1 | Decode running | Decode KV | Blocks freed |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | none | 326.2 | 243 | 0.574 | 18.2 | 94% | 0 |
| BFF 03i | none | 327.6 | 287 | 0.565 | 18.1 | 94% | 14 |
| baseline | 14, client too | 296.7 | 0 | 0.598 | 13.2 | 66% | 0 |
| BFF 03j | 14, client too | 293.4 | 0 | 0.594 | 13.3 | 67% | 0 |

The cap runs also limited the client, so nothing queued at the decode. This prompt set has almost no
similar KV, so BFF has nothing to free there. Evictions are counted differently for the two
connectors, so compare them with care.

## 3. What limits throughput

| Finding | Evidence |
|---|---|
| **121-prompt set is prefill-bound.** | Prefill ~2,100 tok/s ÷ ~5,500 tokens ≈ 0.38 req/s, the measured rate. About 29 waiting at prefill. TTFT ~78 s of ~106 s. |
| **The decode also hits capacity.** | Waiting above 0 in 48-70% of samples, at KV ~93-95%. |
| **The 200-prompt set is decode-bound.** | ~19 waiting at the decode, prefill KV pinned at 93%, prefill at under half its capacity. |
| **Little KV is similar.** | Tokens: 27.6% of blocks are near-duplicates at the same position. KV: 8.5% meet the bar, ~10.8% at a very loose bar, ~88% have nothing close (best cosine under 0.5). Earlier context dominates middle-layer KV. |
| **Freed share is small.** | ~1.8 of ~77 pages per request (~2.3%). The layout costs 1.3% (48,000 against 48,640 tokens). |
| **Over-admission causes evictions.** | The decode admits on prompt size only, then evicts when outputs grow. Each eviction re-prefills prompt plus partial answer. |
| **BFF adds almost no time.** | Decode: ~35 µs in a ~63 ms step. Planning ~24 ms and landing copy ~10 ms per request, off the forward path. Signing ~4.5 ms per request (~0.2%). |

The "44 ms per request, 2% of prefill" signing figure reported on 2026-11-03 was wrong. It averaged
in a one-time 3.8 s rebuild of the projection matrix, fixed in 04a.

**Arguments do not change the picture.** A simulator validated on the 121-prompt baseline (344 s
against 335 s, 15 evictions against 17) shows BFF within -1.5% to +2.4% of mooncake across client
concurrency, output length, decode limit, decode memory and prefill speed. It assumes a fixed freed
share and does not model eviction stalls.

**Datasets scanned for BFF-usable redundancy** (same-position near-duplicates, tokens):
Magicoder ≥5,000 33.3% (only 50 prompts), Magicoder ≥4,096 27.6%, Magicoder 1,000-6,000 10.0%,
SWE-bench oracle 0.6-4.5% (up to 27% at shifted positions, which BFF cannot use), TriviaQA 0.4%,
Hermes function calling 0.2%, RAG-12000 0%.

## 4. The path your runs actually use

`BASELINE=bff_pull_v2` selects `MooncakeConnectorFFv2`. In order:

1. **Activate.** `fast_fusion_main` imports `kv_fast_fusion`, which calls
   `apply_fast_fusion_ascend_patch`. It patches the block pool, KV init and scheduler, and registers
   four connectors, of which you use one.
2. **Layout.** `fast_fusion_core._initialize_kv_caches` and `bff_group_layout` split 79 layers into
   g0 (first and last 20, 40 layers, never deduplicated) and g1 (the middle 39).
3. **Sign on the prefill node.** `_ff_piggyback_sign` → `precompute_and_offer` →
   `signature_rows_for` → `project_signatures`. One 128-dimension signature per g1 block, sent with
   the request's transfer parameters.
4. **Plan on the decode.** `_prefetch_signatures` and `_plan_aligned` → `DedupEngine.plan` →
   `DedupPlanner.plan` → `LshIndex.probe`. Blocks that match are declined, so they are not pulled.
5. **Cap and clip.** `cap_request_decline`, then `clip_request_decline` keeps the lowest-error aliases
   of over-limit requests.
6. **Pull.** `_transfer_kv_cache_timed` reads the remaining blocks over RDMA.
7. **Copy at landing.** `_after_transfer` → `DedupEngine.release_materialized` → `_make_landing_copy`
   copies each representative, with all its tensors, into the request's own block.
8. **Swap at promotion.** `_bff_landed_promotion_apply` → `_landed_swap` → `consensus_landed`,
   `promotion_swaps`, then `_handle_block_merging_with_counts`. The block table points at the
   representative, and the victim goes back to the pool.
9. **Stop tracking.** `patched_free_blocks` tells the dedup index which blocks left the pool, shipped
   to the workers by `FreedBlockLog`.

**Fallbacks that never fired in the best run:** on-demand signature exchange (132 piggyback hits, 0
misses, 0 exchanges); re-read of a refused alias (0); local recompute of an alias (0); hot-block
refusal in the forward path (0).

## 5. Every file

Status: **LIVE** in your runs, **DIAG** diagnostic, **TOOL** run by hand, **GPU** the CUDA stack,
**LEGACY** and **OLD** superseded. Line counts are current. Plan phases are in PLAN.md.

**Nothing leaves `v0.18.0`.** Decided 2026-11-04: GPU, layerwise, hybrid and the alias all stay there.
A phase label in the Plan column means removal on the separate `bff-core` branch only, except that
hybrid and the alias stay on both. Diagnostics move to one module.

| File | Lines | Status | Why it exists | Plan |
|---|---:|---|---|---|
| `kv_fast_fusion/__init__.py` | 35 | LIVE | Package import is the activation: picks the Ascend or the GPU patch. | keep; drop the GPU branch in P3 |
| `kv_fast_fusion/fast_fusion_main.py` | 12 | LIVE | Launcher: importing the package applies the patch, then runs vLLM's CLI. | keep |
| `kv_fast_fusion/constants.py` | 26 | LIVE | THRESHOLD and the group-layout flags. | keep; gains per-group thresholds in P0 |
| `kv_fast_fusion/fast_fusion_core.py` | 296 | LIVE | Rebuilds the KV layout into groups. `bff_group_layout`, `_initialize_kv_caches`. | keep; drop unused layouts in P5 |
| `kv_fast_fusion/hybrid_groups.py` | 74 | LIVE (hybrid only) | Splits Mamba/attention groups. Inert on GLM (`hybrid=False`). | decision D2 |
| `kv_fast_fusion/fast_fusion_block_pool.py` | 511 | LIVE (part) | Patches vLLM's free/evict/cache paths; free flush; dedup-index free hook; alias code and trace. | keep; alias parts go in P5 |
| `kv_fast_fusion/fast_fusion_scheduler.py` | 956 | LIVE (2 of 10 functions) | Block merge at swap, and the HMA invalid-block fix. The rest is GPU or dead. | keep 2 functions; P1, P3 |
| `kv_fast_fusion/pd_dedup_v2.py` | 1,680 | LIVE (core) | Signatures, `DedupEngine` (plan, stage, land), stats, `AliasApplier`. | keep; drop ratio and redirect in P5 |
| `kv_fast_fusion/pd_dedup_plan.py` | 238 | LIVE (core) | The match decision for each incoming block. | keep |
| `kv_fast_fusion/pd_lsh.py` | 515 | LIVE (core) | The block index; cosine and rel_err bars; match-potential grid. | keep |
| `kv_fast_fusion/pd_fuse.py` | 225 | GPU / v1 | Producer clustering helpers for v1 fusion. | P3 / P4 |
| `kv_fast_fusion/compression_hook.py` | 19 | LEGACY | Only legacy runners use it. | P2 |
| `kv_fast_fusion/fast_fusion_flash.py` | 256 | GPU (ratio) | Patched GPU attention forward for ratio mode. | P3 |
| `kv_fast_fusion/fast_fusion_triton_attn.py` | 424 | GPU (ratio) | GPU Triton kernel applying per-layer scales. | P3 |
| `kv_fast_fusion/fast_fusion_pd_patch.py` | 413 | GPU | The GPU counterpart of the Ascend patch. The Ascend path uses one helper from it, on a failure path. | P3; move the helper in P0 |
| `kv_fast_fusion/plot_bff_metrics.py` | 152 | OLD | Terminal plots from an old log format; nothing refers to it. | P1 |
| `kv_fast_fusion/connectors/__init__.py` | 13 | GPU | Imports the NCCL connector, which is what breaks the 13 local tests. | P3 |
| `kv_fast_fusion/connectors/mooncake_connector_ff.py` | 1,889 | GPU + one LIVE value | GPU v1 Mooncake connector. The live path reads `_THRESHOLD_G` from it. | P3; move the value in P0 |
| `kv_fast_fusion/connectors/mooncake_connector_ff_v2.py` | 390 | GPU | GPU v2 transport. | P3 |
| `kv_fast_fusion/connectors/mooncake_connector_ff_v2_legacy.py` | 805 | LEGACY | Frozen measurement baseline, marked DO NOT EDIT. | P3 |
| `kv_fast_fusion/connectors/p2p_nccl_connector_ff.py` | 1,255 | GPU | The original P/D connector, over NCCL. | P3 |
| `kv_fast_fusion/tools/collect_bff_stats.py` | 763 | GPU tool | Stats summary. Only the GPU script calls it; the Ascend script has its own 577-line inline copy. | decision in P6: keep this one and drop the inline copy |
| `kv_fast_fusion/tools/bff_f1_paired.py` | 149 | TOOL | Paired per-sample quality comparison of two runs. Useful; not wired into any script. | keep |
| `kv_fast_fusion/tools/bff_cksum_diff.py` | 64 | GPU tool | Joins NCCL send/recv checksums. | P3 |
| `kv_fast_fusion/tools/nccl_pd_repro.py` | 77 | GPU tool | NCCL repro. | P3 |
| `kv_fast_fusion/legacy/ (8 files)` | 6,566 | LEGACY | The first single-node BFF runners. One function is still used by the GPU ratio path. | P2, after P3 or by moving `_fill_norm_buffers` |
| `kv_fast_fusion_ascend/__init__.py` | 12 | LIVE | Exports the patch. | keep |
| `kv_fast_fusion_ascend/build.py` | 8 | LIVE | Build tag printed in the decode log. | keep |
| `kv_fast_fusion_ascend/fast_fusion_ascend_patch.py` | 933 | LIVE (part) | Installs all patches on the NPU stack; swap at promotion. Also holds v1 redirect code that never fires. | keep; drop v1 in P4/P5 |
| `kv_fast_fusion_ascend/landed_promotion.py` | 123 | LIVE (core) | Rank consensus and the safety proof for swapping at promotion. | keep |
| `kv_fast_fusion_ascend/hybrid_layout.py` | 493 | LIVE (hybrid mostly) | KV geometry. Required by the connector; only trivial on GLM. | decision D2 |
| `kv_fast_fusion_ascend/shared_write_check.py` | 230 | DIAG (on by default) | Detects writes into shared blocks and stale-hash blocks. Found the alias bug. | keep until P5, then default off |
| `kv_fast_fusion_ascend/step_profile.py` | 545 | DIAG (off by default) | Per-step timing and memory probes. | keep; move memory probes out in P5 |
| `kv_fast_fusion_ascend/connectors/merge_stats.py` | 76 | LIVE | Carries landed aliases and signatures between worker and scheduler at TP>1. | keep |
| `kv_fast_fusion_ascend/connectors/mooncake_connector_ff.py` | 2,170 | LIVE (base class) | Group-aware pull transport; v2 subclasses it. Also holds v1 fusion that v2 switches off. | keep; drop fusion in P4 |
| `kv_fast_fusion_ascend/connectors/mooncake_connector_ff_v2.py` | 4,124 | LIVE (core) | The connector your runs use: signing, planning, landing copy, diagnostics, unused async paths. | keep; shrink in P5 |
| `kv_fast_fusion_ascend/connectors/mooncake_layerwise_connector_ff.py` | 2,081 | NOT USED + 4 helpers | Layerwise v1 (`bff`). The live path imports 4 symbols from it. | P4 |
| `kv_fast_fusion_ascend/connectors/mooncake_layerwise_connector_ff_v2.py` | 652 | NOT USED + 1 helper | Layerwise v2 (`bff_v2`). The live path imports `signatures_for_group`. | P4 |
| `f1_benchmark/f1_main.py` | 594 | LIVE | The benchmark client and scoring. Your box has an uncommitted input-length cap. | keep; commit the cap |
| `f1_benchmark/degenerate_outputs.py` | 189 | LIVE | Garbage count and attribution. | keep |
| `f1_benchmark/alias_attribution.py` | 165 | LIVE | Per-request quality against the decline ledger. | keep |
| `f1_benchmark/compare_runs.py` | 198 | LIVE | Sweep table and verdict. | keep |
| `f1_benchmark/scan_block_redundancy.py` | 178 | TOOL | CPU estimate of token-level redundancy. | keep |
| `f1_benchmark/rescore.py` | 113 | TOOL | Offline rescoring against other references. | keep |
| `f1_benchmark/make_overlap_dataset.py` | 185 | TOOL (unused) | Synthetic overlap prompts. You rejected this direction. | P1, optional |
| `f1_benchmark/f1_main_o.py` | 412 | OLD | Old copy of f1_main. Nothing refers to it. | P1 |
| `f1_benchmark/f1_patch_.py` | 53 | OLD | Dataset monkey patch. Nothing refers to it. | P1 |
| `f1_benchmark/f1_dataset.py` | 91 | OLD | Only f1_patch_ uses it. | P1 |
Other tracked paths: 38 test files (14,072 lines),
`patch/` (fixes applied to vLLM, vllm-ascend and Mooncake on the box), `examples/.../ascend_xpyd`
(`run_benchmarks.sh`, `run_best_compare.sh`, `run_throughput.sh`), `examples/.../mooncake_gpu_xpyd`
and `p2p_nccl_xpyd` (GPU scripts), 56 result files under `p2p_nccl_xpyd/f1_results`, and `msg.sh` and
`package*.json` (leftovers).

Untracked in the working copy: `vllm/` (a 2.5 GB checkout from 2025, not what the box runs),
`vllm-ascend/`, `Mooncake/`, `nsys_report.*` and `report1.*` (729 MB of profiler output),
`logs_for_Claude/`, `GLM logs for claude/`, `setup_cuda_norm_scatter.py`,
`kv_fast_fusion/norm_scatter_kernel.*`, `tools/run_nsys_serve.py`, `setting_claude_deepseek.json`.

## 6. Functions in the live files

**`kv_fast_fusion_ascend/connectors/mooncake_connector_ff_v2.py`** (4,124 lines)

| Mark | Functions |
|---|---|
| LIVE | `signature_request_and_plan_groups`, `decline_fraction`, `cap_request_decline`, `clip_request_decline`, `filter_sentinels`, `declined_pairs`, `select_rank_sigs`, `piggyback_sign_here`, `claim_piggyback_sign`, `assemble_sig_payload`, `sig_payload_shape`, `trim_sig_payload_to`, `fit_piggyback_to_ask`, `claim_prefetched`, `warm_group`, `cache_list_device`, `cache_parts`, `_make_copy_alias_blocks`, `_make_landing_copy`; recv thread `run`, `_prefetch_signatures`, `_take_prefetched`, `_plan_aligned`, `_align_and_group`, `_after_transfer`, `_materialize_on_landing`; worker `start_load_kv`, `register_kv_caches`, `signature_rows_for`, `precompute_and_offer`, `note_landed`, `take_landed`; connector `get_kv_connector_stats`, `wait_for_save`, `_ff_piggyback_sign`, `start_load_kv`, `requires_piecewise_for_cudagraph` |
| FALLBACK, never fired in the best run | `_SigServer`, `_SigClient` (`ask_many`, `_exchange`), `signature_batch_plan`, `sig_request_batch_msg`, `sig_reply_batch_msg`, `parse_sig_reply_batch`, `sig_reply_msg`, `parse_sig_reply`, `group_by_peer`, `_ask_for`, `_peer_of`, `ask_shape`, `drain_queue`, `signatures_for_groups_split`, `signatures_for_group_split`, `repull_segments`, `_repull_refused`, `note_failed_blocks`, `take_failed_blocks` |
| OPTIONAL | `dedup_gate_open`, `_dedup_gate_closed` (KV-usage gate, off) |
| DEAD END, off and measured no gain | `partition_by_age`, `SignaturePipeline`, `PendingAsk`, `BatchCoalescer`, `_SigClient.ask`, `_sig_pipeline`, `_exchange_for`, `_plan_for`, `_run_async`; signature cache `split_cached_blocks`, `precompute_signatures`, `drain_precompute`, `serve_signatures_cached` |
| DIAG, transfer checks, off | `VerifySampler`, `classify_mismatch`, `best_matching_row`, `cross_match`, `compare_signature_rows`, `_verify_transfer`, `_diagnose_mismatch`, `_localise`, `_replay`, `_stability`, `_recompute_local`, `_audit_self_consistency`, `_score_audit`, `_audit_buckets`, `_bucket_for`, `hot_block_collisions`, `write_slot`, `slot_trace_fault`, `_trace_slots`, `_check_mirror_matches_device`, `_audit_hot_blocks` |
| DIAG, signing question now closed | `_sig_phase`, `_sig_cold_first`, `_sig_cold_done`, `_sig_alt_check`, `_alloc_stats`, the timing half of `precompute_and_offer` |
| REDIRECT, unused under materialize | `_applier`, `_v2_apply` (apply half), `_write_block_table`, `_warn_on_runaway_groups` (keep this one) |
| UNREFERENCED | `sig_request_msg` (tests only) |

**`kv_fast_fusion_ascend/connectors/mooncake_connector_ff.py`** (2,170 lines)

| Mark | Functions |
|---|---|
| LIVE | `build_layer_group_map`, `group_of`, `build_base_addr_groups`, `block_segment`, `chunk_segments`, `align_per_group`, `flatten_group_lists`, `transfer_indices`, `planned_blocks`, `ff_sig_encode`, `ff_sig_encode_any`, `ff_sig_decode`, `FFSigStash`, `resolve_kv_cache_groups`, `_kv_cache_usage`, `FreedBlockLog`, `MooncakeConnectorMetadataFF`, `MooncakeConnectorSchedulerFF` (`update_state_after_alloc`, `build_connector_meta`, `_collect_fuse_reqs`, `request_finished_all_groups`), `KVCacheRecvingThreadFF` (`_one_batch`, `_issue_transfer`, `_transfer_kv_cache_timed`, `_reformat_after_pull`), `MooncakeConnectorWorkerFF`, `MooncakeConnectorFF` (`update_connector_output`, `wait_for_save`) |
| HYBRID only | `_narrow_mamba_groups`, `_block_len_at`, `_hybrid_regions`, `dedup_registration_regions` |
| NO EFFECT | `admission_deferred`, `note_loads_finished`, `get_num_new_matched_tokens` override (off; it could host an admission policy later) |
| V1 FUSION, switched off by v2 | `FFRowStash`, `FFPendingSource`, `normalize_ff_redirects`, `_ff_pending_source`, `save_kv_layer`, `_ff_accumulate`, the `MooncakeFFProducer` import |
| DIAG | `RecvThreadTimer`, `RegionMap`, `AuditBuckets`, `descriptor_coverage`, `_audit_descriptor_coverage`, `_note_recv_timing`, `_sync_producer_writes` (save barrier, off) |

**`kv_fast_fusion_ascend/fast_fusion_ascend_patch.py`** (933 lines)

| Mark | Functions |
|---|---|
| LIVE | `apply_fast_fusion_ascend_patch`, `_patch_npu_model_runner`, `_ext_of`, `_bff_ingest_landed`, `_bff_landed_promotion_apply`, `_landed_swap`, `_note_swaps`, `_bff_victim_reuse`, `_wrap_scheduler_promotion`, `_wrap_scheduler_update_from_output`, `_bff_ingest_piggyback_sigs`, `_patch_attn_metadata_forward_order`, `forward_ordered`, `_registration_failed` |
| V1 / REDIRECT, no input in your runs | `_bff_promotion_apply`, `_bff_apply_block_merge`, `_bff_sweep_late_maps`, `_rep_gone_bucket`, `_rep_history_record`, `_note_promoted` |

**`kv_fast_fusion/pd_dedup_v2.py`** (1,680 lines)

| Mark | Functions |
|---|---|
| LIVE | `threshold_for`, `key_blocks`, `_row_index`, `_check_layout`, `signature_matrix_by_layer`, `project_signatures`, `SignatureCodec`, `DedupStats`, `DedupEngine` (`plan`, `_plan`, `_record`, `release_materialized`, `_release_locked`, `forget`, `retain`, `alias_errors`, `on_blocks_freed`, `rep_status`) |
| CONCAT, used by the fallback, warm-up and layerwise | `signature_matrix` |
| RATIO, GPU only | `value_blocks`, `block_layer_norms`, `SignatureCodec.kv_norms`, `DedupEngine._scale_for`, `_block_kvn` |
| REDIRECT, unused under materialize | `AliasApplier` (317 lines), `DedupEngine.drain_ready` |
| DIAG | `sharing_stats`, `_bucket_hist`, `_bump`, `_tally` |

**`kv_fast_fusion/fast_fusion_scheduler.py`** (956 lines)

| Mark | Functions |
|---|---|
| LIVE | `_handle_block_merging_with_counts`, `_update_requests_with_invalid_blocks` |
| GPU | `update_from_output` (342 lines), `_bff_record_decode_free`, `_bff_dump_decode_stats` |
| DEAD copies, nothing references them | `_handle_block_merging_with_counts_o`, `_handle_block_merging_with_counts____`, `_handle_block_merging_with_counts_` |
| Only the dead copies use them | `_update_hashes_for_merged_blocks`, `_create_block_id_hash` (referenced only inside this file) |

**`kv_fast_fusion/fast_fusion_block_pool.py`** (511 lines)

| Mark | Functions |
|---|---|
| LIVE | `patched_free_blocks`, `on_blocks_freed`, `kv_cache_usage`, `block_occupancy`, `flush_before_load`, `_flush_on_free`, `_flush_npu`, `_flush_log` |
| ALIAS, no benefit in v2 | `add_block_alias`, `patched_maybe_evict_cached_block`, `_trace_hits`, `patched_get_computed_blocks` |
| TOLERANCE, check whether it is still needed once the alias is gone | `patched_cache_full_blocks` |

## 7. Every flag

132 flags are read by the code. "Set in your best run" lists what the 03j command sets by hand.
"Remove" means remove on `bff-core` only, and `v0.18.0` keeps the flag. `BFF_ALIAS_FUSED`,
`BFF_KEEP_FUSED_HASH` and `BFF_V2_ALIAS_TRACE` stay on both (PLAN.md decision D4), which overrides their
rows below. `BFF_V2_SHARED_WRITE_CHECK` stays on by default.

| Flag | Default | Set in your best run | Kind | What to do |
|---|---|---|---|---|
| `BFF_EDGE_LAYERS` | 0 | yes | core | keep |
| `BFF_FREE_FLUSH` | recv | yes | core | keep; verify the other modes are needed |
| `BFF_FREE_PRERUNNING` | 1 |  | core | fix to on, remove flag (P5) |
| `BFF_LSH_EXACT` | 0 | yes | core | keep; make it the only search if concurrency stays low |
| `BFF_MAX_REL_ERR` | 1.0 | yes | core | keep |
| `BFF_PD_FUSE` | 0 |  | core | keep: the master switch |
| `BFF_PD_STATS_DIR` | . |  | core | keep |
| `BFF_PD_STATS_DUMP_SEC` | 10 |  | core | keep |
| `BFF_PD_STATS_EVERY` | 50 |  | core | keep |
| `BFF_PD_STATS_MAX_AGE_S` | 30 |  | core | keep |
| `BFF_SIG_COMPUTE_DTYPE` | fp16 |  | core | fix to fp16, remove flag (P5) |
| `BFF_SIG_DIM` | 128 |  | core | keep |
| `BFF_THRESHOLD` | 0.75 | yes | core | keep |
| `BFF_V2_ALLOW_FULL_GRAPH` | 0 | yes | core | fix to on, remove flag (P5) |
| `BFF_V2_CLIP_FRAC` | 0 | yes | core | keep |
| `BFF_V2_CLIP_MAX_REL_ERR` | 1.0 |  | core | keep |
| `BFF_V2_CLIP_TAIL` | 2 |  | core | keep |
| `BFF_V2_DEDUP` | 1 | yes | core | keep: it is the control arm |
| `BFF_V2_FREE_AT_PROMOTION` | 0 | yes | core | keep |
| `BFF_V2_MATERIALIZE` | 0 | yes | core | fix to on, remove flag (P5) |
| `BFF_V2_MATERIALIZE_ON_LANDING` | 1 |  | core | fix to on, remove flag (P5) |
| `BFF_V2_MAX_REQ_DECLINE` | 0.5 |  | core | keep |
| `BFF_V2_MAX_RESIDENT` | 32768 |  | core | keep |
| `BFF_V2_PIGGYBACK_STASH_CAP` | 8192 |  | core | keep |
| `BFF_V2_PROTECT_HOT_BLOCKS` | 1 |  | core | fix to on, remove flag (P5) |
| `BFF_V2_RESIDENT` | 1 |  | core | fix to on, remove flag (P5) |
| `BFF_V2_SIG_PIGGYBACK` | 0 | yes | core | fix to on, remove flag (P5) |
| `BFF_V2_SIG_PRECOMPUTE_PER_STEP` | 32 |  | core | keep |
| `BFF_V2_SIG_PROJECTION` | by_layer |  | core | fix to by_layer, remove flag and concat (P5) |
| `BFF_V2_DEDUP_KV_THRESHOLD` | 0.0 |  | optional | keep: a real production gate, unused in your runs |
| `BFF_PD_SAVE_BARRIER` | 0 |  | open | run a test: the log warns the piggyback read may race without it |
| `BFF_MOONCAKE_FF_PULL_V2_PORT_OFFSET` | 22000 |  | fallback | keep |
| `BFF_PULL_V2_SIG_BIND_TIMEOUT` | 5 |  | fallback | keep |
| `BFF_PULL_V2_SIG_TIMEOUT` | 10 |  | fallback | keep while the on-demand exchange exists |
| `BFF_LSH_BITS_PER_TABLE` | 20 |  | index | keep unless exact-only |
| `BFF_LSH_COMPACT_DEAD_FRAC` | 0.25 |  | index | keep |
| `BFF_LSH_COMPACT_MIN_DEAD` | 256 |  | index | keep |
| `BFF_LSH_MAX_CANDIDATES` | 0 |  | index | keep unless exact-only |
| `BFF_LSH_MAX_ENTRIES` | 50000 |  | index | keep |
| `BFF_LSH_MAX_LIVE_OWNERS` | 512 |  | index | keep |
| `BFF_LSH_TABLES` | 16 |  | index | keep unless exact-only |
| `BFF_AUDIT_DESCRIPTORS` | 0 |  | diag | move to diagnostics (P5) |
| `BFF_MEM_PROBE` | 0 |  | diag | move out (P5) |
| `BFF_PD_DEBUG` | 0 |  | diag | keep |
| `BFF_RECV_TIMING` | 0 |  | diag | keep |
| `BFF_RUNNER_CPROFILE` | 0 |  | diag | keep |
| `BFF_SCHED_CPROFILE` | 0 |  | diag | keep |
| `BFF_SCHED_CPROFILE_START` | 4000 |  | diag | keep |
| `BFF_SCHED_CPROFILE_STEPS` | 2000 |  | diag | keep |
| `BFF_STEP_PROFILE` | 0 | yes | diag | keep |
| `BFF_V2_ALIAS_TRACE` | 1 |  | diag | remove with the alias (P5) |
| `BFF_V2_AUDIT_HOT_BLOCKS` | 0 |  | diag | move to diagnostics (P5) |
| `BFF_V2_PROMOTION_LOG_EVERY` | 50 |  | diag | keep |
| `BFF_V2_SHARED_WRITE_CHECK` | 1 |  | diag | default off after P5 |
| `BFF_V2_TRACE_SLOTS` | 0 |  | diag | move to diagnostics (P5) |
| `BFF_V2_VERIFY_DEGRADED_COS` | 0.95 |  | diag | move to diagnostics (P5) |
| `BFF_V2_VERIFY_MAX_NORM_ERR` | 0.10 |  | diag | move to diagnostics (P5) |
| `BFF_V2_VERIFY_MIN_COS` | 0.99 |  | diag | move to diagnostics (P5) |
| `BFF_V2_VERIFY_SPACING_S` | 15 |  | diag | move to diagnostics (P5) |
| `BFF_V2_VERIFY_TRANSFER` | 0 |  | diag | move to diagnostics (P5) |
| `BFF_XFER_AUDIT` | 0 |  | diag | move to diagnostics (P5) |
| `BFF_FF_DONE_TID_MAX` | 4096 |  | GPU | remove (P3) |
| `BFF_FF_DONE_TID_TTL` | 10 |  | GPU | remove (P3) |
| `BFF_FF_ROWS_MAX_PENDING` | 4096 |  | GPU | remove (P3) |
| `BFF_HOLD_LOG_S` | 10 |  | GPU | remove (P3) |
| `BFF_PD_AUDIT` | 0 |  | GPU | remove (P3) |
| `BFF_PD_AUDIT_PAIRS` | 512 |  | GPU | remove (P3) |
| `BFF_PD_AUDIT_STEPS` | 8 |  | GPU | remove (P3) |
| `BFF_PD_CKSUM` | 0 |  | GPU | remove (P3) |
| `BFF_PD_CKSUM_SAMPLE` | 1 |  | GPU | remove (P3) |
| `BFF_PD_FUSE_LOG_EVERY` | 50 |  | GPU | remove (P3) |
| `BFF_PD_LOG_EVERY` | 200 |  | GPU | remove (P3) |
| `BFF_PD_RESUME_LOCAL` | 1 |  | GPU | remove (P3) |
| `BFF_PD_TRACE` | 0 |  | GPU | remove (P3) |
| `BFF_THRESHOLD_G` | - |  | GPU | remove (P3) |
| `BFF_V2_READY_TIMEOUT` | 10 |  | GPU | remove (P3) |
| `VLLM_NCCL_SO_PATH` | - |  | GPU tool | remove (P3) |
| `BFF_KEEP_FUSED_HASH` | 0" if _SCALE_MODE  |  | alias support | remove with the alias (P5) |
| `BFF_PULL_V2_ASYNC_PLAN` | 0 |  | dead end | remove (P5): measured no gain |
| `BFF_PULL_V2_ASYNC_SIG` | 0 |  | dead end | remove (P5) |
| `BFF_PULL_V2_SIG_BATCH` | 64" if ASYNC_SIG e |  | dead end | remove (P5) |
| `BFF_PULL_V2_SIG_DEADLINE_MS` | 200 |  | dead end | remove (P5) |
| `BFF_PULL_V2_SIG_LINGER_MS` | 0 |  | dead end | remove (P5) |
| `BFF_V2_PRODUCER_SIGCACHE` | 0 |  | dead end | remove (P5): replaced by piggyback |
| `BFF_XFER_ASYNC` | 0 |  | dead end | remove (P5) |
| `BFF_XFER_ASYNC_FIRST_TIMEOUT` | 2 |  | dead end | remove (P5) |
| `BFF_XFER_ASYNC_POLL_S` | 0.001 |  | dead end | remove (P5) |
| `BFF_XFER_ASYNC_TIMEOUT` | 30 |  | dead end | remove (P5) |
| `BFF_SCALE_MODE` | raw |  | dead on NPU | remove with ratio (P3); Ascend supports raw only |
| `BFF_REBUILD_GROUPS` | 1 |  | debug | remove (P5) |
| `BFF_BALANCED_GROUPS` | 0 |  | dedup=0 layout | remove (P5): edge layout replaced it |
| `BFF_FF_ASYNC` | 1 |  | layerwise/v1 | remove (P4) |
| `BFF_FF_ASYNC_D2H` | 0 |  | layerwise/v1 | remove (P4) |
| `BFF_FF_PROMO_APPLY` | 1 |  | layerwise/v1 | remove (P4) |
| `BFF_FF_REP_SAFE` | 0 |  | layerwise/v1 | remove (P4) |
| `BFF_FF_TIMING` | 0 |  | layerwise/v1 | remove (P4) |
| `BFF_FF_WORKER_QUEUE` | 8 |  | layerwise/v1 | remove (P4) |
| `BFF_MOONCAKE_FF_PORT_OFFSET` | 20000 |  | layerwise/v1 | remove (P4) |
| `BFF_MOONCAKE_FF_V2_PORT_OFFSET` | 21000 |  | layerwise/v1 | remove (P4) |
| `BFF_PD_CROSS_INDEX` | lsh,matrix |  | layerwise/v1 | remove (P4) |
| `BFF_PD_ENCODED_BATCH_SIZE` | 0 |  | layerwise/v1 | remove (P4) |
| `BFF_PD_INTRA_REQ_FF` | 1 |  | layerwise/v1 | remove (P4) |
| `BFF_PD_MERGE` | nr_tree |  | layerwise/v1 | remove (P4) |
| `BFF_PD_PROJ_DIM` | 512 |  | layerwise/v1 | remove (P4) |
| `BFF_PD_REPR` | full |  | layerwise/v1 | remove (P4) |
| `BFF_SIG_LAYERS` | first |  | layerwise/v1 | remove (P4) |
| `BFF_V2_HYBRID_REDIRECT` | 1 |  | layerwise/v1 | remove (P4) |
| `BFF_V2_SIG_TIMEOUT` | 10,2 |  | layerwise/v1 | remove (P4) |
| `BFF_DORMANT_EMA` | 0.3 |  | legacy | remove (P2) |
| `BFF_DORMANT_THR` | 0.02 |  | legacy | remove (P2) |
| `BFF_FAST_SCATTER` | 1 |  | legacy | remove (P2) |
| `BFF_FUSE_MAX_GROUP` | -1 |  | legacy | remove (P2) |
| `BFF_LSH_PROJ_DIM` | 512 |  | legacy | remove (P2) |
| `BFF_LSH_REPR` | proj |  | legacy | remove (P2) |
| `BFF_MAX_REGISTRY` | 2048 |  | legacy | remove (P2) |
| `BFF_MERGE` | nr_tree |  | legacy | remove (P2) |
| `BFF_PROBE_EVERY` | 1 |  | legacy | remove (P2) |
| `VLLM_USE_V1` | - |  | legacy | remove (P2) |
| `BFF_ALIAS_FUSED` | 1 | yes | no benefit | decision D4: default off, then remove (P5) |
| `BFF_MAX_INFLIGHT_LOADS` | 0 |  | no effect | remove (P5): does not address evictions |
| `BFF_EVICT_FUSED` | 0 |  | old | remove (P5) |
| `BFF_HIT_DEBUG` | 0 |  | old | remove (P5) |
| `BFF_MAX_XFER_SEGMENTS` | 0 |  | old | remove (P5) |
| `BFF_FF_GROUPS` | - |  | old layout | remove (P5): only 2 groups now |
| `BFF_GROUP_SIZE` | 4 |  | old layout | remove (P5) |
| `BFF_FF_APPLY_MAX_AGE` | 16 |  | redirect | remove (P5) |
| `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` | - |  | upstream | copied vLLM behaviour, keep |
| `BFF_FF_APPLY` | 1 |  | v1 | remove (P4) |
| `BFF_FF_PROMOTED_SEEN_CAP` | 4096 |  | v1 | remove (P4) |
| `BFF_FF_REP_HISTORY_CAP` | 4096 |  | v1 | remove (P4) |
| `BFF_FF_SHIP` | 1 |  | v1 | remove (P4) |
| `BFF_FF_AUDIT` | 0 |  | v1 diag | remove (P4) |
| `BFF_ON` | - | | script only | internal to `run_benchmarks.sh`, keep |
| `BFF_FF_RID_LIVE` | - | | script only | read by no code: remove (P4) |
## 8. Directions tried, and what became of them

| Period | Direction | Outcome |
|---|---|---|
| Jun-Jul | Single-node BFF on GPU: compress KV during the forward (`legacy/`) | Superseded by P/D. |
| Jul | P/D over NCCL with the prefill clustering blocks and sending redirects (v1) | 71-73% of redirects resolved to nothing, because the prefill cannot see what the decode holds, and every block was still transferred. |
| Jul | The same design on Ascend layerwise, applied at promotion | Fixed many ordering bugs; same structural limit. |
| Aug | **v2: signatures from the prefill, the decode decides what not to read.** GPU, then Ascend layerwise, then Ascend pull | **The design in use.** |
| Aug-Sep | Chasing wrong KV on Ascend: `_verify_transfer`, self-consistency audit, replay, save barrier | Found real transfer issues. The tools remain, off by default. |
| Sep | Async signature exchange, pipeline, batching | Measured throughput-neutral. Replaced by piggyback. **Dead end.** |
| Sep | Prefill signature cache on a background thread | Caused 485 transfer timeouts when unpaced; replaced by piggyback. **Dead end.** |
| Sep | KV-usage gate, occupancy split, balanced groups, edge layout | Edge layout stays. The gate is kept but unused. Balanced groups replaced. |
| Sep | Full-graph decode and materialize (copy into own block) | **In use.** Fixed the metadata-order bug that made full graph garbage. |
| Sep | Hybrid models (Qwen3-Next, MiniMax) | Works, not exercised on GLM. Keep or drop is decision D2. |
| Oct | Profilers, free flush, LSH compaction, signing across TP | Flush and TP signing are in use; profiler is off by default. |
| Nov | GLM-5.1: exact search, free at promotion, clip, indexer and alias fixes | **In use.** See section 9. |
| Nov | Signing speed: per-layer projection, allocator and first-read theories | All three theories were wrong; the real cause was a one-time rebuild, fixed in 04a. |
| Nov | Overlap dataset | Rejected by you as not a real benchmark. |

## 9. Bugs, fixed and open

| # | Issue | State |
|---|---|---|
| 1 | Mamba groups dropped in `_initialize_kv_caches` | Fixed earlier; hybrid models only. |
| 2 | Alias copy skipped the DSA indexer keys (`kv_cache[2]`) | **Fixed in 03a, confirmed:** garbage 10/121 → 0/121. |
| 3 | Alias key equal to the representative's own hash left a block permanently marked hashed, so later pulls skipped it | **Fixed in 03e, confirmed:** 14-15/121 → 0/121, 10 aliases skipped, 0 stale-hash blocks. |
| 4 | Wire saving counted declines from requests read in full | Fixed in 03d. Real figure ~2.8% (5.6% of g1). |
| 5 | Warm-up built the projection matrix on the wrong group | Fixed in 04a. **Not yet seen on the box:** expect `signature path warmed on group 1`. |
| 6 | The recompute path pastes the partial answer into the user message | **Open, upstream proxy.** Caused one garbage output (`#141`). Affects baseline and BFF alike. |
| 7 | Over-admission at the decode leads to evictions | **Open.** A scheduling policy, separate from dedup. |
| 8 | Piggyback signatures may race the KV write without `BFF_PD_SAVE_BARRIER` | **Open question.** The log warns on every run. No damage seen, but never tested with the barrier on GLM. |
| 9 | The input-length cap in `f1_main.py` existed only on the box | **Committed in 04b** as `--max-in-tokens` / `MAX_IN_TOKENS`. `f1_main` now prints the prompt fingerprint. Check on the box that `MAX_IN_TOKENS=6000` reproduces `9f5e27b8` and `MAX_IN_TOKENS=2000` reproduces `4da4b383`. |
| 10 | `pd_dedup_v2.threshold_for` imported the GPU connector package | **Fixed in 04b.** It caused the 9 local test failures. Needs one box run to confirm. |

## 10. Tests and tooling

**Tests.** About 716 pass locally, 3 skipped, 9 fail, and 4 more files do not load. All 13 share one cause:
this machine's vLLM 0.31.0 has no `p2p` module, and bug 10 imports it. They are not code defects.
The local vLLM is not the box's 0.18.0, so a test passing here is weaker evidence than a test passing
there.

**Running and reading a run.**
- `run_benchmarks.sh`: launches everything for one run. 1,750 lines, 577 of them an inline stats
  report. `results_latest` and `logs_latest` are wiped at each start.
- `run_best_compare.sh`: sweep with every run saved to its own folder.
- `python -m f1_benchmark.compare_runs <folder>`: the table and verdict.
- `python -m f1_benchmark.degenerate_outputs <results> <decode log>`: garbage and attribution.

**Checks for any run.**
1. The decode log shows the build tag you expect.
2. The prompt fingerprint matches the comparison run.
3. Read garbage and CodeBLEU before throughput.
4. Treat differences under about 3% in wall time or 1 point of CodeBLEU as noise.
5. For a counter that logs cumulative values, read the slope between lines, not one line's average.
