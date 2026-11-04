# BFF cleanup plan

**State: P0 is done locally (build 04b) and awaits one box run. `bff-core` does not exist yet, and nothing has been removed.** The file and flag detail is in
[STATUS.md](STATUS.md) sections 5-7, using the phase labels below.

## 1. The approach

**`v0.18.0` keeps everything.** The removals happen only on a new branch, `bff-core`, which holds just
the pipeline with good results. Both can be open at once, because a git worktree gives the branch its
own folder:

```
git worktree add ../kv_fast_fusion_core -b bff-core v0.18.0
```

A plain copied folder inside the repo would be worse: every fix would have to be applied by hand in
two places, and nothing would show what differs. With a branch, `git diff` shows exactly that, and a
fix moves across with `git cherry-pick`.

## 2. Decisions

| # | Question | Decision | Effect |
|---|---|---|---|
| D1 | Remove the GPU stack? | **No, the option stays** | Stays on `v0.18.0`. Not on `bff-core`. |
| D2 | Keep hybrid (MiniMax, Qwen3-Next)? | **Yes** | Stays on both. |
| D3 | Keep layerwise? | **Yes, for now** | Stays on `v0.18.0`. Not on `bff-core`. |
| D4 | Remove the alias? | **No** | Stays on both, on by default, as in the best run. |
| D5 | One pipeline only? | **On a separate branch, and the rest is not removed** | This plan's structure. |
| D6 | Diagnostics | **Keep, in one module** | Done on `v0.18.0`, so both get it. |

## 3. Result

| | `v0.18.0` | `bff-core` (estimated) |
|---|---:|---:|
| Production lines | ~31,400 (unchanged apart from moves) | ~13,500 |
| Flags | 134 | ~58 |
| GPU, NCCL, ratio, legacy, layerwise | Yes | No |
| `bff_pull_v2` and `mooncakev1` arms | Yes | Yes |
| Alias, hybrid, diagnostics, fallbacks | Yes | Yes |

## 4. Rules

1. Before P0: `git tag pre-cleanup`. History is not rewritten.
2. Each phase is its own commit series, with a new tag in `kv_fast_fusion_ascend/build.py`.
3. Remove code, tests and flags together.
4. Run the tests after each commit, on the pinned vLLM from P0.
5. One behaviour change per golden run, otherwise a regression can't be attributed.
6. Check string and dynamic references by hand. `BASELINE` and `run_benchmarks.sh` select connectors
   by name.
7. **Keeping the branches in step:** while `bff-core` is being built, new fixes go on `v0.18.0` and are
   cherry-picked over. Once `bff-core` passes its final run, fixes to shared code go on whichever branch
   you are running, and are cherry-picked to the other. Expect conflicts to grow as the branches drift
   apart. That is the cost of keeping both.

### Golden run

Same settings as the 03j run (alias on), on the 121 prompts (`MIN_TOKENS=4096 MAX_IN_TOKENS=6000`),
fingerprint `9f5e27b8`.

| Check | Pass |
|---|---|
| Build tag in the decode log | The expected one |
| Prompt fingerprint | `9f5e27b8` |
| Garbage outputs | 0 |
| CodeBLEU | Within 1 point of 46.62 |
| F1 | Within 0.01 of 0.656 |
| Throughput | Within 3% of 197.6 tok/s |
| Blocks freed | 214-275 |
| Piggyback misses | 0 |
| `mooncakev1` arm | Starts and gives a sane result, at the end of the plan |

A difference inside those bands is noise, so this catches a break but not a small slowdown.
**Rollback:** `git revert` the phase, or check out the previous tag on the box.

## 5. Phases

| Phase | Branch | What | Lines removed | Box runs |
|---|---|---|---:|---:|
| P0 | `v0.18.0` | Untangle dependencies, one diagnostics module, repo hygiene | 0 | 1 |
| P1, P3, P2 | `bff-core` | Old copies, GPU stack, legacy | ~13,700 | 1 |
| P4 | `bff-core` | Layerwise, v1 fusion | ~2,950 | 1 |
| P5 | `bff-core` | Unused paths inside the live files | ~1,220 | 1-2 |
| P6 | `bff-core` | Scripts | small | 1 |

About 5-6 runs in total, and any could double as a run you were making anyway.

### P0. On `v0.18.0`, nothing removed

These moves help both branches and make later cherry-picks cleaner.

| Step | Why |
|---|---|
| Move `_THRESHOLD_G` from the GPU connector to `constants.py`, and re-export it | `pd_dedup_v2.threshold_for` imports the GPU package to read it. This one import causes all 13 local test problems. |
| Move `_selected_connector_names` out of `fast_fusion_pd_patch.py` | The Ascend failure path imports it from the GPU patch. |
| Move `signatures_for_group` from layerwise v2 to `pd_dedup_v2` | The live path needs it. Both connectors import it from there. |
| Break the cycle: `shared_write_check` imports `_ext_of` from the patch | Move `_ext_of` to a small shared module. |
| **D6:** create `kv_fast_fusion_ascend/diagnostics.py` | See below. |
| Pin a test environment to vLLM 0.18.0 | Local tests run against 0.31.0, not what the box runs. |
| Commit the `f1_main.py` input cap as `--max-in-tokens` | It exists only on the box, and losing it silently changed the prompt set once. **Needs your local change.** |
| Print the prompt fingerprint from `f1_main.py` | Makes a changed prompt set visible at once. |
| `.gitignore`: `nsys_report.*`, `report1.*`, `logs_for_Claude/`, `GLM logs for claude/`, `sweeps/`, `vllm/`, `vllm-ascend/`, `Mooncake/` | 729 MB of profiler output and dependency checkouts sit untracked in the repo folder. |

**By hand:**
- **Move `setting_claude_deepseek.json` out of the repo.** It was not opened and may hold credentials.
  One `git add -A` would commit it.
- `vllm/` is a 2.5 GB checkout from 2025, not the code your box runs. Replace it with 0.18.0.

**The diagnostics module (D6).** About 740 lines move out of `mooncake_connector_ff_v2.py`, with the
same flags and the same off defaults:

- Verify and audit: `VerifySampler`, `classify_mismatch`, `best_matching_row`, `cross_match`,
  `compare_signature_rows`, `_verify_transfer`, `_diagnose_mismatch`, `_localise`, `_replay`,
  `_stability`, `_recompute_local`, `_audit_self_consistency`, `_score_audit`, `_audit_buckets`,
  `_bucket_for`.
- Trace: `write_slot`, `slot_trace_fault`, `_trace_slots`, `_check_mirror_matches_device`,
  `hot_block_collisions`, `_audit_hot_blocks`.

Some are methods reading connector state, so they would take that state as an argument or become a
mixin. `hot_block_collisions` was checked: its only caller is the off-by-default audit, so it moves.
`_sync_device` also moves, and the connector still imports it for the signing profile. `shared_write_check.py`, `step_profile.py`, `RecvThreadTimer`
and the base connector's audit helpers can join later if the first move goes well.
`BFF_V2_SHARED_WRITE_CHECK` stays on by default, since the alias stays.

**Done when:** one golden run passes. The 13 local test problems are gone: on vLLM 0.18.0, 851 tests pass, 84 skip and none fail. Then tag `pre-core` and
create the branch.

### P1, P3, P2. On `bff-core`: everything the live path doesn't load

One commit each, one golden run for the group.

| Remove | Lines |
|---|---:|
| **P1.** `f1_main_o.py`, `f1_patch_.py`, `f1_dataset.py`, `make_overlap_dataset.py` | 741 |
| **P1.** `_handle_block_merging_with_counts_o`, `____`, `_` in the scheduler | 221 |
| **P1.** `plot_bff_metrics.py` | 152 |
| **P1.** `msg.sh`, `package.json`, `package-lock.json`, 56 tracked result files under `p2p_nccl_xpyd/f1_results` (38,998 lines) | not production |
| **P3.** `kv_fast_fusion/connectors/` (5 files) | 4,352 |
| **P3.** `fast_fusion_flash.py`, `fast_fusion_triton_attn.py`, `fast_fusion_pd_patch.py`, `pd_fuse.py` | 1,318 |
| **P3.** `tools/nccl_pd_repro.py`, `tools/bff_cksum_diff.py` | 141 |
| **P3.** `update_from_output`, `_bff_record_decode_free`, `_bff_dump_decode_stats` in the scheduler | ~400 |
| **P3.** GPU branch of `kv_fast_fusion/__init__.py`, GPU tests (3,036), GPU examples (2,204) | not counted |
| **P2.** `legacy/` (8 files) and `compression_hook.py` | 6,585 |

**Check first:**
- `pd_fuse.py` may be imported by the Ascend v1 fusion code. If so, it goes in P4.
- `_wrap_scheduler_update_from_output` must wrap vLLM's method, not the one being removed.
- `_update_hashes_for_merged_blocks` and `_create_block_id_hash` may be needed by the alias code, so
  remove them only if nothing refers to them.

Keep `patch/`: it holds the fixes applied to vLLM, vllm-ascend and Mooncake on the box.

### P4. On `bff-core`: layerwise and v1 fusion

| Remove | Lines |
|---|---:|
| `mooncake_layerwise_connector_ff.py`, `mooncake_layerwise_connector_ff_v2.py` | 2,733 |
| V1 fusion in the base connector: `FFRowStash`, `FFPendingSource`, `normalize_ff_redirects`, `_ff_pending_source`, `_ff_accumulate`, `save_kv_layer`, the `MooncakeFFProducer` import | ~220 |
| The `register_*` calls for those connectors, `BFF_FF_RID_LIVE` in the script | small |
| Layerwise tests | 1,794 (tests) |

`_ff_write_runner_block_table`, `_ext_hash` and `resolve_redirect_rows` serve only the redirect path, so
P4 and the P5 redirect step must land together.

**Do not remove:** `_collect_fuse_reqs`, `ff_sig_*` and `FFSigStash`. Their names look like v1 fusion,
but they carry the piggyback signatures on the live path.

### P5. On `bff-core`: inside the live path

| What | Lines | Risk |
|---|---:|---|
| Redirect: `AliasApplier`, `_write_block_table`, `_bff_apply_block_merge`, `_bff_promotion_apply`, rep history, `_rep_gone_bucket`, `_bff_sweep_late_maps` | ~600 | Medium, in the promotion path. First confirm the alias does not use them. |
| Async: `partition_by_age`, `SignaturePipeline`, `_sig_pipeline`, `_exchange_for`, `_plan_for`, `_run_async` | ~230 | Low |
| Batching: `BatchCoalescer`, `PendingAsk`, `_SigClient.ask`. **Keep `ask_many` and `_exchange`.** | ~92 | Low |
| Signature cache: `split_cached_blocks`, `precompute_signatures`, `drain_precompute`, `serve_signatures_cached` | ~100 | Low |
| Signing profile: `_sig_phase`, `_sig_cold_first`, `_sig_cold_done`, `_sig_alt_check` | ~100 | Low |
| Ratio in `pd_dedup_v2`: `SCALE_MODE`, `block_layer_norms`, `value_blocks`, `kv_norms`, `_block_kvn`, `_scale_for` | ~100 | Low |
| Fix always-on flags as constants, delete the off branches (section 6) | small | Low |

**Stays, by decision or because it needs one:**

- **Alias:** `add_block_alias`, `patched_maybe_evict_cached_block`, `_trace_hits`,
  `patched_get_computed_blocks`, `patched_cache_full_blocks`, `BFF_ALIAS_FUSED`, `BFF_KEEP_FUSED_HASH`,
  `BFF_V2_ALIAS_TRACE` (D4).
- **Hybrid:** `hybrid_groups.py`, `hybrid_layout.py`, `_narrow_mamba_groups` and related code (D2).
- **Fallbacks:** `_SigServer`, `_SigClient`, `_repull_refused`, local recompute. They never fired in the
  best run, but they keep a request alive when signing or a pull fails.
- `BFF_V2_DEDUP_KV_THRESHOLD`: a real production option.
- `admission_deferred`, `note_loads_finished`: a natural hook for an admission policy, if that question
  is ever pursued.
- `_sync_producer_writes`, `BFF_PD_SAVE_BARRIER`: **test before any decision.** The log warns on every
  run that the signature read may race the KV write, and this is the fix for that race.

### P6. On `bff-core`: scripts

| Change | Effect |
|---|---|
| Reduce `BASELINE` arms to `bff_pull_v2` and `mooncakev1` | `BFF_V2_DEDUP=0` already gives the control. |
| Replace the 577-line inline `collect_bff_stats()` with `tools/collect_bff_stats.py` | **Check first that they produce the same report.** They evolved separately. |
| Move the remaining report code to Python under `f1_benchmark/` | `run_benchmarks.sh` falls from 1,750 to about 700 lines. |

**Done when:** a final run of both arms gives the same report lines as today.

## 6. Flags on `bff-core`

STATUS section 7 lists each flag. Its "remove" entries mean **remove on `bff-core` only**.

| Group | Flags | Examples |
|---|---:|---|
| Stay | ~58 | `BFF_EDGE_LAYERS`, `BFF_THRESHOLD`, `BFF_MAX_REL_ERR`, `BFF_V2_CLIP_FRAC`, `BFF_V2_DEDUP`, `BFF_V2_FREE_AT_PROMOTION`, `BFF_ALIAS_FUSED`, LSH settings, profilers, diagnostics |
| P2 | 10 | `BFF_MERGE`, `BFF_DORMANT_*`, `BFF_FAST_SCATTER` |
| P3 | 17 | `BFF_PD_CKSUM`, `BFF_THRESHOLD_G`, `BFF_SCALE_MODE`, `VLLM_NCCL_SO_PATH` |
| P4 | 22 | `BFF_FF_*`, `BFF_PD_MERGE`, `BFF_SIG_LAYERS` |
| P5, removed | 18 | `BFF_PULL_V2_ASYNC_*`, `BFF_XFER_ASYNC*`, `BFF_V2_PRODUCER_SIGCACHE`, `BFF_GROUP_SIZE` |
| P5, fixed as constants | 9 | `BFF_V2_MATERIALIZE`, `BFF_V2_SIG_PIGGYBACK`, `BFF_V2_ALLOW_FULL_GRAPH`, `BFF_V2_RESIDENT`, `BFF_V2_SIG_PROJECTION` |

## 7. Risks

| Risk | What protects against it |
|---|---|
| The branches drift apart, and fixes stop applying cleanly | P0 moves the shared pieces first. Rule 7. |
| A removal breaks something a golden run doesn't exercise, such as a fallback or another TP | The fallbacks stay. Check dynamic references by hand. |
| A small regression hides in the noise (±3%) | One behaviour change per run. Compare prefill tokens and freed blocks too. |
| Cleanup competes with the throughput question for box time | Combine golden runs with runs you would make anyway. |
| The box and the repo drift apart | Pin vLLM. Commit the box-only `f1_main.py` change. Keep `patch/` current. |

## 8. Estimates, and what was not verified

- Whole-file line counts were measured. Those marked `~` come from a function outline, and the final
  sizes and flag counts by group are estimates.
- "Nothing refers to it" came from searching the Python and the scripts. Repeat the search before each
  removal.
- No phase has been tried.
- Test counts come from the last local run on vLLM 0.31.0.
