import os

# Shared BFF constants used by both P/D and (legacy) single-instance paths.
# Single-instance code (kv_fast_fusion_graph_runner.py) re-reads the same
# env vars and keeps its own copy; these are the canonical values for P/D.
# NOTE: block_size is NOT kept here — read it from vllm_config.cache_config.block_size
# at runtime so it always matches the --block-size CLI argument.

THRESHOLD = float(os.environ.get("BFF_THRESHOLD", "0.75"))
BFF_GROUP_SIZE = int(os.environ.get("BFF_GROUP_SIZE", "4"))

# Balanced-groups layout (dedup=0). Default 0 = the warmup(4)+uniform-chunk split above.
# When >= 2, ALL layers are split into this many BALANCED contiguous storage groups instead.
# Rationale (Update 17): concurrency ∝ 1/(max_layers_per_group × num_groups) — the pool is sized
# by the fattest group — so a balanced partition holds max_layers×num_groups at its floor
# (== total layers == baseline concurrency) while minimizing the per-step block-table tax
# (commit_block_table + compute_slot_mapping run once per group). N=2 is the concurrency-optimal
# minimum (>1 group is required by verify_and_split). Only meaningful with BFF_V2_DEDUP=0, where
# the group split is a pure storage layout and no layer is fused.
BFF_BALANCED_GROUPS = int(os.environ.get("BFF_BALANCED_GROUPS", "0"))
