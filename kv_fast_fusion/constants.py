import logging
import os

# Shared BFF constants used by both P/D and (legacy) single-instance paths.
# Single-instance code (kv_fast_fusion_graph_runner.py) re-reads the same
# env vars and keeps its own copy; these are the canonical values for P/D.
# NOTE: block_size is NOT kept here — read it from vllm_config.cache_config.block_size
# at runtime so it always matches the --block-size CLI argument.

THRESHOLD = float(os.environ.get("BFF_THRESHOLD", "0.75"))


def parse_thresholds(raw: str | None) -> dict[int, float]:
    """Parse ``BFF_THRESHOLD_G`` ("1:0.97,2:0.90") into ``{group: threshold}``.

    One bar for every group is a guess, and the wrong one: a group's blocks have a natural
    similarity FLOOR (early layers share a large common component, so arbitrary blocks already sit
    near cosine 0.9), and a threshold under that floor merges everything indiscriminately. Groups
    not listed fall back to ``BFF_THRESHOLD``. Run with ``BFF_PD_AUDIT=1`` to measure the floor
    instead of guessing it."""
    out: dict[int, float] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            gi, thr = part.split(":")
            out[int(gi)] = float(thr)
        except ValueError:
            logging.getLogger(__name__).warning("BFF: ignoring malformed BFF_THRESHOLD_G entry %r", part)
    return out


# Per-group merge thresholds, overriding the global BFF_THRESHOLD.
THRESHOLD_G = parse_thresholds(os.environ.get("BFF_THRESHOLD_G"))

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
# Two-group "edge + middle" layout for dedup=1: group 0 = the first K + last K layers, group 1 = the
# middle. The uniform layers sit at BOTH ends of the model, and group 0 is never dedup-eligible, so
# this quarantines them by construction (no BFF_FF_GROUPS) with the minimum group count. K=7 on a
# 28-layer model gives 14+14, i.e. max_layers×num_groups at its floor. 0 = off; takes precedence over
# BFF_BALANCED_GROUPS.
BFF_EDGE_LAYERS = int(os.environ.get("BFF_EDGE_LAYERS", "0"))
