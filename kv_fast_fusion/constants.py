import os

# Shared BFF constants used by both P/D and (legacy) single-instance paths.
# Single-instance code (kv_fast_fusion_graph_runner.py) re-reads the same
# env vars and keeps its own copy; these are the canonical values for P/D.
# NOTE: block_size is NOT kept here — read it from vllm_config.cache_config.block_size
# at runtime so it always matches the --block-size CLI argument.

THRESHOLD = float(os.environ.get("BFF_THRESHOLD", "0.75"))
BFF_GROUP_SIZE = int(os.environ.get("BFF_GROUP_SIZE", "4"))
