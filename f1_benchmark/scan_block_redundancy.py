"""How much block-level near-duplication do CONCURRENT prompts of a dataset contain? (CPU proxy.)

BFF can only free a decode block when another request that is live at the same time holds a block
whose KV is close to it. That needs similar tokens first, so a token-level scan bounds what any
threshold could achieve, without an NPU:

* each prompt is templated and tokenized as the server would see it, and cut into its full
  ``--block-size``-token blocks at their real positions;
* prompts arrive in dataset order, and the ``--window`` most recent ones are "live" (≈ mean Running);
* a block counts as a **live near-duplicate** when some block of a live prompt (or an earlier block
  of the same prompt) shares a token-shingle Jaccard ≥ ``--jaccard``;
* of those, **same-position** ones match a block at the same index — same RoPE phase. A match
  only at a shifted position has rotated K, so it bounds a position-invariant signature, not the
  current one;
* blocks whose whole prefix chain matches a live prompt's are **prefix-cached** instead — vLLM's
  prefix cache already shares them, so they are not BFF's to claim and are excluded from the above.

It is an UPPER BOUND: similar tokens are necessary for similar KV, not sufficient (context changes
KV in deeper layers). The truth is a GLM run's ``compression potential`` report line; a dataset is
worth that run only if this scan finds something.

    python3 -m f1_benchmark.scan_block_redundancy --model M \\
        --dataset ise-uiuc/Magicoder-Evol-Instruct-110K --input-key instruction
"""

import argparse
import json
import sys
from collections import Counter, deque


def shingles(ids, n):
    """Hashes of the token n-grams of one block (a set; short blocks fall back to unigrams)."""
    if len(ids) < n:
        return {hash((t,)) for t in ids}
    return {hash(tuple(ids[i:i + n])) for i in range(len(ids) - n + 1)}


def scan(prompts_ids, block_size=128, window=18, ngram=4, jaccard=0.8):
    """Count, over full blocks, prefix-cached / live near-duplicate / exact-duplicate blocks.

    ``prompts_ids`` is a list of token-id lists in arrival order. Returns a dict of counts."""
    live = deque()                  # (prompt index, [block keys]) of the live window
    index: dict[int, set] = {}      # shingle -> live block keys
    blocks: dict[tuple, tuple] = {} # block key -> (shingle set, exact tuple)
    chains: dict[tuple, int] = {}   # prefix-chain hash of a live block -> refcount
    out = Counter()

    def _add(key, sh, exact, chain):
        blocks[key] = (sh, exact, chain)
        for s in sh:
            index.setdefault(s, set()).add(key)
        chains[chain] = chains.get(chain, 0) + 1

    def _drop(key):
        sh, _exact, chain = blocks.pop(key)
        for s in sh:
            ks = index.get(s)
            if ks is not None:
                ks.discard(key)
                if not ks:
                    del index[s]
        chains[chain] -= 1
        if not chains[chain]:
            del chains[chain]

    for p, ids in enumerate(prompts_ids):
        # ``window`` live requests INCLUDING this one: evict before looking.
        while len(live) >= window:
            _p, old = live.popleft()
            for k in old:
                _drop(k)
        keys, chain = [], 0
        n_full = len(ids) // block_size
        for b in range(n_full):
            blk = ids[b * block_size:(b + 1) * block_size]
            exact = tuple(blk)
            chain = hash((chain, exact))
            out["blocks"] += 1
            if chain in chains:
                out["prefix_cached"] += 1
            else:
                sh = shingles(blk, ngram)
                overlap = Counter()
                for s in sh:
                    for k in index.get(s, ()):
                        overlap[k] += 1
                best, is_exact, same_pos = 0.0, False, False
                for k, inter in overlap.items():
                    other = blocks[k][0]
                    j = inter / (len(sh) + len(other) - inter)
                    best = max(best, j)
                    if blocks[k][1] == exact:
                        is_exact = True
                    if j >= jaccard and k[1] == b:
                        same_pos = True
                if best >= jaccard:
                    out["live_near_dup"] += 1
                    # Same block index = same absolute positions, so the same RoPE phase. A match
                    # only at a shifted position has rotated K, which a KV signature will not see
                    # as similar however close the tokens are.
                    if same_pos:
                        out["live_near_dup_same_pos"] += 1
                    if is_exact:
                        out["live_exact_dup"] += 1
            key = (p, b)
            _add(key, shingles(blk, ngram), exact, chain)
            keys.append(key)
        live.append((p, keys))
    return dict(out)


def _text(item, keys):
    parts = []
    for k in keys.split("+"):
        v = item.get(k)
        if isinstance(v, list):
            v = "\n".join(x.get("content", "") if isinstance(x, dict) else str(x) for x in v)
        parts.append(str(v or ""))
    return "\n\n".join(p for p in parts if p)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="tokenizer / chat template the server uses")
    p.add_argument("--dataset", required=True, help="HF name or local .jsonl")
    p.add_argument("--config", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--input-key", required=True, help="field, or fields joined with '+'")
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--window", type=int, default=18, help="live requests (≈ mean Running)")
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--ngram", type=int, default=4)
    p.add_argument("--jaccard", type=float, default=0.8)
    p.add_argument("--min-tokens", type=int, default=0,
                   help="skip shorter templated prompts (match the run's MIN_TOKENS filter)")
    p.add_argument("--enable-thinking", action="store_true")
    args = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if args.dataset.endswith(".jsonl"):
        ds = load_dataset("json", data_files=args.dataset, split=args.split)
    else:
        ds = load_dataset(args.dataset, args.config, split=args.split)
    prompts = []
    for item in ds:
        if len(prompts) >= args.num_prompts:
            break
        text = _text(item, args.input_key)
        if not text:
            continue
        templated = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                            add_generation_prompt=True,
                                            enable_thinking=args.enable_thinking)
        ids = tok(templated, add_special_tokens=False).input_ids
        if len(ids) >= args.min_tokens:
            prompts.append(ids)

    c = scan(prompts, args.block_size, args.window, args.ngram, args.jaccard)
    n = c.get("blocks", 0) or 1
    row = {"dataset": args.dataset + (f":{args.config}" if args.config else ""),
           "input_key": args.input_key, "prompts": len(prompts),
           "mean_prompt_tokens": round(sum(map(len, prompts)) / max(1, len(prompts))),
           "full_blocks": c.get("blocks", 0),
           "prefix_cached_pct": round(100.0 * c.get("prefix_cached", 0) / n, 2),
           "live_near_dup_pct": round(100.0 * c.get("live_near_dup", 0) / n, 2),
           "live_near_dup_same_pos_pct": round(100.0 * c.get("live_near_dup_same_pos", 0) / n, 2),
           "live_exact_dup_pct": round(100.0 * c.get("live_exact_dup", 0) / n, 2),
           "window": args.window, "jaccard": args.jaccard, "ngram": args.ngram}
    print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
