"""Build a prompt set whose CONCURRENT requests share near-duplicate KV that prefix caching cannot.

BFF's capacity gain needs live requests whose prompt blocks are near-duplicates of each other. On
Magicoder there are none (a correct dedup index finds 0 matches at con48), so the mechanism cannot
be judged there. Here each prompt is

    [distinct preamble, padded so the document starts at token D]  +  [one of K shared documents]
    +  [a short task]

* The preambles differ, so the prompts share no prefix and exact prefix caching hits nothing.
* Every document starts at the same templated token offset D (a multiple of the block size), so
  its blocks line up at identical positions across requests: same tokens, same RoPE phase. They
  differ only by what attention mixed in from the preamble — the near-duplicate regime BFF targets.
* Small K and short outputs keep many copies of each document resident at once and make
  transferred prompt KV the bulk of the decode pool.

Alignment is MEASURED with the model's own chat template and tokenizer, not assumed: the script
pads each preamble until the templated prefix before the document is exactly D tokens, checks that
the prefix tokenizes the same alone as inside the full prompt, and fails loudly otherwise.

Quality has no ground truth here. Run the baseline first, then write a copy whose references ARE
the baseline's completions, and rescore any run against it with f1_benchmark.rescore:

    python3 -m f1_benchmark.make_overlap_dataset --model M --out overlap.jsonl
    # ... baseline run with F1_DATASET=overlap.jsonl F1_INPUT_KEY=query F1_OUTPUT_KEY=reference
    python3 -m f1_benchmark.make_overlap_dataset --model M --out overlap_ref.jsonl \\
        --references-from <baseline results>/raw_outputs.json
    python3 -m f1_benchmark.rescore <bff results>/raw_outputs.json --dataset-path overlap_ref.jsonl \\
        --model M
"""

import argparse
import json
import random
import sys

TASKS = (
    "Summarize what the code in the shared context does in three sentences.",
    "List the functions or classes defined in the shared context, one line each.",
    "Point out one possible bug in the shared context and explain it briefly.",
    "Suggest two unit tests for the shared context and describe what each checks.",
    "Explain how the first part of the shared context could be made faster.",
    "Describe the inputs and outputs of the main routine in the shared context.",
    "Rewrite the docstring of the first function in the shared context more clearly.",
    "Which parts of the shared context would you refactor first, and why?",
)

HEADER = "\n\n### Shared context\n"
FILLERS = (" ok", " x", " .", " -")


def _load_texts(dataset, split, n):
    from datasets import load_dataset
    ds = load_dataset(dataset, split=split)
    out = []
    for item in ds:
        ins, res = item.get("instruction") or "", item.get("response") or ""
        if ins and res:
            out.append((ins, res))
        if len(out) >= n:
            break
    return out


def _templated(tok, content, enable_thinking):
    return tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=enable_thinking)


def _ids(tok, text):
    return tok(text, add_special_tokens=False).input_ids


def build_document(tok, texts, n_tokens):
    """Concatenate code responses until ``n_tokens``, then cut at exactly that many tokens."""
    buf = ""
    for _ins, res in texts:
        buf += res.strip() + "\n\n"
        if len(_ids(tok, buf)) > n_tokens:
            break
    doc = tok.decode(_ids(tok, buf)[:n_tokens])
    # Leading whitespace would merge with the header's newline and shift the boundary.
    return doc.lstrip()


def build_prompt(tok, core, doc, task, target, enable_thinking):
    """Pad ``core`` so the templated prefix before ``doc`` is exactly ``target`` tokens.

    Returns ``(prompt, offset)``; offset != target means it could not be hit exactly."""
    def offset(pre):
        text = _templated(tok, pre + HEADER + doc + "\n\n### Task\n" + task, enable_thinking)
        return len(_ids(tok, text[:text.index(doc)])), text

    core_ids = _ids(tok, core)
    for filler in FILLERS:
        lo, hi = 0, len(core_ids)
        while True:     # shrink the core until the unpadded prefix fits under the target
            pre = tok.decode(core_ids[:hi])
            n, _ = offset(pre)
            if n <= target or hi == lo:
                break
            hi = max(lo, hi - (n - target) - 1)
        for _ in range(4 * target):
            n, text = offset(pre)
            if n >= target:
                break
            pre += filler
        if n == target:
            full = _ids(tok, text)
            prefix = _ids(tok, text[:text.index(doc)])
            if full[:len(prefix)] == prefix:
                return pre + HEADER + doc + "\n\n### Task\n" + task, n
    return None, n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="tokenizer / chat template to align against")
    p.add_argument("--out", required=True)
    p.add_argument("--num-prompts", type=int, default=200)
    p.add_argument("--docs", type=int, default=2, help="shared documents (fewer = more overlap)")
    p.add_argument("--doc-tokens", type=int, default=3584)
    p.add_argument("--block-size", type=int, default=128)
    p.add_argument("--preamble-blocks", type=int, default=2,
                   help="the document starts at preamble_blocks * block_size templated tokens")
    p.add_argument("--source-dataset", default="ise-uiuc/Magicoder-Evol-Instruct-110K")
    p.add_argument("--split", default="train")
    p.add_argument("--enable-thinking", action="store_true",
                   help="match f1_main's --enable-thinking (the template differs)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--references-from", default=None,
                   help="raw_outputs.json of a baseline run: its completions become the references")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    rng = random.Random(args.seed)
    texts = _load_texts(args.source_dataset, args.split, args.num_prompts + 400 * args.docs)
    preambles, pool = texts[:args.num_prompts], texts[args.num_prompts:]
    per_doc = len(pool) // args.docs
    docs = [build_document(tok, pool[i * per_doc:(i + 1) * per_doc], args.doc_tokens)
            for i in range(args.docs)]
    target = args.preamble_blocks * args.block_size

    refs = None
    if args.references_from:
        with open(args.references_from) as f:
            saved = json.load(f)
        refs = {sid: o.get("generated_text") or ""
                for sid, o in zip(saved.get("sample_ids") or [], saved["outputs"])}

    rows, misses = [], 0
    for i, (ins, _res) in enumerate(preambles):
        core = f"You are assisting developer #{i}. Their background request was: {ins}"
        prompt, _n = build_prompt(tok, core, docs[i % args.docs], rng.choice(TASKS), target,
                                  args.enable_thinking)
        if prompt is None:
            misses += 1
            continue
        sid = str(len(rows))
        rows.append({"query": prompt,
                     "reference": (refs.get(sid) if refs else None) or "n/a",
                     "doc": i % args.docs})
    with open(args.out, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    n_tok = [len(_ids(tok, _templated(tok, r["query"], args.enable_thinking))) for r in rows[:5]]
    print(f"wrote {len(rows)} prompts to {args.out} ({misses} could not be aligned) | "
          f"{args.docs} shared document(s) of {args.doc_tokens} tokens, each at templated token "
          f"{target} | first prompts {n_tok} tokens")
    if refs is not None:
        print(f"references taken from {args.references_from} "
              f"({sum(1 for r in rows if r['reference'] != 'n/a')} of {len(rows)} non-empty)")
    return 0 if rows and not misses else 1


if __name__ == "__main__":
    sys.exit(main())
