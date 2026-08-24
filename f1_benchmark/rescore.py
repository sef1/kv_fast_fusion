"""Score a run's completions offline, from the raw_outputs.json f1_main.py checkpoints.

Inference is the expensive half — ~25 minutes of NPU time for 1024 prompts — and scoring is seconds.
Keeping them separable means a scoring change (a metric fix, a different threshold, a bug like the
`ast.parse` MemoryError that took down con200) costs a re-score rather than a re-run.

Deliberately a standalone script rather than a flag on f1_main: the inference and latency-metrics
blocks sit between dataset loading and scoring in `main()`, so a `--rescore` path would have meant
restructuring the live benchmark for an offline convenience. This imports the same scoring functions,
so the numbers are the same ones f1_main would print.

Usage:
    python3 -m f1_benchmark.rescore RAW_OUTPUTS_JSON [--dataset-path P] [--num-prompts N]
                                    [--input-key K] [--output-key K] [--model M]
"""

import argparse
import json
import sys

import numpy as np

from f1_benchmark.f1_main import (
    check_syntax_validity,
    compute_codebleu_safe,
    f1_score,
    load_dataset_simple,
    normalized_exact_match,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_outputs", help="raw_outputs.json written next to benchmark_results.json")
    p.add_argument("--dataset-path", default="codeparrot_f1_benchmark.jsonl")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--input-key", default="query")
    p.add_argument("--output-key", default="reference")
    p.add_argument("--num-prompts", type=int, default=0,
                   help="0 = as many as the checkpoint holds")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B")
    p.add_argument("--min-tokens", type=int, default=0)
    p.add_argument("--lang", default="python")
    p.add_argument("--out", default=None, help="write the recomputed metrics here as JSON")
    args = p.parse_args()

    with open(args.raw_outputs) as f:
        saved = json.load(f)
    outputs = saved["outputs"]
    print(f"Loaded {len(outputs)} completions from {args.raw_outputs}")

    n = args.num_prompts or len(outputs)
    _prompts, references = load_dataset_simple(
        args.dataset_path, args.hf_split, args.input_key, args.output_key, n,
        model=args.model, min_tokens=args.min_tokens,
    )
    # Prefer the ids recorded at run time: the dataset filter (min_tokens) is order-dependent, so
    # re-deriving them can silently pair a completion with the wrong reference.
    sample_ids = saved.get("sample_ids") or list(references.keys())

    f1_scores, ast_scores, em_scores = [], [], []
    preds, refs = [], []
    n_err = 0
    for output, sample_id in zip(outputs, sample_ids):
        if not (output.get("success") and output.get("generated_text") is not None):
            continue
        gt = references.get(sample_id)
        if not gt:
            continue
        pred = output["generated_text"]
        preds.append(pred)
        refs.append(gt)
        try:
            f1_scores.append(f1_score(pred, gt))
            ast_scores.append(1.0 if check_syntax_validity(output.get("prompt", ""), pred) else 0.0)
            em_scores.append(1.0 if normalized_exact_match(pred, gt) else 0.0)
        except Exception as e:
            n_err += 1
            print(f"  [warn] scoring failed for sample {sample_id}: {type(e).__name__}: {e}")

    if n_err:
        print(f"  [warn] {n_err} sample(s) excluded from the means below")
    if not f1_scores:
        print("No scoreable samples found — is the dataset/--num-prompts the same as the run?")
        return 1

    results = {
        "mean_f1": float(np.mean(f1_scores)),
        "ast_syntax_validity_pct": float(np.mean(ast_scores)) * 100,
        "normalized_exact_match_pct": float(np.mean(em_scores)) * 100,
        "n_scored": len(f1_scores),
        "n_scoring_errors": n_err,
    }
    print("\n=== Evaluation Results (re-scored) ===")
    print(f"  Mean F1 Score: {results['mean_f1']:.4f} (over {results['n_scored']} samples)")
    print(f"  AST Syntax Validity: {results['ast_syntax_validity_pct']:.2f}%")
    print(f"  Normalized Exact Match: {results['normalized_exact_match_pct']:.2f}%")

    cb = compute_codebleu_safe(preds, refs, args.lang)
    if cb:
        results["codebleu"] = cb
        print(f"  CodeBLEU Score: {cb.get('codebleu', 0) * 100:.2f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nMetrics saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
