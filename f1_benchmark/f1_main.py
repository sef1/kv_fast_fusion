import argparse
import asyncio
import json
import os
import time
import ast
import aiohttp
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import AsyncGenerator, List, Dict, Any, Tuple


# ==========================================
# Evaluation Metrics
# ==========================================

def f1_score(prediction: str, ground_truth: str, **kwargs) -> float:
    """Token-level F1 score overlap."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


import ast
import re
import textwrap

def extract_code(text: str) -> str:
    """Strips Markdown backticks and conversational prose."""
    if "```" in text:
        blocks = re.findall(r"```(?:python)?\n?(.*?)```", text, re.DOTALL)
        if blocks:
            return blocks[0].strip()
    return text.strip()

def check_syntax_validity(prompt: str, prediction: str) -> bool:
    """Checks syntax by combining prompt + prediction or dedenting the fragment."""
    clean_pred = extract_code(prediction)
    if not clean_pred:
        return False
    
    # 1. Try parsing full concatenated script
    try:
        ast.parse(prompt + "\n" + clean_pred)
        return True
    except SyntaxError:
        pass
        
    # 2. Try parsing prediction alone after dedenting
    try:
        ast.parse(textwrap.dedent(clean_pred))
        return True
    except SyntaxError:
        return False

def normalized_exact_match(prediction: str, ground_truth: str) -> bool:
    """Exact match comparison ignoring extra blank lines and trailing whitespace."""
    norm_pred = "\n".join([line.strip() for line in prediction.splitlines() if line.strip()])
    norm_gt = "\n".join([line.strip() for line in ground_truth.splitlines() if line.strip()])
    return norm_pred == norm_gt
def compute_codebleu_safe(predictions: List[str], references: List[str], lang: str = "python") -> Dict[str, float]:
    """Safely computes CodeBLEU if `codebleu` package is available."""
    try:
        from codebleu import calc_codebleu
        results = calc_codebleu(
            references=[[ref] for ref in references],
            predictions=predictions,
            lang=lang,
            weights=(0.25, 0.25, 0.25, 0.25)
        )
        return {
            "codebleu": float(results["codebleu"]),
            "ngram_match": float(results.get("ngram_match_score", 0.0)),
            "weighted_ngram_match": float(results.get("weighted_ngram_match_score", 0.0)),
            "syntax_match": float(results.get("syntax_match_score", 0.0)),
            "dataflow_match": float(results.get("dataflow_match_score", 0.0)),
        }
    except (ImportError, ModuleNotFoundError, NameError):
        print("Notice: `codebleu` package is not installed or imported. Skipping CodeBLEU.")
        return None
    except Exception as e:
        print(f"Warning: CodeBLEU computation encountered an error: {e}")
        return None


# ==========================================
# Async Request Generators & Parser
# ==========================================

async def get_request(
    prompts: List[str],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[int, str], None]:
    """Asynchronously yield (index, prompt) paced by the request rate."""
    assert burstiness > 0, f"A positive burstiness factor is expected, but given {burstiness}."
    for i, prompt in enumerate(prompts):
        yield i, prompt
        if request_rate == float("inf"):
            continue
        theta = 1.0 / (request_rate * burstiness)
        interval = np.random.gamma(shape=burstiness, scale=theta)
        if interval > 0:
            await asyncio.sleep(interval)


async def _parse_stream(response, st: float) -> Dict[str, Any]:
    """Parse an OpenAI chat-completions SSE stream, timestamping tokens for latency metrics."""
    ttft = 0.0
    most_recent = st
    itl: List[float] = []
    parts: List[str] = []
    finish_reason = None
    completion_tokens = None
    async for raw in response.content:
        line = raw.decode("utf-8", "ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        data_str = line[len("data:"):].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage")
        if usage:
            completion_tokens = usage.get("completion_tokens")
        choices = chunk.get("choices") or []
        if not choices:
            continue
        c0 = choices[0]
        if c0.get("finish_reason"):
            finish_reason = c0["finish_reason"]
        tok = (c0.get("delta") or {}).get("content")
        if tok:
            now = time.perf_counter()
            if ttft == 0.0:
                ttft = now - st
            else:
                itl.append(now - most_recent)
            most_recent = now
            parts.append(tok)
    content = "".join(parts)
    out_tokens = completion_tokens if completion_tokens is not None else (
        len(itl) + 1 if parts else 0)
    return {
        "generated_text": content,
        "finish_reason": finish_reason,
        "completion_tokens": completion_tokens,
        "gen_chars": len(content),
        "ttft": ttft if ttft > 0 else None,
        "itl": itl,
        "latency": time.perf_counter() - st,
        "output_tokens": out_tokens,
    }


async def run_api_inference(
    prompts: List[str],
    model: str,
    api_url: str,
    gen_config: Dict[str, Any] = None,
    api_key: str = "EMPTY",
    request_rate: float = float("inf"),
    burstiness: float = 1.0,
    max_concurrency: int = None,
    request_timeout: float = 600.0,
    disable_tqdm: bool = False,
    stream: bool = True,
) -> List[Dict[str, Any]]:
    """Async inference against an OpenAI-compatible server."""
    if gen_config is None:
        gen_config = {}

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    api_params = {
        "model": model,
        "max_tokens": gen_config.get("max_new_tokens", 4096),
        "temperature": gen_config.get("temperature", 0.0),
        "top_p": gen_config.get("top_p", 1.0),
    }

    n = len(prompts)
    results: List[Any] = [None] * n
    pbar = None if disable_tqdm else tqdm(total=n)
    effective_concurrency = max_concurrency if max_concurrency else min(n, 16)
    semaphore = asyncio.Semaphore(effective_concurrency)

    async def _send_request(idx: int, prompt: str, session: aiohttp.ClientSession):
        payload = {"messages": [{"role": "user", "content": prompt}], **api_params}
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        async with semaphore:
            try:
                st = time.perf_counter()
                async with session.post(api_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    if stream:
                        rec = await _parse_stream(response, st)
                        rec["prompt"] = prompt
                        rec["success"] = bool(rec["generated_text"])
                        results[idx] = rec
                    else:
                        data = await response.json(content_type=None)
                        if "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            content = choice.get("message", {}).get("content", "")
                            usage = data.get("usage") or {}
                            results[idx] = {"prompt": prompt, "generated_text": content,
                                            "success": True,
                                            "finish_reason": choice.get("finish_reason"),
                                            "completion_tokens": usage.get("completion_tokens"),
                                            "gen_chars": len(content),
                                            "ttft": None, "itl": [], "latency": None,
                                            "output_tokens": usage.get("completion_tokens")}
                        else:
                            results[idx] = {"prompt": prompt, "generated_text": "", "success": False}
            except Exception as e:
                print(f"Request failed: {e}")
                results[idx] = {"prompt": prompt, "generated_text": "", "success": False}
            finally:
                if pbar is not None:
                    pbar.update(1)

    timeout = aiohttp.ClientTimeout(total=request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks: List[asyncio.Task] = []
        async for idx, prompt in get_request(prompts, request_rate, burstiness):
            tasks.append(asyncio.create_task(_send_request(idx, prompt, session)))
        await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()
    return results


def load_dataset_simple(dataset_path: str, split: str, input_key: str, output_key: str,
                        num_samples: int, model: str = None, min_tokens: int = 0):
    """Load dataset from Hugging Face or local JSONL file."""
    from datasets import load_dataset

    if dataset_path.endswith(".jsonl") or os.path.isfile(dataset_path):
        dataset = load_dataset("json", data_files=dataset_path, split=split)
    else:
        dataset = load_dataset(dataset_path, split=split)

    tokenizer = None
    if min_tokens > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

    prompts = []
    references = {}
    n_skipped = 0

    for item in dataset:
        if len(prompts) >= num_samples:
            break
        prompt = item[input_key]
        reference = item[output_key]

        if tokenizer is not None:
            n_tok = len(tokenizer(prompt, add_special_tokens=False).input_ids)
            if n_tok < min_tokens:
                n_skipped += 1
                continue

        references[str(len(prompts))] = reference
        prompts.append(prompt)

    if tokenizer is not None:
        print(f"  token filter: kept {len(prompts)} samples with >= {min_tokens} input "
              f"tokens (skipped {n_skipped} shorter ones)")

    return prompts, references


# ==========================================
# Main Async Driver
# ==========================================

async def main():
    parser = argparse.ArgumentParser(description="Run LLM Inference & Code Benchmarks")
    parser.add_argument("--model", type=str, default="NousResearch/Hermes-3-Llama-3.1-8B")
    parser.add_argument("--dataset-path", type=str, default="codeparrot_f1_benchmark.jsonl")
    parser.add_argument("--hf-split", type=str, default="train")
    parser.add_argument("--input-key", type=str, default="query")
    parser.add_argument("--output-key", type=str, default="reference")
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    
    # Evaluation Toggles
    parser.add_argument("--compute-f1", action="store_true", help="Compute token-level F1 score")
    parser.add_argument("--compute-code-metrics", action="store_true", help="Compute AST Syntax validity, Normalized EM, and CodeBLEU")
    parser.add_argument("--lang", type=str, default="python", help="Language for CodeBLEU and syntax checks")

    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--burstiness", type=float, default=1.0)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--disable-tqdm", action="store_true")
    parser.add_argument("--no-stream", dest="stream", action="store_false")
    parser.add_argument("--max-tokens", type=int, default=6000)
    parser.add_argument("--result-file", type=str, default=None)
    parser.add_argument("--label", type=str, default="")
    parser.set_defaults(stream=True)

    args = parser.parse_args()

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    prompts, references = load_dataset_simple(
        args.dataset_path, args.hf_split, args.input_key, args.output_key, args.num_prompts,
        model=args.model, min_tokens=args.min_tokens,
    )

    # Run inference
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
    gen_config = {"max_new_tokens": args.max_tokens}

    print("\nStarting inference...")
    print(f"  request_rate={args.request_rate}  burstiness={args.burstiness}  "
          f"max_concurrency={args.max_concurrency}  num_prompts={len(prompts)}  "
          f"stream={args.stream}")
    start_time = time.perf_counter()
    outputs = await run_api_inference(
        prompts, args.model, api_url, gen_config=gen_config,
        request_rate=args.request_rate, burstiness=args.burstiness,
        max_concurrency=args.max_concurrency, request_timeout=args.request_timeout,
        disable_tqdm=args.disable_tqdm, stream=args.stream,
    )
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    n_ok = sum(1 for o in outputs if o["success"])
    print(f"Inference completed in {elapsed:.2f} s  |  {n_ok}/{len(outputs)} ok  |  "
          f"achieved {len(outputs)/elapsed:.2f} req/s")

    ok_outs = [o for o in outputs if o.get("success")]
    n_len = sum(1 for o in ok_outs if o.get("finish_reason") == "length")
    n_stop = sum(1 for o in ok_outs if o.get("finish_reason") == "stop")
    length_pct = (100.0 * n_len / len(ok_outs)) if ok_outs else None
    if ok_outs:
        ctoks = [o["completion_tokens"] for o in ok_outs if o.get("completion_tokens") is not None]
        chars = [o.get("gen_chars", 0) for o in ok_outs]
        tok_str = (f"completion_tokens mean={np.mean(ctoks):.0f} median={np.median(ctoks):.0f} "
                   f"max={max(ctoks)}" if ctoks else "completion_tokens n/a")
        print(f"  output: {tok_str} | chars mean={np.mean(chars):.0f} median={np.median(chars):.0f}")
        print(f"  finish_reason: length(=max_tokens)={n_len}/{len(ok_outs)} "
              f"({length_pct:.1f}%)  stop(=EOS)={n_stop}/{len(ok_outs)}")

    def _stats_ms(xs):
        if not xs:
            return None
        a = np.asarray(xs, dtype=float) * 1000.0
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p99": float(np.percentile(a, 99))}

    streamed = [o for o in ok_outs if o.get("ttft") is not None]
    metrics: Dict[str, Any] = {}
    if streamed:
        ttfts = [o["ttft"] for o in streamed]
        e2es = [o["latency"] for o in streamed if o.get("latency") is not None]
        itls = [x for o in streamed for x in (o.get("itl") or [])]
        tpots = [(o["latency"] - o["ttft"]) / (o["output_tokens"] - 1)
                 for o in streamed
                 if o.get("output_tokens") and o["output_tokens"] > 1 and o.get("latency")]
        total_out = sum((o.get("output_tokens") or 0) for o in streamed)
        metrics = {
            "ttft_ms": _stats_ms(ttfts),
            "tpot_ms": _stats_ms(tpots),
            "itl_ms": _stats_ms(itls),
            "e2e_latency_ms": _stats_ms(e2es),
            "total_output_tokens": total_out,
            "output_throughput_toks_s": (total_out / elapsed) if elapsed else None,
        }
        _fmt = lambda d: (f"mean={d['mean']:.1f} median={d['median']:.1f} p99={d['p99']:.1f}"
                          if d else "n/a")
        print(f"  latency ms: TTFT[{_fmt(metrics['ttft_ms'])}] "
              f"TPOT[{_fmt(metrics['tpot_ms'])}] ITL[{_fmt(metrics['itl_ms'])}] "
              f"E2E[{_fmt(metrics['e2e_latency_ms'])}]")
        if metrics["output_throughput_toks_s"]:
            print(f"  throughput: {metrics['output_throughput_toks_s']:.1f} output tok/s "
                  f"({total_out} toks) | {len(outputs)/elapsed:.2f} req/s")

    if outputs:
        print("\nSample Output [0]:")
        print(f"Prompt: {outputs[0].get('prompt', '')[:200]}...")
        print(f"Generated Text: {outputs[0].get('generated_text', '')[:200]}...")

    # ==========================================
    # Evaluation Block
    # ==========================================
    eval_results: Dict[str, Any] = {}

    if args.compute_f1 or args.compute_code_metrics:
        sample_ids = list(references.keys())
        eval_preds = []
        eval_refs = []
        f1_scores = []
        ast_valid_scores = []
        em_scores = []

        for output, sample_id in zip(outputs, sample_ids):
            if output.get('success') and output.get('generated_text') is not None:
                gt = references[sample_id]
                pred = output['generated_text']
                if gt:
                    eval_preds.append(pred)
                    eval_refs.append(gt)
                    
                    if args.compute_f1:
                        f1_scores.append(f1_score(pred, gt))

                    if args.compute_code_metrics:
                        ast_valid_scores.append(1.0 if check_syntax_validity(output.get("prompt", ""), pred) else 0.0)
                        em_scores.append(1.0 if normalized_exact_match(pred, gt) else 0.0)

        print("\n=== Evaluation Results ===")
        if args.compute_f1 and f1_scores:
            mean_f1 = float(np.mean(f1_scores))
            eval_results["mean_f1"] = mean_f1
            eval_results["per_sample_f1"] = f1_scores
            print(f"  Mean F1 Score: {mean_f1:.4f} (over {len(f1_scores)} samples)")

        if args.compute_code_metrics and eval_preds:
            mean_ast = float(np.mean(ast_valid_scores)) * 100
            mean_em = float(np.mean(em_scores)) * 100
            eval_results["ast_syntax_validity_pct"] = mean_ast
            eval_results["normalized_exact_match_pct"] = mean_em
            print(f"  AST Syntax Validity: {mean_ast:.2f}%")
            print(f"  Normalized Exact Match: {mean_em:.2f}%")

            codebleu_res = compute_codebleu_safe(eval_preds, eval_refs, lang=args.lang)
            if codebleu_res:
                eval_results["codebleu"] = codebleu_res
                print(f"  CodeBLEU Score: {codebleu_res['codebleu'] * 100:.2f}")
                print(f"    - N-gram Match: {codebleu_res['ngram_match'] * 100:.2f}")
                print(f"    - Weighted N-gram: {codebleu_res['weighted_ngram_match'] * 100:.2f}")
                print(f"    - Syntax Match: {codebleu_res['syntax_match'] * 100:.2f}")
                print(f"    - Dataflow Match: {codebleu_res['dataflow_match'] * 100:.2f}")
            else:
                print("  CodeBLEU: Skipped (install via `pip install codebleu` to enable)")

    # Save output summary
    summary = {
        "label": args.label,
        "config": {
            "model": args.model, "dataset_path": args.dataset_path,
            "num_prompts": len(prompts), "max_concurrency": args.max_concurrency,
            "request_rate": (args.request_rate if np.isfinite(args.request_rate) else "inf"),
            "burstiness": args.burstiness,
            "max_tokens": args.max_tokens, "stream": args.stream,
        },
        "completed": n_ok,
        "elapsed_s": elapsed,
        "request_throughput_rps": (len(outputs) / elapsed) if elapsed else None,
        "finish_length_pct": length_pct,
        "evaluation": eval_results,
        **metrics,
    }
    result_file = args.result_file or os.path.join(args.result_dir, "benchmark_results.json")
    os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
    with open(result_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {result_file}")


if __name__ == "__main__":
    asyncio.run(main())