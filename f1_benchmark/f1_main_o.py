import argparse
import asyncio
import json
import os
import time
import aiohttp
import numpy as np
from collections import Counter
from tqdm import tqdm
from typing import AsyncGenerator, List, Dict, Any, Tuple
  
# standalone evaluation 
  
def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
  
async def get_request(
    prompts: List[str],
    request_rate: float,
    burstiness: float = 1.0,
) -> AsyncGenerator[Tuple[int, str], None]:
    """Asynchronously yield (index, prompt) paced by the request rate, mirroring
    `vllm.benchmarks.serve.get_request`: inter-arrival times follow a gamma distribution
    (shape=`burstiness`, scale=1/(rate·burstiness)) — `burstiness`==1 is a Poisson
    process, <1 burstier, >1 more uniform. `request_rate`==inf fires everything at once."""
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
    """Parse an OpenAI chat-completions SSE stream, timestamping tokens for latency metrics.

    Mirrors `vllm.benchmarks.serve`: TTFT = first-token arrival − send time; ITL = gaps
    between successive token chunks; output_tokens from the `include_usage` final chunk
    (falls back to the chunk count). Returns the per-request record."""
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
        if usage:  # the include_usage chunk carries the authoritative token count
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
    """Async inference against an OpenAI-compatible server (same arrival/concurrency model
    as `vllm bench serve`).

    Requests are *dispatched* at `request_rate` req/s with gamma-distributed inter-arrival
    times; an `asyncio.Semaphore(max_concurrency)` caps in-flight requests to the server so
    fusion actually sees many concurrent prefills. A tqdm bar advances as each request
    completes. Result order matches `prompts` (F1 pairing stays correct).

    When `stream` (default), requests use SSE streaming so per-request TTFT / ITL / TPOT are
    captured (see `_parse_stream`); `--no-stream` falls back to a single non-streamed JSON
    response (no latency breakdown, only F1 + throughput)."""
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
    # Cap in-flight requests; default to the pre-async behavior (min(n, 16)) when unset.
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
                        # The disagg proxy streams the model JSON via make_response(generator),
                        # which Quart labels text/html — so aiohttp's default .json() rejects the
                        # mimetype even though the body is valid JSON. content_type=None skips the
                        # check. Harmless against a direct vLLM server (application/json).
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
    connector = aiohttp.TCPConnector(limit=0)  # don't let the client throttle below max_concurrency
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
    """Load dataset without vLLM dependencies.

    When `min_tokens > 0`, each prompt is tokenized with the model's HF tokenizer and
    samples shorter than `min_tokens` input tokens are skipped; we keep scanning the
    split until `num_samples` qualifying samples are collected (or it is exhausted).
    `references` keys stay positionally aligned with `prompts` so F1 pairing is correct.
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_path, split=split)

    tokenizer = None
    if min_tokens > 0:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)

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
  
async def main():
    parser = argparse.ArgumentParser(description="Run F1 Score Benchmark")
    parser.add_argument("--model", type=str, default="NousResearch/Hermes-3-Llama-3.1-8B")  
    parser.add_argument("--dataset-path", type=str, default="nvidia/OpenMathInstruct-2")  
    parser.add_argument("--hf-split", type=str, default="train")  
    parser.add_argument("--input-key", type=str, default="problem")  
    parser.add_argument("--output-key", type=str, default="generated_solution")  
    parser.add_argument("--num-prompts", type=int, default=30)  
    parser.add_argument("--host", type=str, default="localhost")  
    parser.add_argument("--port", type=int, default=8000)  
    parser.add_argument("--compute-f1", action="store_true")
    parser.add_argument("--result-dir", type=str, default="./results")
    parser.add_argument("--request-rate", type=float, default=float("inf"),
                        help="Arrival rate (req/s). 'inf' fires all at once.")
    parser.add_argument("--burstiness", type=float, default=1.0,
                        help="Arrival burstiness (gamma shape): 1=Poisson, <1 burstier, >1 uniform.")
    parser.add_argument("--max-concurrency", type=int, default=None,
                        help="Cap on in-flight requests (drives concurrent prefills for fusion).")
    parser.add_argument("--request-timeout", type=float, default=600.0,
                        help="Per-request HTTP timeout (s); raise it under heavy load.")
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="Skip samples whose prompt has fewer than this many input "
                             "tokens (0 = no filter). Uses the model's HF tokenizer.")
    parser.add_argument("--disable-tqdm", action="store_true",
                        help="Disable the tqdm progress bar.")
    parser.add_argument("--no-stream", dest="stream", action="store_false",
                        help="Use a single non-streamed response (disables TTFT/ITL/TPOT).")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="Max generated tokens per request (maps to OpenAI max_tokens).")
    parser.add_argument("--result-file", type=str, default=None,
                        help="Path for the summary JSON (default <result-dir>/f1_results.json). "
                             "Use a config-tagged name to compare runs.")
    parser.add_argument("--label", type=str, default="",
                        help="Free-form run label stored in the summary (e.g. the BFF config).")
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

    # Output-length / termination diagnostics — the decisive check for whether a config
    # (e.g. fusion) generates longer / non-terminating outputs vs baseline. If many requests
    # have finish_reason="length", they hit max_tokens (no EOS) → that, not the script,
    # explains a low req/s at equal token-throughput.
    ok_outs = [o for o in outputs if o.get("success")]
    n_len = sum(1 for o in ok_outs if o.get("finish_reason") == "length")
    n_stop = sum(1 for o in ok_outs if o.get("finish_reason") == "stop")
    length_pct = (100.0 * n_len / len(ok_outs)) if ok_outs else None
    if ok_outs:
        ctoks = [o["completion_tokens"] for o in ok_outs if o.get("completion_tokens") is not None]
        chars = [o.get("gen_chars", 0) for o in ok_outs]
        tok_str = (f"completion_tokens mean={np.mean(ctoks):.0f} median={np.median(ctoks):.0f} "
                   f"max={max(ctoks)}" if ctoks else "completion_tokens n/a (no usage in response)")
        print(f"  output: {tok_str} | chars mean={np.mean(chars):.0f} median={np.median(chars):.0f}")
        print(f"  finish_reason: length(=max_tokens)={n_len}/{len(ok_outs)} "
              f"({length_pct:.1f}%)  stop(=EOS)={n_stop}/{len(ok_outs)}")

    # Latency metrics (streaming only): TTFT / TPOT / ITL + output throughput, mirroring
    # `vllm bench serve`. TPOT = (e2e − TTFT) / (output_tokens − 1).
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

    # Print sample output
    if outputs:
        print("\nSample Output [0]:")
        print(f"Prompt: {outputs[0].get('prompt', '')}")
        print(f"Generated Text: {outputs[0].get('generated_text', '')}")

    # Compute F1 scores
    mean_f1 = None
    f1_scores: List[float] = []
    if args.compute_f1:
        sample_ids = list(references.keys())
        for output, sample_id in zip(outputs, sample_ids):
            if output.get('success') and output.get('generated_text'):
                ground_truth = references[sample_id]
                if ground_truth:
                    f1_scores.append(f1_score(output['generated_text'], ground_truth))

        n_excluded = len(outputs) - len(f1_scores)
        if n_excluded:
            print(f"  note: {n_excluded}/{len(outputs)} samples excluded from F1 "
                  f"(failed request or empty output/reference) — mean is over the rest")
        if f1_scores:
            mean_f1 = float(np.mean(f1_scores))
            print(f"\nMean F1 score: {mean_f1:.4f}  (over {len(f1_scores)} samples)")

    # Always save a summary: accuracy + throughput + end-to-end time + latency, tagged by
    # --label so a sweep (cc/nr_tree × THRESHOLD × BFF_GROUP_SIZE) is easy to tabulate.
    summary = {
        "label": args.label,
        "config": {
            "model": args.model, "dataset_path": args.dataset_path,
            "num_prompts": len(prompts), "max_concurrency": args.max_concurrency,
            # store "inf" (not float inf → invalid JSON Infinity) when firing all at once
            "request_rate": (args.request_rate if np.isfinite(args.request_rate) else "inf"),
            "burstiness": args.burstiness,
            "max_tokens": args.max_tokens, "stream": args.stream,
        },
        "completed": n_ok,
        "elapsed_s": elapsed,
        "request_throughput_rps": (len(outputs) / elapsed) if elapsed else None,
        "finish_length_pct": length_pct,
        "mean_f1": mean_f1,
        "num_f1_samples": len(f1_scores),
        **metrics,
    }
    result_file = args.result_file or os.path.join(args.result_dir, "f1_results.json")
    os.makedirs(os.path.dirname(result_file) or ".", exist_ok=True)
    with open(result_file, "w") as f:
        json.dump({**summary, "per_sample_f1": f1_scores}, f, indent=2)
    print(f"\nSummary saved to {result_file}")

if __name__ == "__main__":
    asyncio.run(main())