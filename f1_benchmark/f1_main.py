import argparse
import asyncio
import contextlib
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
    """Calculate F1 score between prediction and ground truth."""  
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
) -> List[Dict[str, Any]]:
    """Async inference against an OpenAI-compatible server (same arrival/concurrency model
    as `vllm bench serve`).

    Requests are *dispatched* at `request_rate` req/s with gamma-distributed inter-arrival
    times; an `asyncio.Semaphore(max_concurrency)` caps in-flight requests to the server so
    fusion actually sees many concurrent prefills. A tqdm bar advances as each request
    completes. Result order matches `prompts` (F1 pairing stays correct)."""
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
    semaphore = (asyncio.Semaphore(max_concurrency)
                 if max_concurrency else contextlib.nullcontext())

    async def _send_request(idx: int, prompt: str, session: aiohttp.ClientSession):
        payload = {"messages": [{"role": "user", "content": prompt}], **api_params}
        async with semaphore:
            try:
                async with session.post(api_url, headers=headers, json=payload) as response:
                    response.raise_for_status()
                    data = await response.json()
                if "choices" in data and data["choices"]:
                    content = data["choices"][0].get("message", {}).get("content", "")
                    results[idx] = {"prompt": prompt, "generated_text": content,
                                    "success": True}
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

    args = parser.parse_args()
      
    # Load dataset   
    print(f"Loading dataset from {args.dataset_path}...")  
    prompts, references = load_dataset_simple(
        args.dataset_path, args.hf_split, args.input_key, args.output_key, args.num_prompts,
        model=args.model, min_tokens=args.min_tokens,
    )
      
    # Run inference  
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"  
    gen_config = {} # or: {"max_new_tokens": 512, "temperature": 0.0}  
      
    print("\nStarting inference...")
    print(f"  request_rate={args.request_rate}  burstiness={args.burstiness}  "
          f"max_concurrency={args.max_concurrency}  num_prompts={len(prompts)}")
    start_time = time.perf_counter()
    outputs = await run_api_inference(
        prompts, args.model, api_url, gen_config=gen_config,
        request_rate=args.request_rate, burstiness=args.burstiness,
        max_concurrency=args.max_concurrency, request_timeout=args.request_timeout,
        disable_tqdm=args.disable_tqdm,
    )
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    n_ok = sum(1 for o in outputs if o["success"])
    print(f"Inference completed in {elapsed:.2f} s  |  {n_ok}/{len(outputs)} ok  |  "
          f"achieved {len(outputs)/elapsed:.2f} req/s")
      
    # Print sample output  
    if outputs:  
        print("\nSample Output [0]:")  
        print(f"Prompt: {outputs[0]['prompt']}")  
        print(f"Generated Text: {outputs[0]['generated_text']}")  
      
    # Compute F1 scores  
    if args.compute_f1:  
        f1_scores = []  
        sample_ids = list(references.keys())  
          
        for output, sample_id in zip(outputs, sample_ids):  
            if output['success'] and output['generated_text']:  
                generated_text = output['generated_text']  
                ground_truth = references[sample_id]  
                if ground_truth:  
                    score = f1_score(generated_text, ground_truth)  
                    f1_scores.append(score)  
          
        if f1_scores:  
            mean_f1 = np.mean(f1_scores)  
            print(f"\nMean F1 score: {mean_f1:.4f}")  
              
            # Save results  
            if not os.path.exists(args.result_dir):  
                os.makedirs(args.result_dir)  
              
            result_file = os.path.join(args.result_dir, "f1_results.json")  
            results = {  
                "mean_f1_score": mean_f1,  
                "num_samples": len(f1_scores),  
                "per_sample_f1": f1_scores,  
            }  
              
            with open(result_file, "w") as f:  
                json.dump(results, f, indent=2)  
              
            print(f"Results saved to {result_file}")  
  
if __name__ == "__main__":
    asyncio.run(main())