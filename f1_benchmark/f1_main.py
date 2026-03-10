import argparse  
import json  
import os  
import time  
import numpy as np  
import requests  
from concurrent.futures import ThreadPoolExecutor  
from collections import Counter  
from typing import List, Dict, Any  
  
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
  
def run_api_inference(  
    prompts: List[str],   
    model: str,   
    api_url: str,   
    api_key: str = "EMPTY",  
    gen_config: Dict[str, Any] = None  
) -> List[Dict[str, Any]]:  
    """Pure Python inference - no vLLM dependencies."""  
    if gen_config is None:  
        gen_config = {}  
      
    headers = {  
        "Content-Type": "application/json",  
        "Authorization": f"Bearer {api_key}",  
    }  
  
    api_params = {  
        "model": model,  
        "max_tokens": gen_config.get("max_new_tokens", 512),  
        "temperature": gen_config.get("temperature", 0.0),  
        "top_p": gen_config.get("top_p", 1.0),  
    }  
  
    def _send_request(prompt):  
        payload = {"messages": [{"role": "user", "content": prompt}], **api_params}  
        try:  
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)  
            response.raise_for_status()  
            data = response.json()  
              
            if "choices" in data and data["choices"]:  
                content = data["choices"][0].get("message", {}).get("content", "")  
                return {  
                    "prompt": prompt,  
                    "generated_text": content,  
                    "success": True  
                }  
            return {  
                "prompt": prompt,  
                "generated_text": "",  
                "success": False  
            }  
        except Exception as e:  
            print(f"Request failed: {e}")  
            return {  
                "prompt": prompt,  
                "generated_text": "",  
                "success": False  
            }  
  
    with ThreadPoolExecutor(max_workers=min(len(prompts), 16)) as executor:  
        results = list(executor.map(_send_request, prompts))  
      
    return results  
  
def load_dataset_simple(dataset_path: str, split: str, input_key: str, output_key: str, num_samples: int):  
    """Load dataset without vLLM dependencies."""  
    from datasets import load_dataset  
      
    dataset = load_dataset(dataset_path, split=split)  
    prompts = []  
    references = {}  
      
    for i, item in enumerate(dataset):  
        if i >= num_samples:  
            break  
        prompt = item[input_key]  
        reference = item[output_key]  
          
        prompts.append(prompt)  
        references[str(i)] = reference  
      
    return prompts, references  
  
def main():  
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
      
    args = parser.parse_args()  
      
    # Load dataset   
    print(f"Loading dataset from {args.dataset_path}...")  
    prompts, references = load_dataset_simple(  
        args.dataset_path, args.hf_split, args.input_key, args.output_key, args.num_prompts  
    )  
      
    # Run inference  
    api_url = f"http://{args.host}:{args.port}/v1/chat/completions"  
    gen_config = {} # or: {"max_new_tokens": 512, "temperature": 0.0}  
      
    print("\nStarting inference...")  
    start_time = time.perf_counter()  
    outputs = run_api_inference(prompts, args.model, api_url, gen_config)  
    end_time = time.perf_counter()  
      
    print(f"Inference completed in {end_time - start_time:.2f} seconds")  
      
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
    main()