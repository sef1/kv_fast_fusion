# KV fast fusion\
kv cache fast fusion example

**Download vLLM and apply patch**\
git clone https://github.com/vllm-project/vllm.git \
cd vllm \
git checkout e4ca6e3a99816920df80a1e0a72cd3658d9d134b \
git apply /path/to/KV_fast_fusion.patch

**Start api server with:**\
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0 python path/to/vllm/entrypoints/api_server.py --host=<server-ip> --model NousResearch/Hermes-3-Llama-3.1-8B --port=<port> --log-level info --enforce-eager --max_num_seqs <1 for cff 256 for bff> --block-size 16 --max_num_batched_tokens 512
  
**Run random benchmark:**\
python /path/to/benchmarks/benchmark_serving.py --host <server-ip> --port <port> --model  NousResearch/Hermes-3-Llama-3.1-8B --backend deepspeed-mii --dataset-name random --num-prompts 256 --random-input-len 32768 --random-output-len 256 --request-rate 100 --random-range-ratio 0.5
