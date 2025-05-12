# KV fast fusion
kv cache fast fusion installation guide and running example. \
Install in an environment with python3.12. \

**Download vLLM and apply patch**
```shell
pip install pandas datasets
git clone https://github.com/vllm-project/vllm.git 
cd vllm 
git checkout e4ca6e3a99816920df80a1e0a72cd3658d9d134b 
VLLM_USE_PRECOMPILED=1 pip install --editable .
wget https://anonymous.4open.science/api/repo/kv_fast_fusion-7A7D/file/KV_fast_fusion.patch 
git apply KV_fast_fusion.patch
```
**For CFF**
export BATCH_SIZE=1

**for BFF**
export BATCH_SIZE=256
**Start api server with:**\
VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_V1=0 python -I vllm/entrypoints/api_server.py --host 127.0.0.1 --model NousResearch/Hermes-3-Llama-3.1-8B --port 10000 --log-level info --enforce-eager --max_num_seqs 1 --block-size 16 --max_num_batched_tokens 512
  
**Run random benchmark:**\
python benchmarks/benchmark_serving.py --host 127.0.0.1 --port 10000 --model  NousResearch/Hermes-3-Llama-3.1-8B --backend deepspeed-mii --dataset-name random --num-prompts 256 --random-input-len 32768 --random-output-len 256 --request-rate 100 --random-range-ratio 0.5
