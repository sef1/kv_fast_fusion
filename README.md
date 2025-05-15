# KV fast fusion
kv cache fast fusion installation guide and running example. \
Install in an environment with python3.12. 

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
```shell
export MAX_NUM_SEQ=1
```
**for BFF**
```shell
export MAX_NUM_SEQ=256
```
**In one terminal - Start vLLM api server with:**\
```shell
VLLM_USE_V1=0 python -I vllm/entrypoints/api_server.py --host 127.0.0.1 --model NousResearch/Hermes-3-Llama-3.1-8B --port 10001 --log-level info --enforce-eager --max_num_seqs $MAX_NUM_SEQ --block-size 16 --max_num_batched_tokens 512 --thr 0.7 --num_chunks_to_compress 32
  ```
**In another terminal - Run vLLM random benchmark:**\
```shell
python benchmarks/benchmark_serving.py --host 127.0.0.1 --port 10001 --model NousResearch/Hermes-3-Llama-3.1-8B --backend deepspeed-mii --dataset-name random --num-prompts 256 --random-input-len 1024 --random-output-len 1024 --request-rate 100 --random-range-ratio 0.5
```
