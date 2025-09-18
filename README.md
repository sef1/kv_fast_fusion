# KV Fast Fusion: Installation & Quickstart Guide

This guide walks you through installing and running **Joint Encoding of KV Blocks**, a blocks-sharing optimization for vLLM that enables efficient key-value cache compression using Batch Fast Fusion (BFF) and Chunks Fast Fusion (CFF).

---

## 📦 Requirements

- Python **3.12**
- Linux environment with GPU support
- Tested on vLLM commit: `e4ca6e3a99816920df80a1e0a72cd3658d9d134b`

---

## ⚙️ Step 1: Install Dependencies

```bash
pip install pandas datasets
```

---

## 🚀 Step 2: Clone and Patch vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout e4ca6e3a99816920df80a1e0a72cd3658d9d134b

# Install vLLM with precompiled components
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Download and apply KV Fast Fusion patch
wget https://anonymous.4open.science/api/repo/kv_fast_fusion-7A7D/file/KV_fast_fusion.patch
git apply KV_fast_fusion.patch
```

---

## 🧪 Step 3: Set Up Environment Variables

Set the number of parallel requests to control fusion behavior:

- For **Batch Fast Fusion (BFF)**:
  ```bash
  export MAX_NUM_SEQ=256
  ```

- For **Chunk Fast Fusion (CFF)**:
  ```bash
  export MAX_NUM_SEQ=1
  ```

---

## 🖥️ Step 4: Launch vLLM API Server

In one terminal:

```bash
VLLM_USE_V1=0 \
python -I vllm/entrypoints/api_server.py \
  --host 127.0.0.1 \
  --port 10001 \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --log-level info \
  --enforce-eager \
  --max_num_seqs $MAX_NUM_SEQ \
  --block-size 16 \
  --max_num_batched_tokens 512 \
  --thr 0.7 \
  --num_chunks_to_compress 32
```

---

## 📊 Step 5: Run Benchmark Client

In another terminal:

```bash
python benchmarks/benchmark_serving.py \
  --host 127.0.0.1 \
  --port 10001 \
  --model NousResearch/Hermes-3-Llama-3.1-8B \
  --backend deepspeed-mii \
  --dataset-name random \
  --num-prompts 256 \
  --random-input-len 1024 \
  --random-output-len 1024 \
  --request-rate 100 \
  --random-range-ratio 0.5
```

---

## 📌 Notes

- Adjust `--thr` to control the **similarity threshold** for block fusion.
- The benchmark script uses random input; replace it with real datasets to measure end-to-end accuracy or F1 score.
- Fusion results are logged via the vLLM server; monitor the logs for compression statistics.
