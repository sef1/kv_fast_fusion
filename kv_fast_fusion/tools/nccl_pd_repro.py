"""Minimal cross-process NCCL send/recv repro of the vLLM P2P engine's data path.

Two processes, one GPU each, exactly the engine's calls: NCCLLibrary wrapper,
ncclCommInitRank(2, uid, rank), ncclSend/ncclRecv on a dedicated stream, sync.
If THIS hangs, the breakage is NCCL/driver/OS level and no vLLM reinstall can fix it.
If it passes, the fault is inside the vLLM/torch layer above.

Usage: python nccl_pd_repro.py <rank:0|1> <gpu> <uid_file>
rank 0 writes the unique id to uid_file and SENDS; rank 1 reads it and RECVS.
"""
import os
import sys
import time

import torch

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclDataTypeEnum,
)

rank, gpu, uid_file = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
device = torch.device("cuda:0")
torch.cuda.set_device(device)

lib = NCCLLibrary(os.environ.get("VLLM_NCCL_SO_PATH"))
print(f"[rank{rank}] nccl version {lib.ncclGetVersion()}", flush=True)

if rank == 0:
    uid = lib.ncclGetUniqueId()
    with open(uid_file, "wb") as f:
        f.write(bytes(uid.internal))
else:
    while not os.path.exists(uid_file):
        time.sleep(0.1)
    time.sleep(0.2)
    from vllm.distributed.device_communicators.pynccl_wrapper import ncclUniqueId
    uid = ncclUniqueId()
    raw = open(uid_file, "rb").read()
    for i, b in enumerate(raw):
        uid.internal[i] = b

t0 = time.time()
print(f"[rank{rank}] ncclCommInitRank(2, uid, {rank}) ...", flush=True)
comm = lib.ncclCommInitRank(2, uid, rank)
print(f"[rank{rank}] comm init OK in {time.time()-t0:.1f}s", flush=True)

n = 1 << 20  # 1M float16 = 2 MB, roughly a KV layer chunk
stream = torch.cuda.Stream()
if rank == 0:
    tensor = torch.full((n,), 3.25, dtype=torch.float16, device=device)
else:
    tensor = torch.zeros(n, dtype=torch.float16, device=device)

t0 = time.time()
op = "ncclSend" if rank == 0 else "ncclRecv"
print(f"[rank{rank}] {op} {n} elems ...", flush=True)
with torch.cuda.stream(stream):
    fn = lib.ncclSend if rank == 0 else lib.ncclRecv
    fn(
        buffer_type(tensor.data_ptr()),
        n,
        ncclDataTypeEnum.from_torch(tensor.dtype),
        rank ^ 1,
        comm,
        cudaStream_t(stream.cuda_stream),
    )
stream.synchronize()
dt = time.time() - t0
if rank == 1:
    ok = torch.allclose(tensor, torch.full_like(tensor, 3.25))
    print(f"[rank1] RECV DONE in {dt:.2f}s, data correct: {ok}", flush=True)
else:
    print(f"[rank0] SEND DONE in {dt:.2f}s", flush=True)
