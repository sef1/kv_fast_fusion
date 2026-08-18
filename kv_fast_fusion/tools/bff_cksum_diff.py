#!/usr/bin/env python3
"""Join BFF CKSUM send/recv lines by tid → settle KV corruption vs saturation.

Usage:
    python bff_cksum_diff.py <prefill*.log ...> <decode*.log ...>
(order doesn't matter; 'send' lines come from producers, 'recv' from the decode.)

Verdict:
  - h-mismatch  > 0  → real byte corruption on those (request, layer) tids.
  - shape/nblk-mismatch > 0 → truncation (fewer blocks arrived than the block table).
  - unpaired (send w/o recv, or recv w/o send) → transfer set mismatch (the deadlock class).
  - all matched, 0 mismatch → KV transferred perfectly ⇒ NOT corruption ⇒ saturation.
"""
import re, sys

# BFF CKSUM send | <tid> | h=<hex> | shape=(...) | nblk=<n>
LINE = re.compile(r"BFF CKSUM (send|recv) \| (\S+) \| h=([0-9a-f]+) \| shape=(\([^)]*\)) \| nblk=(\d+)")

sends, recvs = {}, {}
for fp in sys.argv[1:]:
    try:
        for ln in open(fp, errors="replace"):
            m = LINE.search(ln)
            if not m:
                continue
            side, tid, h, shape, nblk = m.groups()
            (sends if side == "send" else recvs)[tid] = (h, shape, int(nblk))
    except OSError as e:
        print(f"  (skip {fp}: {e})")

all_tids = set(sends) | set(recvs)
h_mismatch = shape_mismatch = matched = 0
send_only = recv_only = 0
examples = []
for tid in all_tids:
    s, r = sends.get(tid), recvs.get(tid)
    if s and not r:
        send_only += 1; continue
    if r and not s:
        recv_only += 1; continue
    if s[0] != r[0]:
        h_mismatch += 1
        if len(examples) < 5:
            examples.append(f"  H-MISMATCH {tid}\n    send h={s[0]} shape={s[1]} nblk={s[2]}\n    recv h={r[0]} shape={r[1]} nblk={r[2]}")
    elif s[1] != r[1] or s[2] != r[2]:
        shape_mismatch += 1
        if len(examples) < 5:
            examples.append(f"  SHAPE/NBLK-MISMATCH {tid}: send {s[1]}/{s[2]} vs recv {r[1]}/{r[2]}")
    else:
        matched += 1

print(f"tids: send={len(sends)} recv={len(recvs)} | matched(h ok)={matched} "
      f"h-mismatch={h_mismatch} shape-mismatch={shape_mismatch} "
      f"send-only={send_only} recv-only={recv_only}")
for e in examples:
    print(e)
if h_mismatch == 0 and shape_mismatch == 0 and send_only == 0 and recv_only == 0 and matched > 0:
    print("VERDICT: KV transferred byte-perfect → NOT corruption → the con=400 degradation is SATURATION.")
elif h_mismatch or shape_mismatch:
    print("VERDICT: REAL CORRUPTION — localized above; audit the producer gather / consumer inject / chunked accumulation for those tids.")
elif send_only or recv_only:
    print("VERDICT: transfer SET mismatch (send/recv unpaired) — producer/consumer build_connector_meta disagree.")
else:
    print("VERDICT: no CKSUM lines found — was BFF_PD_CKSUM=1 set on all engines?")
