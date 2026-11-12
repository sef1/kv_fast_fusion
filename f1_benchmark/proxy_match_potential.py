#!/usr/bin/env python3
"""Estimate BFF's match potential for a prompt set off the box, with a proxy MLA model.

GLM-5.1 cannot run here, so a smaller model with the SAME kind of cache supplies the vectors:
DeepSeek-V2-Lite is MLA with a 512-wide latent (``kv_a_layernorm`` output), which is exactly the
``kv[0]`` BFF signs on GLM-5.1. Everything after the vectors is the production code, unchanged:

* :func:`kv_fast_fusion.pd_dedup_v2.signature_matrix` signs each 128-token block over the middle
  layers (the BFF_EDGE_LAYERS split, scaled to the proxy's depth);
* :class:`kv_fast_fusion.pd_dedup_plan.DedupPlanner` plans every arriving request against the blocks
  the decode would hold, with BFF_LSH_EXACT=1, BFF_THRESHOLD=0.75 and BFF_MAX_REL_ERR=0.5 as on the box.

The replay: requests arrive in file order; the decode holds the last ``--live`` requests. A block a
live or recent prompt already covers through an identical prefix is a prefix-cache hit, so it is not
planned (vLLM shares it in both arms). Kept blocks become resident; aliased ones do not.

It is a proxy: different weights and tokenizer give different vectors. Calibrate it first on the two
Magicoder sets the box has measured (families about 40% at the current bar, the 121 set about 8.5%);
only if it reproduces those is its number for a new set worth acting on.

    # 1. vectors (GPU), one file per prompt set
    python -m f1_benchmark.proxy_match_potential extract --prompts longbench_e_2k4k.jsonl --out lb.pt
    python -m f1_benchmark.proxy_match_potential extract --scenario families --out fam.pt
    # 2. replay (CPU)
    python -m f1_benchmark.proxy_match_potential simulate lb.pt fam.pt --live 8 16 24
"""
import argparse
import os
import sys

# The box's settings. Set before the BFF modules read them at import.
os.environ.setdefault("BFF_LSH_EXACT", "1")
os.environ.setdefault("BFF_MAX_REL_ERR", "0.5")
os.environ.setdefault("BFF_THRESHOLD", "0.75")

BLOCK = 128
PROXY = "deepseek-ai/DeepSeek-V2-Lite-Chat"
GLM_TOKENIZER = "zai-org/GLM-5.1"
# run_best_compare's Magicoder scenarios: (min tokens, max tokens, prompts, families, fingerprint).
SCENARIOS = {"121": (4096, 6000, 121, False, "9f5e27b8"),
             "200": (1000, 2000, 200, False, "4da4b383"),
             "families": (2000, 4096, 219, True, "872781bc")}
GLM_LAYERS, GLM_EDGE = 79, 20   # GLM-5.1 incl. its MTP layer, and BFF_EDGE_LAYERS on the box


def middle_layers(n_layers: int, edge_frac: float = GLM_EDGE / GLM_LAYERS) -> list:
    """The proxy's group-1 layers: the same share cut off each end as BFF_EDGE_LAYERS on GLM-5.1."""
    edge = round(n_layers * edge_frac)
    return list(range(edge, n_layers - edge))


def layout_groups(spec: str, n_proxy: int) -> list:
    """The shared groups of a box layout, as proxy layers.

    ``spec`` is ``both:EDGE:SPLIT`` (today's BFF_EDGE_LAYERS/BFF_EDGE_SPLIT) or ``end:EDGE:SPLIT``
    (the same with BFF_EDGE_AT_END=1). Each shared GLM-5.1 group maps to the proxy layers at the same
    relative depth. Returns ``[(glm_layer_count, [proxy layers])]``: the box signs and frees every
    group on its own, so the replay must too."""
    from kv_fast_fusion.fast_fusion_core import bff_group_layout
    where, edge, split = spec.split(":")
    names = list(range(GLM_LAYERS))
    shared = bff_group_layout(names, 0, 4, int(edge), int(split), edge_at_end=where == "end")[1:]
    out = []
    for glm, _ in shared:
        proxy = sorted({li * n_proxy // GLM_LAYERS for li in glm})
        out.append((len(glm), proxy))
    return out


def load_prompts(args) -> tuple:
    if args.prompts:
        import json
        with open(args.prompts, encoding="utf-8") as f:
            rows = [json.loads(line) for line in f]
        return [r["prompt"] for r in rows], [r.get("task", "") for r in rows]
    from f1_benchmark.f1_main import load_dataset_simple
    lo, hi, n, fam, want = SCENARIOS[args.scenario]
    prompts, _ = load_dataset_simple("ise-uiuc/Magicoder-Evol-Instruct-110K", "train", "instruction",
                                     "response", n, model=args.glm_tokenizer, min_tokens=lo,
                                     max_in_tokens=hi, group_families=fam)
    from f1_benchmark.f1_main import prompt_fingerprint
    got = prompt_fingerprint(prompts)
    if got != want:
        sys.exit(f"Scenario {args.scenario}: fingerprint {got}, the box uses {want}. Not the same prompts.")
    return prompts, [args.scenario] * len(prompts)


def extract(args) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kv_fast_fusion.pd_dedup_v2 import signature_matrix

    prompts, tags = load_prompts(args)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    # transformers' own DeepSeek-V2 code, not the repo's remote code (written for transformers 4).
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    layers = model.model.layers
    # One signature set per BFF_EDGE_LAYERS value (GLM-5.1 counts, scaled to the proxy's depth);
    # the forward pass is shared, so extra settings cost almost nothing.
    by_edge = {e: middle_layers(len(layers), e / GLM_LAYERS) for e in args.edge_layers}
    by_layout = {s: layout_groups(s, len(layers)) for s in (args.layouts or [])}
    mids = sorted({li for m in by_edge.values() for li in m}
                  | {li for groups in by_layout.values() for _, m in groups for li in m})
    # Nothing after the last signed layer is needed: cut the stack there.
    model.model.layers = layers[:mids[-1] + 1]
    for e, m in by_edge.items():
        print(f"proxy {args.model}: {len(layers)} layers; edge {e} (GLM) signs {m[0]}..{m[-1]} ({len(m)})")
    for s, groups in by_layout.items():
        print(f"layout {s}: " + ", ".join(f"{g[0]}..{g[-1]} ({n} GLM layers)" for n, g in groups))

    captured = {}

    def hook(li):
        def fn(_mod, _inp, out):
            captured[li] = out.detach()
        return fn

    # MLA: the latent itself. Otherwise the normalized key before RoPE (Qwen3's k_norm), the nearest
    # thing to a position-free latent; k_proj as the last resort.
    attn = layers[mids[0]].self_attn
    name = next(n for n in ("kv_a_layernorm", "k_norm", "k_proj") if hasattr(attn, n))
    print(f"vectors from self_attn.{name}")
    handles = [getattr(layers[li].self_attn, name).register_forward_hook(hook(li)) for li in mids]
    jl = [None]
    out = {"model": f"{args.model} ({name})", "layers": mids, "block": BLOCK, "tags": tags, "ids": [], "sigs": [],
           "norms": [], "by_edge": {e: {"layers": m, "sigs": [], "norms": []} for e, m in by_edge.items()},
           "n_layers": len(layers),
           "by_layout": {s: [{"glm_layers": n, "layers": m, "sigs": [], "norms": []} for n, m in groups]
                         for s, groups in by_layout.items()}}
    jls = {e: [None] for e in by_edge}
    # One JL cache per signed width: groups of different layer counts must not share one.
    jl_width = {}
    for i, prompt in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        nb = ids.shape[1] // BLOCK
        out["ids"].append(ids[0].tolist())
        if nb == 0:
            out["sigs"].append(None)
            out["norms"].append(None)
            for d in out["by_edge"].values():
                d["sigs"].append(None)
                d["norms"].append(None)
            for groups in out["by_layout"].values():
                for d in groups:
                    d["sigs"].append(None)
                    d["norms"].append(None)
            continue
        captured.clear()
        with torch.no_grad():
            model.model(input_ids=ids.to(model.device), use_cache=False)
        # [1, T, 512] per layer -> [blocks, 128, 512]: the layout signature_matrix reads for MLA.
        # (k_norm gives [1, T, heads, dim]; flatten the heads like a latent.)
        for k, (e, m) in enumerate(by_edge.items()):
            kv = [captured[li][0, :nb * BLOCK].reshape(nb, BLOCK, -1) for li in m]
            sig, norms = signature_matrix(kv, list(range(nb)), True, jls[e])
            out["by_edge"][e]["sigs"].append(sig.float().cpu())
            out["by_edge"][e]["norms"].append(norms.float().cpu())
            if k == 0:          # the first setting doubles as the default set
                out["sigs"].append(sig.float().cpu())
                out["norms"].append(norms.float().cpu())
        for groups in out["by_layout"].values():
            for d in groups:
                kv = [captured[li][0, :nb * BLOCK].reshape(nb, BLOCK, -1) for li in d["layers"]]
                sig, norms = signature_matrix(kv, list(range(nb)), True, jl_width.setdefault(len(kv), [None]))
                d["sigs"].append(sig.float().cpu())
                d["norms"].append(norms.float().cpu())
        if i % 20 == 0:
            print(f"  {i + 1}/{len(prompts)}: {ids.shape[1]} tokens, {nb} blocks", flush=True)
    for h in handles:
        h.remove()
    torch.save(out, args.out)
    print(f"Saved {len(prompts)} prompts to {args.out}")
    return 0


TOKEN_DIM = 64      # per-token projection width for the shift search


def window_matches(xb, sqb, xa, sqa, nb: int, self_partner: bool):
    """For each of B's first ``nb`` 128-token blocks, which offsets in A hold a matching window.

    ``x*`` are per-token projected vectors ``[T, d]`` and ``sq*`` exact per-token squared norms
    ``[T]``. A window's cosine is the sum of per-token dot products over the norms of the whole
    windows — the same quantity the block signature estimates, at every offset instead of only the
    block grid. Accepts as the planner does: cos > 0.75 and rel_err <= 0.5.

    Returns ``(same, aligned, any)``, three bool tensors ``[nb]``: a match at B's own position, at a
    different block boundary of A, at any offset. With ``self_partner`` A is B and only windows
    wholly before the block count."""
    import torch
    la = xa.shape[0]
    if la < BLOCK:
        z = torch.zeros(nb, dtype=torch.bool, device=xb.device)
        return z, z, z
    w = xb[:nb * BLOCK].reshape(nb, BLOCK, -1).permute(0, 2, 1).contiguous()      # [nb, d, 128]
    score = torch.nn.functional.conv1d(xa.t().unsqueeze(0), w)[0]                  # [nb, la-127]
    ca = torch.cat([sqa.new_zeros(1), sqa.cumsum(0)])
    an = (ca[BLOCK:] - ca[:-BLOCK]).clamp(min=1e-12).sqrt()                        # [la-127]
    bn = sqb[:nb * BLOCK].reshape(nb, BLOCK).sum(1).clamp(min=1e-12).sqrt()       # [nb]
    cos = score / (bn[:, None] * an[None, :])
    r = an[None, :] / bn[:, None]
    err = (1 + r * r - 2 * r * cos).clamp(min=0).sqrt()
    ok = (cos > 0.75) & (err <= 0.5)
    off = torch.arange(ok.shape[1], device=ok.device)
    own = torch.arange(nb, device=ok.device) * BLOCK
    if self_partner:
        ok &= off[None, :] + BLOCK <= own[:, None]
    at_own = off[None, :] == own[:, None]
    aligned = (off % BLOCK == 0)[None, :] & ~at_own
    return (ok & at_own).any(1), (ok & aligned).any(1), ok.any(1)


def shift(args) -> int:
    """Same-position, block-shifted and any-offset match potential, from per-token proxy vectors."""
    import collections

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts, _tags = load_prompts(args)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device)
    model.eval()
    layers = model.model.layers
    mids = middle_layers(len(layers))
    model.model.layers = layers[:mids[-1] + 1]
    attn = layers[mids[0]].self_attn
    name = next(n for n in ("kv_a_layernorm", "k_norm", "k_proj") if hasattr(attn, n))
    captured = {}

    def hook(li):
        def fn(_mod, _inp, out):
            captured[li] = out.detach()
        return fn

    handles = [getattr(layers[li].self_attn, name).register_forward_hook(hook(li)) for li in mids]
    proj = None
    live = collections.deque(maxlen=args.live)
    history = collections.deque(maxlen=args.cache)
    n = {"blocks": 0, "prefix": 0, "planned": 0, "same": 0, "aligned": 0, "any": 0}
    for i, prompt in enumerate(prompts):
        text = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
        ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids
        captured.clear()
        with torch.no_grad():
            model.model(input_ids=ids.to(model.device), use_cache=False)
        full = torch.cat([captured[li][0].reshape(ids.shape[1], -1).float() for li in mids], dim=1)
        if proj is None:
            g = torch.Generator(device="cpu").manual_seed(7)
            proj = (torch.randn(full.shape[1], TOKEN_DIM, generator=g) / TOKEN_DIM ** 0.5).to(full.device)
        x, sq = full @ proj, (full * full).sum(1)
        del full
        id_list = ids[0].tolist()
        nb = len(id_list) // BLOCK
        pre = prefix_blocks(id_list, list(history))
        if nb > pre:
            same = torch.zeros(nb, dtype=torch.bool, device=x.device)
            aligned, anyo = same.clone(), same.clone()
            for xa, sqa in list(live) + [(x, sq)]:
                s, a, o = window_matches(x, sq, xa, sqa, nb, self_partner=xa is x)
                same |= s
                aligned |= a
                anyo |= o
            keep = slice(pre, nb)
            n["same"] += int(same[keep].sum())
            n["aligned"] += int((same | aligned)[keep].sum())
            n["any"] += int(anyo[keep].sum())
        n["blocks"] += nb
        n["prefix"] += min(pre, nb)
        n["planned"] += max(0, nb - pre)
        live.append((x, sq))
        history.append(id_list)
        if i % 50 == 0:
            print(f"  {i + 1}/{len(prompts)}", flush=True)
    for h in handles:
        h.remove()
    p = n["planned"] or 1
    label = args.prompts or args.scenario
    print(f"{label}: {len(prompts)} prompts, {n['blocks']} blocks, {n['prefix']} prefix-cached, {n['planned']} "
          f"planned, live {args.live} | matchable: same position {100 * n['same'] / p:.1f}% | "
          f"+ shifted by whole blocks {100 * n['aligned'] / p:.1f}% | + any offset {100 * n['any'] / p:.1f}%")
    return 0


def prefix_blocks(ids: list, others: list) -> int:
    """Full blocks of ``ids`` that an identical block-aligned prefix in ``others`` covers."""
    best = 0
    for o in others:
        n = 0
        while (n + 1) * BLOCK <= min(len(ids), len(o)) and \
                ids[n * BLOCK:(n + 1) * BLOCK] == o[n * BLOCK:(n + 1) * BLOCK]:
            n += 1
        best = max(best, n)
    return best


def simulate_one(data: dict, live: int, cache: int) -> dict:
    """Replay one prompt set through the real planner. Returns block counts."""
    from kv_fast_fusion import pd_lsh
    from kv_fast_fusion.pd_dedup_plan import DedupPlanner, IncomingBlock

    planner = DedupPlanner()
    proj = [None]
    next_id = 0
    resident = []             # per request still live: its kept block ids
    slot_of = {}              # block id -> its block index in its own prompt (= absolute position / 128)
    tot = {"blocks": 0, "prefix": 0, "planned": 0, "dropped": 0, "none": 0, "cells": {},
           "moved": 0}
    for i, (ids, sig, norms) in enumerate(zip(data["ids"], data["sigs"], data["norms"])):
        if len(resident) >= live:
            planner.forget(1, resident.pop(0))
        if sig is None:
            resident.append([])
            continue
        nb = sig.shape[0]
        pre = prefix_blocks(ids, data["ids"][max(0, i - cache):i])
        tot["blocks"] += nb
        tot["prefix"] += pre
        rows = list(range(pre, nb))
        if not rows:
            resident.append([])
            continue
        hashes = pd_lsh.sub_hashes(sig, pd_lsh.get_proj(proj, sig.shape[1], sig.device))
        bids = list(range(next_id, next_id + nb))
        next_id += nb
        slot_of.update((b, s) for s, b in enumerate(bids))
        incoming = [IncomingBlock(f"r{i}", 1, s, bids[s], s) for s in rows]
        plan = planner.plan(1, incoming, sig, hashes, norms.tolist(), 0.75)
        keep = set(plan.keep.get((f"r{i}", 1), []))
        kept = [s for s in rows if s in keep]
        planner.register(1, sig[kept], [hashes[s] for s in kept], norms[kept].tolist(),
                         [bids[s] for s in kept])
        resident.append([bids[s] for s in kept])
        tot["planned"] += len(rows)
        tot["dropped"] += plan.n_dropped()
        # An alias whose representative sits at a different position: its RoPE key is for the
        # wrong place. The planner has no position check, so this counts how often that happens.
        for s, rep in plan.alias.get((f"r{i}", 1), {}).items():
            tot["moved"] += slot_of.get(rep, s) != s
        tot["none"] += plan.potential_none
        for c, n in plan.potential.items():
            tot["cells"][c] = tot["cells"].get(c, 0) + n
    return tot


def within(cells: dict, cos_min: float, err_max: float) -> int:
    """Same reading of the potential grid as run_benchmarks.sh's report."""
    n = 0
    for key, c in cells.items():
        lo, hi = key.split("|")
        if hi != "?" and float(lo) >= cos_min - 1e-9 and float(hi) <= err_max + 1e-9:
            n += c
    return n


def simulate(args) -> int:
    import torch
    bars = ((0.75, 0.5, "current bar"), (0.7, 0.6, ""), (0.6, 0.7, ""), (0.0, float("inf"), "any"))
    for path in args.files:
        data = torch.load(path, weights_only=False)
        variants = {None: data}
        if data.get("by_edge"):
            variants = {e: {"ids": data["ids"], "sigs": d["sigs"], "norms": d["norms"], "layers": d["layers"]}
                        for e, d in data["by_edge"].items()}
        print(f"{path}: {len(data['ids'])} prompts, proxy {data['model']}")
        if data.get("by_layout") and args.layouts_only:
            variants = {}
        for spec, groups in (data.get("by_layout") or {}).items():
            print(f" layout {spec} (each shared group signed and freed on its own, as on the box)")
            for live in args.live:
                parts, blocks, kv = [], 0, 0.0
                for d in groups:
                    t = simulate_one({"ids": data["ids"], "sigs": d["sigs"], "norms": d["norms"]},
                                     live, args.cache)
                    blocks = t["blocks"]
                    kv += t["dropped"] * d["glm_layers"]
                    parts.append(f"proxy {d['layers'][0]}-{d['layers'][-1]}: {t['dropped']} "
                                 f"({100 * t['dropped'] / max(1, t['planned']):.1f}%)")
                print(f"  live {live:2d}: KV not pulled {100 * kv / max(1, blocks * GLM_LAYERS):.2f}% of all | "
                      + " | ".join(parts))
        for e, v in variants.items():
            share = ""
            if e is not None:
                share = (f" | edge {e}: signs proxy layers {v['layers'][0]}..{v['layers'][-1]}; a match frees "
                         f"{(GLM_LAYERS - 2 * e) / GLM_LAYERS:.0%} of a token's KV on GLM-5.1")
            print(f" {share}")
            for live in args.live:
                t = simulate_one(v, live, args.cache)
                p = t["planned"] or 1
                print(f"  live {live:2d}: {t['blocks']} blocks, {t['prefix']} prefix-cached, {t['planned']} planned"
                      f" | not pulled {t['dropped']} ({100 * t['dropped'] / p:.1f}% of planned, "
                      f"{100 * t['dropped'] / max(1, t['blocks']):.1f}% of all), "
                      f"{t['moved']} of them from a different position")
                print("           match potential: " + " | ".join(
                    f"cos>={c:g} err<={e2:g}: {100 * within(t['cells'], c, e2) / p:.1f}%"
                    + (f" [{lbl}]" if lbl else "") for c, e2, lbl in bars))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="run the proxy model and save block signatures")
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--prompts", help="a jsonl with a 'prompt' field, e.g. from make_longbench_e")
    src.add_argument("--scenario", choices=sorted(SCENARIOS), help="a run_best_compare Magicoder set")
    e.add_argument("--model", default=PROXY)
    e.add_argument("--glm-tokenizer", default=GLM_TOKENIZER, help="selects the scenario prompts as the box does")
    e.add_argument("--device", default="cuda:0")
    e.add_argument("--edge-layers", type=int, nargs="+", default=[GLM_EDGE],
                   help="BFF_EDGE_LAYERS values to sign for, in GLM-5.1 layers (scaled to the proxy)")
    e.add_argument("--layouts", nargs="+", default=[],
                   help="box layouts to sign group by group: both:EDGE:SPLIT or end:EDGE:SPLIT "
                        "(BFF_EDGE_AT_END=1), e.g. both:10:3 end:10:3")
    e.add_argument("--out", required=True)
    h = sub.add_parser("shift", help="upper bound with matches at shifted positions allowed (GPU)")
    hsrc = h.add_mutually_exclusive_group(required=True)
    hsrc.add_argument("--prompts")
    hsrc.add_argument("--scenario", choices=sorted(SCENARIOS))
    h.add_argument("--model", default=PROXY)
    h.add_argument("--glm-tokenizer", default=GLM_TOKENIZER)
    h.add_argument("--device", default="cuda:0")
    h.add_argument("--live", type=int, default=16)
    h.add_argument("--cache", type=int, default=64)
    s = sub.add_parser("simulate", help="replay saved signatures through the BFF planner")
    s.add_argument("files", nargs="+")
    s.add_argument("--live", type=int, nargs="+", default=[8, 16, 24],
                   help="requests the decode holds at once (the box's Running)")
    s.add_argument("--cache", type=int, default=64, help="recent prompts the prefix cache still holds")
    s.add_argument("--layouts-only", action="store_true", help="skip the single-group edge settings")
    args = ap.parse_args(argv)
    return {"extract": extract, "simulate": simulate, "shift": shift}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
