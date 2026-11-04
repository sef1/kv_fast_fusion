"""The sweep comparison: what it reads from a saved run, and which run it calls best. CPU only."""

import json

from f1_benchmark import compare_runs as cr

ENGINE = ("INFO Engine 000: Avg prompt throughput: {p} tokens/s, Avg generation throughput: {g} tokens/s, "
          "Running: {r} reqs, Waiting: {w} reqs, GPU KV cache usage: {k}%, Prefix cache hit rate: 0.0%\n")
SERVING = ("Inference completed in {wall} s | 2/2 ok | achieved 0.36 req/s\n"
           "  latency ms: TTFT[mean=78000.0 median=1 p99=1] TPOT[mean=70.0 median=37.5 p99=1]\n"
           "  throughput: {tok} output tok/s (100 toks) | 0.36 req/s\n"
           "  Mean F1 Score: 0.6500 (over 2 samples)\n  AST Syntax Validity: 41.00%\n  CodeBLEU Score: 46.00\n")
CODE = "\n".join(f"def handler_{i}(request, user_{i}):\n    return render(request, {i})" for i in range(60))


def _save(root, name, arm, limit, tok, garbage=False, prompts=("a", "b"), stage="sweep"):
    run = root / name
    (run / "logs_latest").mkdir(parents=True)
    (run / "results_latest").mkdir()
    (run / "DONE").write_text("")
    (run / "config.txt").write_text(f"arm={arm}\ndecode_limit={limit}\nstage={stage}\nMODEL=x\n")
    (run / "logs_latest" / "x-serving.txt").write_text(SERVING.format(wall=300.0, tok=tok))
    (run / "logs_latest" / "prefill-0.txt").write_text(ENGINE.format(p=2000.0, g=0, r=2, w=30, k=40) * 3)
    (run / "logs_latest" / "decode-0.txt").write_text(ENGINE.format(p=0, g=200, r=8, w=1, k=90) * 3)
    bad = "x" * 2000 if garbage else CODE
    (run / "results_latest" / "raw_outputs.json").write_text(json.dumps(
        [{"prompt": p, "generated_text": bad} for p in prompts]))


def test_it_integrates_the_prompt_tokens_the_prefill_processed(tmp_path):
    _save(tmp_path, "r", "baseline", "48", 197.0)
    run = cr.read_run(str(tmp_path / "r"))
    assert run["prefill_k"] == 60.0            # 3 samples x 2000 tok/s x 10 s
    assert run["tok_s"] == 197.0 and run["codebleu"] == 46.0 and run["max_running"] == 8


def test_the_best_run_is_the_fastest_without_garbage(tmp_path):
    _save(tmp_path, "a", "baseline", "8", 190.0)
    _save(tmp_path, "b", "baseline", "9", 199.0)
    _save(tmp_path, "c", "baseline", "48", 210.0, garbage=True)
    _save(tmp_path, "d", "bff", "9", 300.0)
    assert cr.best_of(cr.read_sweep(str(tmp_path)), "baseline")["limit"] == "9"


def test_an_unfinished_run_is_ignored(tmp_path):
    _save(tmp_path, "a", "baseline", "9", 199.0)
    _save(tmp_path, "b", "baseline", "48", 500.0)
    (tmp_path / "b" / "DONE").unlink()
    assert [r["name"] for r in cr.read_sweep(str(tmp_path))] == ["a"]


def test_different_prompts_are_called_out(tmp_path):
    _save(tmp_path, "a", "baseline", "9", 199.0)
    _save(tmp_path, "b", "bff", "9", 201.0, prompts=("a", "other"))
    assert any("NOT use the same prompts" in ln for ln in cr.compare_arms(cr.read_sweep(str(tmp_path)), "9", "9"))


def test_the_comparison_names_who_is_ahead_on_each_measure(tmp_path):
    _save(tmp_path, "a", "baseline", "9", 190.0)
    _save(tmp_path, "b", "bff", "9", 200.0)
    lines = cr.compare_arms(cr.read_sweep(str(tmp_path)), "9", "9")
    assert "BFF ahead" in next(ln for ln in lines if "throughput" in ln)


def test_each_arm_gets_its_own_best_limit(tmp_path):
    _save(tmp_path, "a", "baseline", "8", 199.0)
    _save(tmp_path, "b", "baseline", "9", 190.0)
    _save(tmp_path, "c", "bff", "8", 190.0)
    _save(tmp_path, "d", "bff", "9", 205.0)
    runs = cr.read_sweep(str(tmp_path))
    assert (cr.best_of(runs, "baseline")["limit"], cr.best_of(runs, "bff")["limit"]) == ("8", "9")


def test_a_confirmation_run_cannot_change_the_limit_it_confirms(tmp_path):
    _save(tmp_path, "a", "baseline", "8", 199.0)
    _save(tmp_path, "b", "baseline", "9", 190.0)
    _save(tmp_path, "c", "baseline", "9", 400.0, stage="confirm")
    assert cr.best_of(cr.read_sweep(str(tmp_path)), "baseline")["limit"] == "8"


def test_the_verdict_uses_only_fresh_runs_at_each_arms_limit(tmp_path):
    _save(tmp_path, "a", "baseline", "8", 500.0)
    _save(tmp_path, "b", "bff", "9", 100.0)
    _save(tmp_path, "c", "baseline", "8", 190.0, stage="confirm")
    _save(tmp_path, "d", "bff", "9", 200.0, stage="confirm")
    lines = cr.compare_arms(cr.read_sweep(str(tmp_path)), "8", "9", stage="confirm")
    assert "baseline at limit 8 (1 run(s)) vs BFF at limit 9 (1 run(s))" in lines[0]
    assert "BFF ahead" in next(ln for ln in lines if "throughput" in ln)
