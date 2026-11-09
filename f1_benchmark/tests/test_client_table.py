"""The client table and the script that collects it. A stand-in replaces the real benchmark."""

import json
import os
import stat
import subprocess
import sys

import pytest

from f1_benchmark import client_table as ct

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPT = os.path.join(ROOT, "examples/online_serving/disaggregated_serving_mooncake_ascend_xpyd/run_client_table.sh")
CODE = "```python\n## a.py\n" + "\n".join(f"def f{i}(x):\n    return x + {i}" for i in range(40)) + "\n```"
ENGINE = "INFO 12:00:00 x: prompt throughput: 2000.0 tokens/s, generation throughput: 200.0 tokens/s, Running: 8 reqs, Waiting: 1 reqs, GPU KV cache usage: 90.0%\n"


def save(root, cell, name, arm="baseline", tok=280.0, cfg="", text=CODE, prompt="## Code: a.py"):
    run = root / cell / name
    (run / "logs_latest").mkdir(parents=True)
    (run / "results_latest").mkdir()
    (run / "DONE").write_text("")
    (run / "config.txt").write_text(f"arm={arm}\ndecode_limit=24\nscenario=families\nPROXY_CONTINUE_ANSWER=1\n{cfg}\n")
    (run / "logs_latest" / "x-serving.txt").write_text(f"Inference completed in 400.0 s\n  throughput: {tok} output tok/s (1 toks) | 0.5 req/s\n  Mean F1 Score: 0.6300 (over 2 samples)\n")
    (run / "logs_latest" / "prefill-0.txt").write_text(ENGINE)
    (run / "logs_latest" / "decode-0.txt").write_text(ENGINE + "BFF KV sizing | num_gpu_blocks=754 | groups=2 [40, 39] | x\n")
    (run / "logs_latest" / "proxy.txt").write_text("BFF proxy continued answer | partial_chars=5 retry=1\n" * 3)
    (run / "console.txt").write_text("bff v2 wire saving: 40.0% of blocks\n" if arm == "bff" else "")
    (run / "results_latest" / "raw_outputs.json").write_text(json.dumps(
        {"outputs": [{"prompt": prompt, "generated_text": text}, {"prompt": prompt + "b", "generated_text": text}]}))
    (run / "results_latest" / "benchmark_results.json").write_text(json.dumps(
        {"completed": 2, "total_output_tokens": 1200, "finish_length_pct": 20.0,
         "ttft_ms": {"median": 50000, "p99": 80000}, "tpot_ms": {"median": 45, "p99": 300},
         "e2e_latency_ms": {"median": 90000, "p99": 180000}}))


def rows(root):
    lines, warnings = ct.table(ct.read_table(str(root)))
    return {ln.split("|")[3].strip(): ln for ln in lines[2:]}, warnings


def test_the_label_comes_from_the_config_not_the_folder_name(tmp_path):
    save(tmp_path, "c", "baseline_r1")
    save(tmp_path, "c", "clip07_r1", arm="bff", cfg="BFF_V2_CLIP_FRAC=0.4")
    r, _ = rows(tmp_path)
    assert "clip 0.4" in r and "clip 0.7" not in r


def test_settings_are_named_from_what_ran():
    assert ct.setting_of({"arm": "baseline"}) == "baseline"
    assert ct.setting_of({"arm": "bff", "BFF_V2_MAX_REQ_DECLINE": "1"}) == "no cap"
    assert ct.setting_of({"arm": "bff", "BFF_V2_DEDUP": "0"}) == "dedup off"
    assert ct.setting_of({"arm": "bff", "BFF_V2_CLIP_FRAC": "0.7", "BFF_V2_FREE_AT_PROMOTION": "0"}) == "clip 0.7, no freeing"
    assert ct.setting_of({"arm": "bff", "BFF_V2_MAX_REQ_DECLINE": "1", "BFF_EDGE_LAYERS": "20",
                          "BFF_EDGE_SPLIT": "1"}) == "no cap"
    assert ct.setting_of({"arm": "bff", "BFF_V2_MAX_REQ_DECLINE": "1", "BFF_EDGE_LAYERS": "10",
                          "BFF_EDGE_SPLIT": "3"}) == "no cap, edge 10 split 3"


def test_kv_not_pulled_counts_every_signed_group(tmp_path):
    """Split layout: the saving covers g1-g3 (59 of 79 layers), not just the last group (19)."""
    save(tmp_path, "c", "split", arm="bff")
    run = tmp_path / "c" / "split"
    (run / "logs_latest" / "decode-0.txt").write_text(
        ENGINE + "BFF KV sizing | num_gpu_blocks=1503 | groups=4 [20, 20, 20, 19] | x\n")
    (run / "console.txt").write_text("bff v2 wire saving: 8.1% of blocks\n")
    out = ct.extras(str(run), {"arm": "bff"})
    assert abs(out["kv_not_pulled"] - 8.1 * 59 / 79) < 1e-6


def test_a_gain_needs_every_repeat_ahead(tmp_path):
    for i, t in enumerate((280.0, 285.0)):
        save(tmp_path, "c", f"baseline_r{i}", tok=t)
    for i, t in enumerate((300.0, 305.0)):
        save(tmp_path, "c", f"nocap_r{i}", arm="bff", tok=t, cfg="BFF_V2_MAX_REQ_DECLINE=1")
    r, w = rows(tmp_path)
    assert "every repeat ahead" in r["no cap"] and not w
    assert "(280.0-285.0)" in r["baseline"] and "| - |" in r["baseline"]


def test_overlapping_repeats_are_not_called_a_gain(tmp_path):
    for i, t in enumerate((280.0, 300.0)):
        save(tmp_path, "c", f"baseline_r{i}", tok=t)
    for i, t in enumerate((295.0, 305.0)):
        save(tmp_path, "c", f"nocap_r{i}", arm="bff", tok=t, cfg="BFF_V2_MAX_REQ_DECLINE=1")
    assert "no clear difference" in rows(tmp_path)[0]["no cap"]


def test_one_run_is_marked_unconfirmed(tmp_path):
    save(tmp_path, "c", "baseline_r1")
    save(tmp_path, "c", "nocap_r1", arm="bff", tok=300.0, cfg="BFF_V2_MAX_REQ_DECLINE=1")
    assert "unconfirmed" in rows(tmp_path)[0]["no cap"]


def test_unlike_runs_in_one_cell_are_called_out(tmp_path):
    save(tmp_path, "c", "baseline_r1")
    save(tmp_path, "c", "nocap_r1", arm="bff", cfg="BFF_V2_MAX_REQ_DECLINE=1\nTHROUGHPUT_ONLY=1")
    assert any("fixed-length and normal" in w for w in rows(tmp_path)[1])


def test_fixed_runs_report_no_quality_and_ignore_forced_text(tmp_path):
    save(tmp_path, "c", "baseline_r1", cfg="THROUGHPUT_ONLY=1", text=CODE + "\n" + "⟩" * 2000)
    r, _ = rows(tmp_path)
    cells = [c.strip() for c in r["baseline"].split("|")]
    assert cells[12] == "-" and cells[15] == "0", "the tail check fires on forced text, which is not garbage"


def test_a_garbled_answer_counts_in_normal_and_fixed_runs(tmp_path):
    save(tmp_path, "n", "baseline_r1", text="pythonpython``\n`\n with already way" + CODE)
    save(tmp_path, "f", "baseline_r1", cfg="THROUGHPUT_ONLY=1", text="pythonpython``\n`\n with already way" + CODE)
    lines, _ = ct.table(ct.read_table(str(tmp_path)))
    assert all(ln.split("|")[15].strip() == "2" for ln in lines[2:])


def test_evictions_and_kv_saved_are_read(tmp_path):
    save(tmp_path, "c", "nocap_r1", arm="bff", cfg="BFF_V2_MAX_REQ_DECLINE=1")
    run = ct.read_table(str(tmp_path))[0]
    assert run["evictions"] == 3 and abs(run["kv_not_pulled"] - 40.0 * 39 / 79) < 1e-6


def test_main_writes_both_files(tmp_path, capsys):
    save(tmp_path, "c", "baseline_r1")
    assert ct.main([str(tmp_path)]) == 0
    assert (tmp_path / "client_table.md").read_text().count("selected in BFF's favour") == 1
    assert "baseline_r1" in (tmp_path / "client_runs.csv").read_text()


# ------------------------------------------------------------------ the collection script
FAKE = """#!/bin/bash
rm -rf "$LOGS_LATEST" "$RESULTS_LATEST"; mkdir -p "$LOGS_LATEST" "$RESULTS_LATEST"
if [[ -n "${FAIL_ON:-}" && -e "$FAIL_ON" ]]; then rm -f "$FAIL_ON"; echo crashed; exit 1; fi
echo "$BASELINE $DECODE_MAX_CONCURRENCY ${THROUGHPUT_ONLY:-} ${BFF_V2_CLIP_FRAC:-} ${BFF_V2_MAX_REQ_DECLINE:-} $PROXY_CONTINUE_ANSWER" >> "$CALLS"
echo "  throughput: 280.0 output tok/s (1 toks) | 0.5 req/s" > "$LOGS_LATEST/x-serving.txt"
touch "$LOGS_LATEST/decode-0.txt" "$LOGS_LATEST/prefill-0.txt"
echo "Benchmark done"
"""


@pytest.fixture
def box(tmp_path):
    fake = tmp_path / "fake.sh"
    fake.write_text(FAKE)
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = dict(os.environ, RUN_SCRIPT=str(fake), LOGS_LATEST=str(tmp_path / "logs"), RESULTS_LATEST=str(tmp_path / "res"),
               TABLE_DIR=str(tmp_path / "table"), CALLS=str(tmp_path / "calls"), EXPECT_FINGERPRINT="", MODEL="m",
               PATH=os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"])
    for k in ("BFF_V2_CLIP_FRAC", "BFF_V2_MAX_REQ_DECLINE", "PROXY_CONTINUE_ANSWER", "THROUGHPUT_ONLY", "DRY_RUN"):
        env.pop(k, None)

    def run(**kw):
        return subprocess.run(["bash", SCRIPT], env={**env, **kw}, capture_output=True, text=True, cwd=ROOT)

    return run, tmp_path


def calls(tmp_path):
    f = tmp_path / "calls"
    return f.read_text().splitlines() if f.exists() else []


def test_the_plan_is_thirty_runs(box):
    run, _ = box
    out = run(DRY_RUN="1").stdout
    assert "0 done, 30 to run" in out
    assert out.count("families_normal/") == 9 and out.count("families_fixed/") == 9
    assert out.count("121_normal/") == 6 and out.count("200_normal/") == 6


def test_it_stops_after_a_batch_and_resumes_where_it_stopped(box):
    run, tmp = box
    first = run(BATCH="2", ONLY="families_normal", REPEATS="1")
    assert first.returncode == 0 and "2 runs done, 1 left" in first.stdout and "Restart the container" in first.stdout
    assert len(calls(tmp)) == 2
    second = run(BATCH="2", ONLY="families_normal", REPEATS="1")
    assert "3 runs done, 0 left" in second.stdout and "Restart the container" not in second.stdout
    assert len(calls(tmp)) == 3, "the finished runs were not run again"


def test_each_arm_gets_its_settings_and_the_restart_fix(box):
    run, tmp = box
    run(BATCH="9", ONLY="families_fixed", REPEATS="1")
    assert calls(tmp) == ["mooncakev1 24 1   1", "bff_pull_v2 24 1 0.7  1", "bff_pull_v2 24 1 0.4 1 1"]


def test_even_repeats_reverse_the_order(box):
    run, tmp = box
    run(BATCH="9", ONLY="121_normal", REPEATS="2")
    assert [c.split()[0] for c in calls(tmp)] == ["mooncakev1", "bff_pull_v2", "bff_pull_v2", "mooncakev1"]


def test_a_failed_run_stops_it_and_is_tried_again(box):
    run, tmp = box
    flag = tmp / "fail"
    flag.write_text("")
    failed = run(ONLY="121_normal", REPEATS="1", FAIL_ON=str(flag))
    assert failed.returncode == 1 and "FAILED" in failed.stdout
    assert len(calls(tmp)) == 0
    ok = run(ONLY="121_normal", REPEATS="1")
    assert ok.returncode == 0 and len(calls(tmp)) == 2


def test_the_slow_limit_guard_lets_the_table_limits_through(box):
    run, tmp = box
    assert run(ONLY="200_normal", REPEATS="1").returncode == 0
    assert all(c.split()[1] == "32" for c in calls(tmp))
