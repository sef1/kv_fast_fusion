"""The sibling-content check, and its column in compare_runs and the client table. CPU only."""
import json

from f1_benchmark import client_table as ct
from f1_benchmark import compare_runs as cr
from f1_benchmark import sibling_drift as sd

BODY = "\n".join(f"Context line {i}: the project uses Flask, SQLAlchemy and WTForms for item {i}." for i in range(80))
FILES = ("models.py", "forms.py", "views.py", "config.py")
PROMPTS = [f"## Code: {name} Write code with triple quotes.\n{BODY}" for name in FILES]
UNRELATED = "\n".join(f"Totally different request number {i} about sorting algorithms in Rust." for i in range(80))


def code(name: str, variant: int = 0) -> str:
    """A believable answer for one file; ``variant`` changes the wording a little, like a second repeat."""
    stem = name.split(".")[0]
    lines = [f"## {name}", "```python"]
    lines += [f"class {stem.title()}Thing{k}(Base{variant}):\n    field_{stem}_{k} = Column({k})  # {stem} part {k}"
              for k in range(30)]
    return "\n".join(lines + ["```"])


def answers(copy_at=None, variant=0):
    """One answer per prompt; at ``copy_at`` the answer writes its SIBLING's file under its own header."""
    out = [code(n, variant) for n in FILES]
    if copy_at is not None:
        out[copy_at] = f"## {FILES[copy_at]}\n" + code(FILES[copy_at + 1], variant).split("\n", 1)[1]
    return out


def test_near_identical_prompts_close_in_order_are_siblings():
    sibs = sd.siblings(PROMPTS + [UNRELATED])
    assert sibs[0] == [1, 2, 3] and sibs[4] == []
    far = sd.siblings([PROMPTS[0]] + [UNRELATED] * 20 + [PROMPTS[1]], window=16)
    assert far[0] == [] and far[-1] == []


def test_a_copy_of_the_siblings_answer_is_flagged_and_an_own_answer_is_not():
    refs = [answers(variant=1), answers(variant=2)]
    hits = sd.drifted(answers(copy_at=0), refs, PROMPTS)
    assert [(i, j) for i, j, _o, _s in hits] == [(0, 1)]
    i, j, own, other = hits[0]
    assert other > own + sd.MARGIN
    assert sd.drifted(answers(), refs, PROMPTS) == []


def test_answers_without_a_sibling_prompt_are_never_judged():
    prompts = [PROMPTS[0], UNRELATED]
    texts = [code("forms.py"), code("models.py")]      # swapped, but the prompts are not siblings
    assert sd.drifted(texts, [[code("models.py"), code("forms.py")]], prompts) == []


def _run(name, arm, outs, fixed=False, fp="f"):
    return {"name": name, "arm": arm, "fingerprint": fp, "fixed": fixed,
            "outputs": [{"prompt": p, "generated_text": t} for p, t in zip(PROMPTS, outs)]}


def test_mark_judges_baselines_against_each_other_and_skips_what_cannot_be_judged():
    b1, b2 = _run("b1", "baseline", answers(variant=1)), _run("b2", "baseline", answers(variant=2))
    bad = _run("bff", "bff", answers(copy_at=0))
    fixed = _run("fx", "bff", answers(copy_at=0), fixed=True)
    other = _run("o", "bff", answers(copy_at=0), fp="g")
    sd.mark([b1, b2, bad, fixed, other])
    assert (b1["sibling"], b2["sibling"], bad["sibling"]) == (0, 0, 1)
    assert fixed["sibling"] is None, "fixed-length text is not judged"
    assert other["sibling"] is None, "a run with no baseline of the same prompts gets no count"
    lone = _run("b", "baseline", answers())
    sd.mark([lone])
    assert lone["sibling"] is None, "one baseline has nothing to be judged against"


def test_describe_names_both_files_and_the_kv_replaced(tmp_path):
    (tmp_path / "results_latest").mkdir()
    (tmp_path / "results_latest" / "bff_stats_1.json").write_text(
        json.dumps({"request_ledger": {"req-0": {"declined": 17, "planned": 19}}}))
    run = _run("nocap_r1", "bff", answers(copy_at=0))
    run["outputs"][0]["response_id"] = "req-0"
    run["dir"] = str(tmp_path)
    line = sd.describe(run, (0, 1, 0.11, 0.75))
    assert "prompt 0 (models.py)" in line and "prompt 1's (forms.py)" in line and "17/19" in line


# ---- the columns -------------------------------------------------------------------------------------------

SERVING = "Inference completed in 300.0 s\n  throughput: {tok} output tok/s (100 toks) | 0.36 req/s\n"


def _save(root, name, arm, outs, cfg=""):
    run = root / name
    (run / "logs_latest").mkdir(parents=True)
    (run / "results_latest").mkdir()
    (run / "DONE").write_text("")
    (run / "config.txt").write_text(f"arm={arm}\ndecode_limit=24\nstage=sweep\nscenario=families\n{cfg}\n")
    (run / "logs_latest" / "x-serving.txt").write_text(SERVING.format(tok=300.0))
    (run / "results_latest" / "raw_outputs.json").write_text(json.dumps(
        {"outputs": [{"prompt": p, "generated_text": t} for p, t in zip(PROMPTS, outs)]}))


def test_compare_runs_prints_the_count_and_each_flagged_answer(tmp_path):
    _save(tmp_path, "baseline_r1", "baseline", answers(variant=1))
    _save(tmp_path, "baseline_r2", "baseline", answers(variant=2))
    _save(tmp_path, "nocap_r1", "bff", answers(copy_at=0))
    runs = cr.read_sweep(str(tmp_path))
    lines = cr.table(runs)
    col = lines[0].split().index("sib")
    assert [ln.split()[col] for ln in lines[1:]] == ["0", "0", "1"]
    detail = cr.sibling_lines(runs)
    assert len(detail) == 2 and "nocap_r1: prompt 0 (models.py) reads like prompt 1's (forms.py)" in detail[1]
    assert any("sibling content" in ln and "0 vs 1" in ln for ln in cr.compare_arms(runs, "24", "24"))


def test_compare_runs_shows_a_dash_without_a_second_baseline(tmp_path):
    _save(tmp_path, "baseline_r1", "baseline", answers())
    lines = cr.table(cr.read_sweep(str(tmp_path)))
    assert lines[1].split()[lines[0].split().index("sib")] == "-"


def test_client_table_has_a_sibling_column_and_lists_the_answers(tmp_path):
    for i, v in enumerate((1, 2)):
        _save(tmp_path / "families_normal", f"baseline_r{i}", "baseline", answers(variant=v))
        _save(tmp_path / "families_fixed", f"baseline_r{i}", "baseline", answers(variant=v), cfg="THROUGHPUT_ONLY=1")
    _save(tmp_path / "families_normal", "nocap_r1", "bff", answers(copy_at=0), cfg="BFF_V2_MAX_REQ_DECLINE=1")
    _save(tmp_path / "families_fixed", "nocap_r1", "bff", answers(copy_at=0),
          cfg="BFF_V2_MAX_REQ_DECLINE=1\nTHROUGHPUT_ONLY=1")
    lines, _ = ct.table(ct.read_table(str(tmp_path)))
    head = [c.strip() for c in lines[0].split("|")]
    col = head.index("Sibling")
    got = {(c[2].strip(), c[3].strip()): c[col].strip() for c in (ln.split("|") for ln in lines[2:])}
    assert got == {("normal", "baseline"): "0", ("normal", "no cap"): "1",
                   ("fixed", "baseline"): "-", ("fixed", "no cap"): "-"}
    assert ct.main([str(tmp_path)]) == 0
    md = (tmp_path / "client_table.md").read_text()
    assert "read like a sibling" in md and "families_normal/nocap_r1: prompt 0 (models.py)" in md
    assert "sibling" in (tmp_path / "client_runs.csv").read_text().splitlines()[0]
