"""make_longbench_e: the token count, the selection and the record format, with no network."""
import json

from f1_benchmark import make_longbench_e as mk


class _Enc:
    def __init__(self, ids):
        self.input_ids = ids


class FakeTok:
    """Counts one token per word; the chat template adds two words, like a real one adds tokens."""

    def __init__(self, template=True):
        self.template = template
        self.kwargs = None

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):
        if not self.template:
            raise ValueError("no chat template")
        assert tokenize is False, "must render text, not ids (ids come back as a dict on newer transformers)"
        self.kwargs = kw
        return "<user> " + msgs[0]["content"] + " <assistant>"

    def __call__(self, text, add_special_tokens=False):
        return _Enc(text.split())


def test_count_includes_template_and_turns_thinking_off():
    tok = FakeTok()
    assert mk.make_counter(tok)("a b c") == 5
    assert tok.kwargs == {"enable_thinking": False}


def test_count_falls_back_to_raw_prompt_without_template():
    assert mk.make_counter(FakeTok(template=False))("a b c") == 3


def _row(words, length=1000, answers=("x",), _id="i"):
    return {"context": " ".join(["w"] * words), "input": "q", "length": length,
            "answers": list(answers), "_id": _id}


def test_select_filters_by_template_length_and_keeps_order():
    count = lambda p: len(p.split())
    rows = [_row(10, _id="short"), _row(2000, _id="a"), _row(9000, _id="long"), _row(2100, _id="b")]
    out = mk.select({"multifieldqa_en_e": iter(rows)}, count, n=10, min_tok=1500, max_tok=4000)
    assert [r["id"] for r in out] == ["a", "b"]
    assert all(1500 <= r["tokens"] <= 4000 for r in out)


def test_select_word_prefilter_skips_out_of_range_rows_but_not_code_tasks():
    count = lambda p: len(p.split())
    qa = mk.select({"hotpotqa_e": iter([_row(2000, length=100)])}, count, 10, 0, 10 ** 6)
    code = mk.select({"lcc_e": iter([_row(2000, length=100)])}, count, 10, 0, 10 ** 6)
    assert qa == [] and len(code) == 1


def test_select_spreads_evenly_and_stops_reading_early():
    count = lambda p: len(p.split())
    read = []

    def rows(tag):
        for i in range(100):
            read.append(tag)
            yield _row(2000, _id=f"{tag}{i}")

    out = mk.select({"qasper_e": rows("q"), "samsum_e": rows("s")}, count, n=4, min_tok=0, max_tok=10 ** 6)
    assert [r["task"] for r in out] == ["qasper_e", "qasper_e", "samsum_e", "samsum_e"]
    assert read.count("q") == 2 and read.count("s") == 2


def test_reference_is_a_string_and_all_answers_kept():
    count = lambda p: 1
    out = mk.select({"qasper_e": iter([_row(5, answers=("yes", "Yes."))])}, count, 1, 0, 10)
    assert out[0]["reference"] == "yes" and out[0]["answers"] == ["yes", "Yes."]
    json.dumps(out[0])


def test_main_writes_jsonl_from_a_local_zip(tmp_path, monkeypatch):
    import sys
    import types
    import zipfile

    zpath = tmp_path / "data.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("data/trec_e.jsonl", "\n".join(json.dumps(_row(30, _id=f"t{i}")) for i in range(3)))
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **k: FakeTok())
    monkeypatch.setitem(sys.modules, "transformers", fake)
    out = tmp_path / "o.jsonl"
    rc = mk.main(["--tokenizer", "x", "--zip", str(zpath), "--tasks", "trec_e", "--n", "2",
                  "--min-tok", "0", "--max-tok", "1000", "--out", str(out)])
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert rc == 0 and [l["id"] for l in lines] == ["t0", "t1"]
    assert set(lines[0]) == {"prompt", "reference", "answers", "task", "id", "tokens"}
