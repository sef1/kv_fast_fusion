"""The input-length filter decides which prompts a run uses, so results depend on it being exact."""

import hashlib
import sys
import types

import pytest

pytest.importorskip("aiohttp")

from f1_benchmark import f1_main  # noqa: E402


@pytest.fixture
def rows(monkeypatch):
    """Prompts whose token count is their word count, without a dataset or a tokenizer download."""
    data = [{"q": " ".join("w" * 1 for _ in range(n)), "r": str(n)} for n in (5, 10, 15, 20, 25)]

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=text.split())

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *a, **k: data))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tok())))
    return data


def _lengths(**kw):
    prompts, refs = f1_main.load_dataset_simple("x", "train", "q", "r", 100, model="m", **kw)
    assert len(prompts) == len(refs)
    return [len(p.split()) for p in prompts]


def test_no_bounds_keeps_everything(rows):
    assert _lengths() == [5, 10, 15, 20, 25]


def test_only_a_minimum_behaves_as_before(rows):
    assert _lengths(min_tokens=15) == [15, 20, 25]


def test_both_bounds_are_inclusive(rows):
    assert _lengths(min_tokens=10, max_in_tokens=20) == [10, 15, 20]


def test_a_maximum_alone_still_loads_the_tokenizer(rows):
    assert _lengths(max_in_tokens=10) == [5, 10]


def test_skipped_prompts_do_not_count_towards_num_samples(rows):
    prompts, _ = f1_main.load_dataset_simple("x", "train", "q", "r", 2, model="m", min_tokens=10,
                                             max_in_tokens=20)
    assert [len(p.split()) for p in prompts] == [10, 15]


def test_the_fingerprint_matches_compare_runs():
    outs = [{"prompt": "a"}, {"prompt": "b"}]
    formula = hashlib.sha1("\n".join(o.get("prompt", "") for o in outs).encode()).hexdigest()[:8]
    assert f1_main.prompt_fingerprint(["a", "b"]) == formula
