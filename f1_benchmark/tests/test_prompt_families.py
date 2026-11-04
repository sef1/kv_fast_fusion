"""The family filter decides a whole scenario's prompts, and being off must change nothing."""

import sys
import types

import pytest

pytest.importorskip("aiohttp")

from f1_benchmark import f1_main  # noqa: E402


def _base(n=100):
    return list(range(n))


def test_near_identical_prompts_form_a_family():
    a, b = _base(), _base()
    b[5] = 999          # differs early, so a prefix cache could not share these
    assert f1_main.prompt_families([a, b]) == [[0, 1]]


def test_a_prompt_without_a_partner_is_left_out():
    other = list(range(1000, 1100))
    assert f1_main.prompt_families([_base(), other]) == []


def test_the_match_is_by_position_not_by_content():
    shifted = [7] + _base()[:-1]    # the same tokens one place later match almost nowhere
    assert f1_main.prompt_families([_base(), shifted]) == []


def test_the_bar_is_exact():
    a = _base()
    just_over = _base()
    just_under = _base()
    for i in range(20):
        just_over[i] = 5000 + i     # 80 of 100 match
    for i in range(21):
        just_under[i] = 6000 + i    # 79 of 100 match
    assert f1_main.prompt_families([a, just_over]) == [[0, 1]]
    assert f1_main.prompt_families([a, just_under]) == []


def test_a_short_prompt_inside_a_long_one_is_related():
    assert f1_main.prompt_families([_base(50), _base(100)]) == [[0, 1]]


def test_a_chain_is_one_family_ordered_by_first_appearance():
    far = list(range(5000, 5100))
    mid = _base()
    near = _base()
    for i in range(15):
        mid[i] = 9000 + i
    for i in range(15, 30):
        near[i] = 8000 + i          # near and mid match on 70, but both match the base on 85
    assert f1_main.prompt_families([far, far[:], mid, _base(), near]) == [[0, 1], [2, 3, 4]]


@pytest.fixture
def data(monkeypatch):
    def make(tokens, tag):
        return {"q": " ".join(str(t) for t in tokens), "r": tag}

    rows = [make([1000 + i for i in range(10)], "lonely"),
            make(range(10), "a1"),
            make([2000 + i for i in range(10)], "b1"),
            make(range(10), "a2"),
            make([2000 + i for i in range(10)], "b2"),
            make([3000 + i for i in range(30)], "too long")]

    class Tok:
        def __call__(self, text, add_special_tokens=False):
            return types.SimpleNamespace(input_ids=[int(x) for x in text.split()])

    monkeypatch.setitem(sys.modules, "datasets",
                        types.SimpleNamespace(load_dataset=lambda *a, **k: rows))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *a, **k: Tok())))


def _load(n=100, **kw):
    return f1_main.load_dataset_simple("x", "train", "q", "r", n, model="m", **kw)


def test_families_are_grouped_and_references_follow_their_prompts(data):
    prompts, refs = _load(max_in_tokens=20, group_families=True)
    assert [refs[str(i)] for i in range(len(prompts))] == ["a1", "a2", "b1", "b2"]
    assert prompts[0] == prompts[1] and prompts[2] == prompts[3] and prompts[0] != prompts[2]


def test_num_samples_applies_after_grouping(data):
    prompts, refs = _load(n=3, max_in_tokens=20, group_families=True)
    assert [refs[str(i)] for i in range(3)] == ["a1", "a2", "b1"]


def test_off_by_default_nothing_changes(data):
    prompts, refs = _load(n=3)
    assert [refs[str(i)] for i in range(3)] == ["lonely", "a1", "b1"]
