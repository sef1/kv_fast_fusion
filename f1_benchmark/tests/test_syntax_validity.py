"""Regression tests for the F1 benchmark's syntax-validity scorer.

These exist because of a concrete loss: the con200 run finished all 1023 completions, then the
scoring pass died with `MemoryError` out of `ast.parse` on a single degenerate generation, throwing
away ~25 minutes of NPU time and every other sample's score with it.

The rule this file pins: `check_syntax_validity` answers a question ("is this valid Python?") and
must always answer it. Input the parser cannot survive is not valid Python, so `False` is correct —
raising is not.

CPU only, no NPU, no server.
"""

import pytest

from f1_benchmark.f1_main import check_syntax_validity


# =====================================================================================
# the ordinary cases still work
# =====================================================================================
def test_valid_python_is_valid():
    assert check_syntax_validity("", "x = 1\n") is True


def test_a_fenced_code_block_is_extracted_and_parsed():
    assert check_syntax_validity("", "Here you go:\n```python\ndef f():\n    return 1\n```") is True


def test_prose_is_not_valid_python():
    assert check_syntax_validity("", "I want this SVG to look like a painting.") is False


def test_an_empty_prediction_is_not_valid():
    assert check_syntax_validity("", "") is False


def test_a_ragged_indented_fragment_is_not_valid():
    """Documents actual behaviour rather than the intent behind the dedent path.

    `extract_code` ends with `text.strip()`, which removes the leading indent of the FIRST line
    only. A uniformly indented fragment therefore arrives at `textwrap.dedent` already ragged —
    line 1 at column 0, the rest indented — so dedent finds no common prefix and the second parse
    attempt fails too. The second attempt is close to dead code as a result.

    Left as-is deliberately: changing it would move the AST-validity numbers and break comparability
    with every run measured so far. Noted here so the next person does not read the dedent branch as
    working."""
    assert check_syntax_validity("", "    x = 1\n    y = 2\n") is False


# =====================================================================================
# the pathological cases that used to abort the whole run
# =====================================================================================
def test_deeply_nested_brackets_do_not_raise():
    """The shape that killed con200. CPython's parser cost is superlinear in nesting depth, so a
    runaway completion of unbalanced brackets exhausts memory rather than raising SyntaxError."""
    assert check_syntax_validity("", "(" * 100_000) is False


def test_a_null_byte_does_not_raise():
    """ast.parse raises ValueError, not SyntaxError, on embedded nulls."""
    assert check_syntax_validity("", "x = 1\x00") is False


def test_an_oversized_blob_is_refused_without_parsing():
    """Above the size guard we decline to parse at all. A 6000-token runaway is never valid Python,
    so the answer is unchanged and the cost becomes bounded."""
    assert check_syntax_validity("", "[" * 300_000) is False


def test_a_pathological_prompt_cannot_poison_a_good_prediction():
    """The first attempt concatenates prompt + prediction. A huge prompt must not make a valid
    prediction unscoreable — the second attempt still parses the prediction alone."""
    assert check_syntax_validity("(" * 100_000, "x = 1\n") is True


@pytest.mark.parametrize("payload", [
    "(" * 50_000,
    "[" * 50_000,
    "{" * 50_000,
    "\x00" * 100,
    "def f(" * 20_000,
])
def test_no_hostile_input_escapes_as_an_exception(payload):
    """The invariant, stated directly: whatever the model emitted, scoring returns a bool."""
    assert check_syntax_validity("", payload) in (True, False)
