"""The garbage detector and its join with the free-at-promotion trace. CPU only."""

import pytest

from f1_benchmark import degenerate_outputs as dg

U1 = "0a1b2c3d-0000-4000-8000-000000000001"
U2 = "0a1b2c3d-0000-4000-8000-000000000002"
U3 = "0a1b2c3d-0000-4000-8000-000000000003"

CODE = "\n".join(f"def handler_{i}(request, user_{i}):\n    return render(request, 'page{i}.html', "
                 f"{{'user': user_{i}, 'items': fetch({i})}}" for i in range(40))


def test_ordinary_code_is_not_degenerate():
    assert not dg.is_degenerate(CODE)


def test_a_repetition_loop_is_degenerate():
    assert dg.is_degenerate(CODE[:500] + "texttext" * 300)


def test_a_punctuation_tail_is_degenerate():
    assert dg.is_degenerate(CODE[:500] + " ] ]] [' ] ,  ]) \n" * 120)


def test_a_short_answer_is_never_called_degenerate():
    assert not dg.is_degenerate("   ]]  " * 10)


@pytest.mark.parametrize("opening", [
    "pythonpython``\n`\n with already wayq,`` with other",
    "pythonpythonó``\nq11`` other\n with with`` with way",
    " or or or or)\n```python\n## pins/views.py\nfrom rest_framework",
    " TO:: DO DO DO Z Z  \n\n    DOZE   \n```python\n## pins/views.py",
    " TO:::: ZZOZOZOZOZOZEZEZEZEZEZEZEZEZE\nZEZEZE and so on for a while",
    " right    CT        or or or or or TO TO## Code: pins/views.py",
    ",,  ,,,,, [ (,,,  (   \n\n, ' ' ' : :  \n\n\n\n\n\n\n and more text",
    ':\n ## ` " ##^;  explanation <,ooo code q s #^ ## ## s logoolu lines',
])
def test_answers_broken_from_the_first_token_are_garbled(opening):
    """All from BFF runs where the KV was shifted by one block. Several recover into real code, which
    is why the tail check reported 0."""
    assert dg.is_garbled(opening + "\n" + CODE)
    assert dg.is_garbage(opening + "\n" + CODE)


@pytest.mark.parametrize("opening", [
    "```python\n## pins/views.py\n",
    "## Code: users/models.py\n```python\n",
    "Here is a much funkier version of your SVG",
    "This SVG appears to be an emoji of a",
    " Code: models.py\n```python\n## models.py\nfrom flask import Flask",
    "python\n## user.py\nfrom __future__ import annotations\nimport os",
    "sql\nSELECT `number of caps` FROM table_11585 WHERE name = 'x'",
    " give this SVG an underwater feel, we need to add a filter and",
    " are two main ways to add a reflection to an SVG on the web",
])
def test_good_answers_are_not_garbled_even_with_a_dropped_first_token(opening):
    """The lower ones lost their first token and are otherwise fine, in a baseline run."""
    assert not dg.is_garbled(opening + CODE)


def test_an_opening_that_is_mostly_not_letters_is_garbled():
    assert dg.is_garbled(' an-[\n \nITYT \n\n\nS\n\n,\nss,sm � funcals\nelt �\n,s left\n\nss �ss ), �, �, �,,,\n) assumption,, note assumption assumption codin')


def test_a_few_stray_characters_are_garbled():
    assert dg.is_garbled(" not")


def test_an_empty_answer_is_a_failed_request_not_garbage():
    assert not dg.is_garbled("") and not dg.is_garbled("   \n ")


def test_the_report_counts_garbled_answers():
    good = {"generated_text": CODE, "response_id": "chatcmpl-a"}
    bad = {"generated_text": "pythonpython``\n`\n with" + CODE, "response_id": "chatcmpl-b"}
    assert "1/2 (1 garbled from the first token)" in dg.report([good, bad])[0]


def test_the_trace_names_each_role():
    log = (f"INFO BFF swap | req=chatcmpl-{U1} g=1 pos=3 victim=51 rep=41 owner=chatcmpl-{U2}\n"
           f"INFO BFF victim reuse | req=chatcmpl-{U3} g=0 pos=0 block=51 freed by "
           f"chatcmpl-{U1} 12 ms earlier\n")

    roles = dg.parse_trace(log)

    assert roles == {"swapped": {U1}, "owner": {U2}, "recipient": {U3}}
    assert dg.roles_of(U1, roles) == ["swapped"]
    assert dg.roles_of("nope", roles) == ["none"]


def test_the_report_counts_and_attributes_garbage():
    outputs = [{"generated_text": CODE, "response_id": f"chatcmpl-{U1}"},
               {"generated_text": "x" * 2000, "response_id": f"chatcmpl-{U3}"}]
    ledger = {U3: {"declined": 0, "planned": 40, "capped": False}}
    log = (f"BFF victim reuse | req=chatcmpl-{U3} g=1 pos=2 block=7 freed by chatcmpl-{U1} "
           "3 ms earlier\n")

    lines = dg.report(outputs, ledger, log)

    assert lines[0] == "degenerate outputs: 1/2"
    assert "recipient=1" in lines[1] and "none=0" in lines[1]
    assert lines[2] == f"  #1 {U3} declined=0/40 | recipient"


def test_shared_writes_and_lifecycle_events_join_the_garbage():
    outputs = [{"generated_text": "x" * 2000, "response_id": f"chatcmpl-{U2}"},
               {"generated_text": CODE, "response_id": f"chatcmpl-{U1}"}]
    log = (f"INFO BFF swap | req=chatcmpl-{U1} g=1 pos=0 victim=51 rep=41 owner=chatcmpl-{U2}\n"
           f"ERROR BFF shared-block write | req=chatcmpl-{U1}-ab12cd34 g=1 pos=0 block=41 "
           f"ref_cnt=2 tokens=[0,200) roles=swapped | other holders: chatcmpl-{U2}-ef56ab78@0\n"
           f"INFO BFF preempt | req=chatcmpl-{U1}-ab12cd34 computed=4100 roles=swapped\n")

    lines = dg.report(outputs, {}, log)

    assert lines[0] == "degenerate outputs: 1/2"
    assert "written=1 writer=0 preempted=0" in lines[2]
    assert "1 written, 1 writer, 1 preempted, 0 recomputed" in lines[2]
    assert lines[3] == f"  #0 {U2} declined=-/- | owner | written"


def test_an_alias_hit_marks_the_hitter_and_the_owner_of_the_hit_block():
    outputs = [{"generated_text": "x" * 2000, "response_id": f"chatcmpl-{U2}"},
               {"generated_text": CODE, "response_id": f"chatcmpl-{U1}"}]
    log = (f"INFO BFF swap | req=chatcmpl-{U1} g=1 pos=0 victim=51 rep=41 owner=chatcmpl-{U2} "
           "rep_ref=2 victim_ref=0\n"
           f"INFO BFF rearrival | req=chatcmpl-{U1}-ab12cd34 local_hit_tokens=128\n"
           f"INFO BFF alias hit | req=chatcmpl-{U1}-ab12cd34 g=1 pos=0 block=41 via=alias "
           f"of=chatcmpl-{U1}-00000000@0 ref_cnt=3 hit_tokens=128\n")

    lines = dg.report(outputs, {}, log)

    assert "hit_owner=1" in lines[2] and "rearrived=0" in lines[2]
    assert lines[3] == f"  #0 {U2} declined=-/- | owner | hit_owner"


def test_clipped_requests_get_their_own_verdict_line():
    outputs = [{"generated_text": "x" * 2000, "response_id": f"chatcmpl-{U1}"},
               {"generated_text": CODE, "response_id": f"chatcmpl-{U2}"}]
    ledger = {U1: {"declined": 30, "planned": 40, "capped": True, "clipped": 10},
              U2: {"declined": 28, "planned": 40, "capped": True, "clipped": 9}}

    lines = dg.report(outputs, ledger)

    assert lines[1] == "  clipped requests (BFF_V2_CLIP_FRAC): 1/2 degenerate, 19 aliases kept"
    assert lines[2] == f"  #0 {U1} declined=30/40 capped clipped=10"


def test_without_a_log_the_report_still_counts():
    lines = dg.report([{"generated_text": "x" * 2000}])

    assert lines == ["degenerate outputs: 1/1", "  #0 ? declined=-/-"]


def test_a_log_without_any_trace_skips_attribution():
    lines = dg.report([{"generated_text": "x" * 2000}], {}, "INFO nothing relevant\n")

    assert lines[1].startswith("  (no free-at-promotion trace")
    assert lines[2] == "  #0 ? declined=-/-"
