"""The garbage detector and its join with the free-at-promotion trace. CPU only."""

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


def test_without_a_log_the_report_still_counts():
    lines = dg.report([{"generated_text": "x" * 2000}])

    assert lines == ["degenerate outputs: 1/1", "  #0 ? declined=-/-"]


def test_a_log_without_any_trace_skips_attribution():
    lines = dg.report([{"generated_text": "x" * 2000}], {}, "INFO nothing relevant\n")

    assert lines[1].startswith("  (no free-at-promotion trace")
    assert lines[2] == "  #0 ? declined=-/-"
