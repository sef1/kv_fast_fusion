"""Pin the scale mode so the suite means the same thing whatever the shell has exported.

`BFF_SCALE_MODE` is read once at import into `pd_dedup_v2.SCALE_MODE` and `pd_lsh.RATIO_MODE`, and
it changes real behaviour: ratio mode ranks candidates by cosine instead of substitution error, and
refuses to alias at all when the producer shipped no norms. Every test written before that mode
existed therefore asserts raw-mode behaviour without saying so, and running the suite in a shell
that had exported `BFF_SCALE_MODE=ratio` — which is exactly what happens while working on the mode —
failed 16 of them for reasons that have nothing to do with the code under test.

Raw is the default here because it is what those tests mean. A test about ratio mode opts in by
monkeypatching the same two attributes, which runs after this fixture and so wins.
"""

import pytest


@pytest.fixture(autouse=True)
def _pin_scale_mode(monkeypatch):
    from kv_fast_fusion import pd_dedup_v2, pd_lsh
    monkeypatch.setattr(pd_dedup_v2, "SCALE_MODE", "raw")
    monkeypatch.setattr(pd_lsh, "RATIO_MODE", False)
