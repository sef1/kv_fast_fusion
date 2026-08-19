"""A connector that fails to register must not fail silently when the run needs it.

`apply_fast_fusion_pd_patch` registers each connector inside a `try/except` so a box without
mooncake can still run the NCCL path. That is right, but it swallowed the failure for the connector
the run actually SELECTED: the process kept going and died ~400 lines later inside pydantic with
`Unsupported connector type: MooncakeConnectorFFv2`, naming neither the module nor the ImportError.

That is exactly what happened on 2026-08-19 at commits `c7bce3e01` / `6122e3126`, where
`mooncake_connector_ff_v2.py` was committed but `pd_lsh.py` was not — so `from kv_fast_fusion import
pd_lsh` raised, BOTH GPU connectors silently dropped out of the registry, and two debugging cycles
went into a message that pointed nowhere near the cause.

CPU only; no vLLM engine, no GPU.
"""

import pytest

from kv_fast_fusion.fast_fusion_pd_patch import (
    _registration_failed,
    _selected_connector_names,
)

CFG = ('{"kv_connector":"MooncakeConnectorFFv2","kv_role":"kv_producer",'
       '"kv_connector_extra_config":{"mooncake_protocol":"tcp"}}')
MULTI = ('{"kv_connector":"MultiConnector","kv_role":"kv_producer","kv_connector_extra_config":'
         '{"connectors":[{"kv_connector":"MooncakeLayerwiseConnectorFFv2"},'
         '{"kv_connector":"AscendStoreConnector"}]}}')


def _argv(monkeypatch, *args):
    monkeypatch.setattr("sys.argv", ["vllm", "serve", "model", *args])


def test_the_selected_connector_is_read_from_the_launch_flag(monkeypatch):
    _argv(monkeypatch, "--kv-transfer-config", CFG)
    assert _selected_connector_names() == {"MooncakeConnectorFFv2"}


def test_the_equals_form_of_the_flag_is_read_too(monkeypatch):
    _argv(monkeypatch, f"--kv-transfer-config={CFG}")
    assert _selected_connector_names() == {"MooncakeConnectorFFv2"}


def test_connectors_nested_under_multiconnector_count_as_selected(monkeypatch):
    """The Ascend path wraps the mover in MultiConnector, so the name that matters is not at the
    top level. Missing it would restore the silent failure for exactly that deployment."""
    _argv(monkeypatch, "--kv-transfer-config", MULTI)
    assert _selected_connector_names() == {
        "MultiConnector", "MooncakeLayerwiseConnectorFFv2", "AscendStoreConnector"}


def test_a_selected_connector_that_fails_to_register_is_fatal(monkeypatch):
    _argv(monkeypatch, "--kv-transfer-config", CFG)
    with pytest.raises(RuntimeError, match="selected by --kv-transfer-config"):
        _registration_failed("MooncakeConnectorFFv2", ImportError("no module named pd_lsh"))


def test_the_original_cause_survives_in_the_message(monkeypatch):
    """The whole point: the ImportError must be readable at the point of failure, not inferred."""
    _argv(monkeypatch, "--kv-transfer-config", CFG)
    with pytest.raises(RuntimeError) as ei:
        _registration_failed("MooncakeConnectorFFv2", ImportError("cannot import name 'pd_lsh'"))
    assert "pd_lsh" in str(ei.value)
    assert isinstance(ei.value.__cause__, ImportError)


def test_an_unselected_connector_that_fails_is_only_a_warning(monkeypatch):
    """A box without mooncake must still run the NCCL path. This is the behaviour the try/except
    existed for, and it has to survive the fix."""
    _argv(monkeypatch, "--kv-transfer-config", CFG)
    _registration_failed("P2pNcclConnectorFF", ImportError("no mooncake here"))


@pytest.mark.parametrize("args", [(), ("--kv-transfer-config", "{not json"),
                                  ("--kv-transfer-config", "{}")])
def test_an_undeterminable_selection_warns_rather_than_killing_the_process(monkeypatch, args):
    """Processes that carry no --kv-transfer-config (the API server front-end) or an unparseable
    one must not be refused: they never needed the connector, and killing them would be a worse
    bug than the one being fixed."""
    _argv(monkeypatch, *args)
    _registration_failed("MooncakeConnectorFFv2", ImportError("boom"))
