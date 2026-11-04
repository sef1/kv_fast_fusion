"""Which KV connectors this process was launched with. Shared by the GPU and Ascend patches."""


def _selected_connector_names() -> set:
    """Connector names this process was actually launched with, from ``--kv-transfer-config``.

    Read from ``sys.argv`` because the patch runs at import, long before a ``VllmConfig`` exists.
    Includes connectors nested under MultiConnector's ``kv_connector_extra_config.connectors``.
    Returns an empty set when the flag is absent or unparseable — callers must treat that as
    "unknown", never as "not selected"."""
    import json
    import sys

    argv = sys.argv
    raw = None
    for i, a in enumerate(argv):
        if a == "--kv-transfer-config" and i + 1 < len(argv):
            raw = argv[i + 1]
        elif a.startswith("--kv-transfer-config="):
            raw = a.split("=", 1)[1]
    if not raw:
        return set()
    try:
        cfg = json.loads(raw)
    except Exception:
        return set()

    names = set()

    def _walk(node):
        if not isinstance(node, dict):
            return
        name = node.get("kv_connector")
        if isinstance(name, str):
            names.add(name)
        extra = node.get("kv_connector_extra_config")
        if isinstance(extra, dict):
            for child in extra.get("connectors") or ():
                _walk(child)

    _walk(cfg)
    return names
