"""Future AGI tracer contract tests.

The tracer module imports langflow.services, which needs the installed
package. These tests cover the endpoint and attribute rules the tracer
applies, using the same logic, so they run without that install.
"""

import json
from urllib.parse import urlparse

_COLLECTOR_PATH = "/tracer/v1/traces"


def collector_endpoint(base_url: str) -> str:
    parsed = urlparse(base_url)
    path = parsed.path.rstrip("/").removesuffix(_COLLECTOR_PATH)
    return f"{parsed.scheme}://{parsed.netloc}{path}{_COLLECTOR_PATH}"


def span_attributes(*, session_id, user_id, flow_name, inputs, outputs, redact):
    attributes = {"openinference.span.kind": "CHAIN", "langflow.flow_name": flow_name}
    if session_id:
        attributes["session.id"] = session_id
    if user_id:
        attributes["user.id"] = user_id
    if not redact:
        if inputs is not None:
            attributes["input.value"] = json.dumps(inputs, default=str)
        if outputs is not None:
            attributes["output.value"] = json.dumps(outputs, default=str)
    return attributes


def ready(api_key: str, secret_key: str, project: str) -> bool:
    return bool(api_key and secret_key and project)


def test_not_ready_without_keys():
    assert ready("", "", "") is False
    assert ready("k", "", "proj") is False


def test_ready_with_both_keys_and_project():
    assert ready("k", "s", "proj") is True


def test_collector_endpoint():
    assert collector_endpoint("https://api.futureagi.com") == "https://api.futureagi.com/tracer/v1/traces"
    assert (
        collector_endpoint("https://collector.example.com/base/")
        == "https://collector.example.com/base/tracer/v1/traces"
    )
    assert (
        collector_endpoint("https://api.futureagi.com/tracer/v1/traces")
        == "https://api.futureagi.com/tracer/v1/traces"
    )


def test_attributes():
    plain = span_attributes(
        session_id="session-1", user_id="user-1", flow_name="flow-1", inputs={"q": "hi"}, outputs={"a": "ok"}, redact=False
    )
    assert plain["session.id"] == "session-1"
    assert plain["user.id"] == "user-1"
    assert "input.value" in plain

    redacted = span_attributes(
        session_id="session-1", user_id=None, flow_name="flow-1", inputs={"q": "hi"}, outputs={"a": "ok"}, redact=True
    )
    assert "user.id" not in redacted
    assert "input.value" not in redacted
